import time

from benchmarks.qa_utils import decode, generate_batch_indices, inputs_for_model
from transformers import BertTokenizerFast, BertForQuestionAnswering, is_torch_available, is_tf_available, logger
from transformers.data.processors.squad import SquadV2Processor, squad_convert_example_to_features_init, SquadFeatures, \
    _new_check_is_max_context, MULTI_SEP_TOKENS_TOKENIZERS_SET, \
    _improve_answer_span
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

from transformers.tokenization_bert import whitespace_tokenize

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf


def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []
    truncated_query = tokenizer.tokenize(example.question_text)[:max_query_length]
    truncated_query_ids = tokenizer.convert_tokens_to_ids(truncated_query)
    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
            padding="max_length",
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query_ids) - sequence_pair_added_tokens,
            return_token_type_ids=True,
            is_pretokenized=True,
            add_special_tokens=True
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query_ids) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"][0]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][0][: encoded_dict["input_ids"][0].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"][0]) - 1 - encoded_dict["input_ids"][0][::-1].index(
                    tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][0][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"][0]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids[0])

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query_ids) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query_ids) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"][0].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"][0])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query_ids) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]): -(len(truncated_query_ids) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"][0] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"][0], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query_ids) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"][0],
                span["attention_mask"][0],
                span["token_type_ids"][0],
                cls_index,
                p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features


def squad_convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        return_dataset=False,
        tqdm_enabled=True,
):
    squad_convert_example_to_features_init(tokenizer)
    tokenizer._tokenizer.no_padding()
    # Defining helper methods
    features = []
    for example in examples:
        features.append(
            squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training))

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
            features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                if ex.token_type_ids is None:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )
                else:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        if "token_type_ids" in tokenizer.model_input_names:
            train_types = (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    "feature_index": tf.int64,
                    "qas_id": tf.string,
                },
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )
        else:
            train_types = (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "feature_index": tf.int64, "qas_id": tf.string},
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        return features


if __name__ == '__main__':

    t0 = time.time()

    # ========================
    # Settings
    # ========================

    root_path = Path('E:/Coding/data-resources/squad/')
    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64
    topk = 1
    max_answer_len = 15
    batch_size = 64
    n_iter = 1

    loading_times = []
    feature_preparation_times = []
    forward_pass_times = []

    # ========================
    # Initialization
    # ========================
    # for i in range(n_iter):
    #     t1 = time.time()
    #     tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
    #     model = BertForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad").cuda()
    #     t2 = time.time()
    #     loading_times.append(t2 - t1)
    tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
    model = BertForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad").cuda()
    # ========================
    # Pre-processing
    # ========================

    processor = SquadV2Processor()
    data = processor.get_dev_examples(root_path)[:1000]

    for i in range(n_iter):
        t1 = time.time()

        features = squad_convert_examples_to_features(
            examples=data,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
        )

        batch_indices = generate_batch_indices(features, batch_size)

    # for i in range(n_iter):
    #     t1 = time.time()
    #
    #     features = squad_convert_examples_to_features(
    #         examples=data,
    #         tokenizer=tokenizer,
    #         max_seq_length=max_seq_length,
    #         doc_stride=doc_stride,
    #         max_query_length=max_query_length,
    #         is_training=False,
    #     )
    #
    #     batch_indices = generate_batch_indices(features, batch_size)
    #
    #     # ========================
    #     # Forward pass
    #     # ========================
    #
    #     all_answers = []
    #     forward_time = []
    #     postprocessing_time = []
    #     for (start, end) in tqdm(batch_indices):
    #         t_batch = time.time()
    #         batch_features = features[start: end]
    #         fw_args = inputs_for_model([f.__dict__ for f in batch_features])
    #
    #         with torch.no_grad():
    #             # Retrieve the score for the context tokens only (removing question tokens)
    #             fw_args = {k: torch.tensor(v).cuda() for (k, v) in fw_args.items()}
    #             start, end = model(**fw_args)
    #             start, end = start.cpu().numpy(), end.cpu().numpy()
    #         t_forward = time.time()
    #         forward_time.append(t_forward - t_batch)
    #
    #         example_to_feature_ids = dict()
    #         for feature_index, feature in enumerate(batch_features):
    #             example_to_feature_ids[feature.example_index] = feature_index + 1
    #
    #         feature_id_start = 0
    #
    #         for (example_id, max_feature_id) in example_to_feature_ids.items():
    #             answers = []
    #             for example_feature_id in range(feature_id_start, max_feature_id):
    #                 feature = batch_features[example_feature_id]
    #                 start_ = start[example_feature_id]
    #                 end_ = end[example_feature_id]
    #                 example = data[example_id]
    #                 # Normalize logits and spans to retrieve the answer
    #                 start_ = np.exp(start_) / np.sum(np.exp(start_))
    #                 end_ = np.exp(end_) / np.sum(np.exp(end_))
    #
    #                 # Mask padding and question
    #                 start_, end_ = (
    #                     start_ * np.abs(np.array(feature.p_mask) - 1),
    #                     end_ * np.abs(np.array(feature.p_mask) - 1),
    #                 )
    #
    #                 # TODO : What happens if not possible
    #                 # Mask CLS
    #                 start_[0] = end_[0] = 0
    #
    #                 starts, ends, scores = decode(start_, end_, topk, max_answer_len)
    #                 char_to_word = np.array(example.char_to_word_offset)
    #                 # Convert the answer (tokens) back to the original text
    #                 answers += [
    #                     {
    #                         "score": score.item(),
    #                         "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
    #                         "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
    #                         "answer": " ".join(
    #                             example.doc_tokens[feature.token_to_orig_map[s]: feature.token_to_orig_map[e] + 1]
    #                         ),
    #                     }
    #                     for s, e, score in zip(starts, ends, scores)
    #                 ]
    #                 feature_id_start = max_feature_id
    #             answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: topk]
    #             all_answers += answers
    #         postprocessing_time.append(time.time() - t_forward)
    #     t2 = time.time()
    #     forward_pass_times.append(t2 - t1)

    # print(f'Inference: {sum(forward_pass_times) / len(forward_pass_times)}s')
    print(f'Feature generation: {sum(feature_preparation_times) / len(feature_preparation_times)}s')
    # print(f'Total: {sum(loading_times) / len(loading_times)}s')
