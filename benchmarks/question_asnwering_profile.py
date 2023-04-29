import time
from typing import Union, Dict, List, Tuple

from torch import nn, Tensor

from transformers import BertTokenizer, BertForQuestionAnswering, BertTokenizerFast, PreTrainedTokenizer, \
    PreTrainedTokenizerFast
from transformers.data.processors.squad import squad_convert_examples_to_features, SquadV2Processor, SquadFeatures
from pathlib import Path
import torch
import numpy as np


# Select Fast or Slow tokenizer
# Run profiling:
# python -mcProfile -o sst2.prof question_answering_profile.py
# tuna sst2.prof


def load_model(model_name: str) -> (BertForQuestionAnswering, BertTokenizer):
    _tokenizer = BertTokenizer.from_pretrained(model_name)
    _model = BertForQuestionAnswering.from_pretrained(model_name).cuda()
    return _model, _tokenizer


def load_model_fast(model_name: str) -> (BertForQuestionAnswering, BertTokenizerFast):
    _tokenizer = BertTokenizerFast.from_pretrained(model_name)
    _model = BertForQuestionAnswering.from_pretrained(model_name).cuda()
    return _model, _tokenizer


def inputs_for_model(features: Union[dict, List[dict]]) -> Dict:
    args = ["input_ids", "attention_mask"]

    if isinstance(features, dict):
        return {k: features[k] for k in args}
    else:
        return {k: [feature[k] for feature in features] for k in args}


def decode(start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
    """
    Take the output of any QuestionAnswering head and will generate probalities for each span to be
    the actual answer.
    In addition, it filters out some unwanted/impossible cases like answer len being greater than
    max_answer_len or answer end position being before the starting position.
    The method supports output the k-best answer through the topk argument.
    Args:
        start: numpy array, holding individual start probabilities for each token
        end: numpy array, holding individual end probabilities for each token
        topk: int, indicates how many possible answer span(s) to extract from the model's output
        max_answer_len: int, maximum size of the answer to extract from the model's output
    """
    #     Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)
    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]
    start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
    return start, end, candidates[0, start, end]


def generate_batch_indices(features, batch_size):
    example_features_length = dict()
    for feature in features:
        if feature.example_index in example_features_length:
            example_features_length[feature.example_index] += 1
        else:
            example_features_length[feature.example_index] = 1

    batch_indices = []
    batch_length = 0
    start = 0
    end = 0
    for idx in range(len(example_features_length)):
        if batch_length + example_features_length[idx] <= batch_size:
            end += example_features_length[idx]
            batch_length += example_features_length[idx]
        else:
            batch_indices.append((start, end))
            start = end
            end += example_features_length[idx]
            batch_length = 1
    batch_indices.append((start, end))
    return batch_indices


def prepare_features(_data: List, _max_seq_length: int, _doc_stride: int, _max_query_length: int, _batch_size: int,
                     _tokenizer: PreTrainedTokenizer):
    features = squad_convert_examples_to_features(
        examples=_data,
        tokenizer=_tokenizer,
        max_seq_length=_max_seq_length,
        doc_stride=_doc_stride,
        max_query_length=_max_query_length,
        is_training=False,
        threads=4
    )

    batch_indices = generate_batch_indices(features, _batch_size)
    return features, batch_indices


def prepare_features_fast(_data: List, _max_seq_length: int, _doc_stride: int, _max_query_length: int, _batch_size: int,
                          _tokenizer: PreTrainedTokenizerFast):
    features_list = []
    for example_idx, example in enumerate(_data):
        # Define the side we want to truncate / pad and the text/pair sorting
        question_first = bool(_tokenizer.padding_side == "right")

        encoded_inputs = _tokenizer(
            text=example.question_text if question_first else example.context_text,
            text_pair=example.context_text if question_first else example.question_text,
            padding="longest",
            truncation="only_second" if question_first else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_tensors="np",
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        # When the input is too long, it's converted in a batch of inputs with overflowing tokens
        # and a stride of overlap between the inputs. If a batch of inputs is given, a special output
        # "overflow_to_sample_mapping" indicate which member of the encoded batch belong to which original batch sample.
        # Here we tokenize examples one-by-one so we don't need to use "overflow_to_sample_mapping".
        # "num_span" is the number of output samples generated from the overflowing tokens.
        num_spans = len(encoded_inputs["input_ids"])

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
        p_mask = np.asarray(
            [
                [tok != 1 if question_first else 0 for tok in encoded_inputs.sequence_ids(span_id)]
                for span_id in range(num_spans)
            ]
        )

        # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
        if _tokenizer.cls_token_id is not None:
            cls_index = np.nonzero(encoded_inputs["input_ids"] == _tokenizer.cls_token_id)
            p_mask[cls_index] = 0

        features = []
        for span_idx in range(num_spans):
            features.append(
                SquadFeatures(
                    input_ids=encoded_inputs["input_ids"][span_idx],
                    attention_mask=encoded_inputs["attention_mask"][span_idx],
                    token_type_ids=encoded_inputs["token_type_ids"][span_idx],
                    p_mask=p_mask[span_idx].tolist(),
                    encoding=encoded_inputs[span_idx],
                    # We don't use the rest of the values - and actually
                    # for Fast tokenizer we could totally avoid using SquadFeatures and SquadExample
                    cls_index=None,
                    token_to_orig_map={},
                    example_index=example_idx,
                    unique_id=0,
                    paragraph_len=0,
                    token_is_max_context=0,
                    tokens=[],
                    start_position=0,
                    end_position=0,
                    is_impossible=False,
                    qas_id=None,
                )
            )
        features_list.extend(features)

    batch_indices = generate_batch_indices(features_list, _batch_size)
    return features_list, batch_indices


def post_process(_batch_features: List, _starts: Tensor, _ends: Tensor, _tokenizer: PreTrainedTokenizer) -> List:
    topk = 1
    example_to_feature_ids = dict()
    for feature_index, feature in enumerate(_batch_features):
        example_to_feature_ids[feature.example_index] = feature_index + 1

    feature_id_start = 0
    _all_answers = []

    for (example_id, max_feature_id) in example_to_feature_ids.items():
        answers = []
        for example_feature_id in range(feature_id_start, max_feature_id):
            feature = _batch_features[example_feature_id]
            start_ = _starts[example_feature_id]
            end_ = _ends[example_feature_id]
            example = data[example_id]
            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(start_) / np.sum(np.exp(start_))
            end_ = np.exp(end_) / np.sum(np.exp(end_))
            if _tokenizer.padding_side == 'right':
                p_mask = feature.p_mask + [1] * (len(start_) - len(feature.p_mask))
            else:
                p_mask = [1] * (len(start_) - len(feature.p_mask)) + feature.p_mask

            # Mask padding and question
            start_, end_ = (
                start_ * np.abs(np.array(p_mask) - 1),
                end_ * np.abs(np.array(p_mask) - 1),
            )

            # TODO : What happens if not possible
            # Mask CLS
            start_[0] = end_[0] = 0

            start_pos, end_pos, scores = decode(start_, end_, topk, max_answer_len)

            if not _tokenizer.is_fast:
                char_to_word = np.array(example.char_to_word_offset)
                # Convert the answer (tokens) back to the original text
                answers += [
                    {
                        "score": score.item(),
                        "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                        "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                        "answer": " ".join(
                            example.doc_tokens[feature.token_to_orig_map[s]: feature.token_to_orig_map[e] + 1]
                        ),
                    }
                    for s, e, score in zip(start_pos, end_pos, scores)
                ]
            else:
                question_first = bool(_tokenizer.padding_side == "right")
                enc = feature.encoding

                # Sometimes the max probability token is in the middle of a word so:
                # - we start by finding the right word containing the token with `token_to_word`
                # - then we convert this word in a character span with `word_to_chars`
                answers += [
                    {
                        "score": score.item(),
                        "start": enc.word_to_chars(
                            enc.token_to_word(s), sequence_index=1 if question_first else 0
                        )[0],
                        "end": enc.word_to_chars(enc.token_to_word(e), sequence_index=1 if question_first else 0)[
                            1
                        ],
                        "answer": example.context_text[
                                  enc.word_to_chars(enc.token_to_word(s), sequence_index=1 if question_first else 0)[
                                      0
                                  ]: enc.word_to_chars(enc.token_to_word(e), sequence_index=1 if question_first else 0)[
                                      1
                                  ]
                                  ],
                    }
                    for s, e, score in zip(start_pos, end_pos, scores)
                ]
            feature_id_start = max_feature_id
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: topk]
        _all_answers += answers

    return _all_answers


def run_benchmark(_n_iter: int, _data: List, _tokenizer: PreTrainedTokenizer, _model: nn.Module):
    forward_times = []
    postprocessing_time = []

    for i in range(_n_iter):
        t1 = time.time()
        _all_answers = []
        t_tokenization = time.time()
        if isinstance(_tokenizer, BertTokenizerFast):
            _features, _batch_indices = prepare_features_fast(
                data, max_seq_length, doc_stride, max_query_length, batch_size,
                tokenizer)
        else:
            _features, _batch_indices = prepare_features(
                data, max_seq_length, doc_stride, max_query_length, batch_size,
                tokenizer)

        feature_preparation_times.append(time.time() - t_tokenization)

        for (_start, _end) in _batch_indices:
            _t_batch = time.time()
            _batch_features = _features[_start: _end]
            _fw_args = tokenizer.pad(inputs_for_model([f.__dict__ for f in _batch_features]))

            with torch.no_grad():
                # Retrieve the score for the context tokens only (removing question tokens)
                _fw_args = {k: torch.tensor(v).cuda() for (k, v) in _fw_args.items()}
                _model_outputs = model(**_fw_args)
                _starts, _ends = _model_outputs['start_logits'].cpu().numpy(), _model_outputs[
                    'end_logits'].cpu().numpy()
            torch.cuda.synchronize()
            t_forward = time.time()
            forward_times.append(t_forward - _t_batch)
            _all_answers.extend(post_process(_batch_features, _starts, _ends, _tokenizer))
            postprocessing_time.append(time.time() - t_forward)

        total_times.append(time.time() - t1)

    print(f'Forward pass: {sum(forward_times) / len(total_times)}s')
    print(f'Feature generation: {sum(feature_preparation_times) / len(total_times)}s')
    print(f'Post-processing  {sum(postprocessing_time) / len(total_times)}s')
    print(f'Total: {sum(total_times) / len(total_times)}s')


if __name__ == '__main__':
    # ========================
    # Settings
    # ========================

    root_path = Path('E:/Coding/data-resources/squad/')
    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64
    max_answer_len = 15
    batch_size = 64
    n_iter = 10
    fast_tokenizer = True
    total_times = []
    feature_preparation_times = []
    forward_pass_times = []

    # ========================
    # Initialization
    # ========================
    (model, tokenizer) = load_model(
        "bert-large-cased-whole-word-masking-finetuned-squad") if not fast_tokenizer else load_model_fast(
        "bert-large-cased-whole-word-masking-finetuned-squad")

    # ========================
    # Warmup
    # ========================
    processor = SquadV2Processor()
    data = processor.get_dev_examples(root_path)[:1000]

    all_answers = []
    if isinstance(tokenizer, BertTokenizerFast):
        features, batch_indices = prepare_features_fast(
            data, max_seq_length, doc_stride, max_query_length, batch_size,
            tokenizer)
    else:
        features, batch_indices = prepare_features(
            data, max_seq_length, doc_stride, max_query_length, batch_size,
            tokenizer)

    for (start, end) in batch_indices:
        t_batch = time.time()
        batch_features = features[start: end]
        fw_args = tokenizer.pad(inputs_for_model([f.__dict__ for f in batch_features]))

        with torch.no_grad():
            # Retrieve the score for the context tokens only (removing question tokens)
            fw_args = {k: torch.tensor(v).cuda() for (k, v) in fw_args.items()}
            model_outputs = model(**fw_args)
            starts, ends = model_outputs['start_logits'].cpu().numpy(), model_outputs['end_logits'].cpu().numpy()
        all_answers.extend(post_process(batch_features, starts, ends, tokenizer))

    # ========================
    # Pipeline execution
    # ========================
    run_benchmark(n_iter, data, tokenizer, model)
