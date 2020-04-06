import multiprocessing
import time

from benchmarks.qa_utils import decode, generate_batch_indices, inputs_for_model
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers.data.processors.squad import squad_convert_examples_to_features, SquadV2Processor
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

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

    t1 = time.time()
    print(f'Set-up: {(t1 - t0) * 1000}ms')

    # ========================
    # Initialization
    # ========================

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad").cuda()

    t2 = time.time()
    print(f'initialization: {(t2 - t1)}s')

    # ========================
    # Pre-processing
    # ========================

    processor = SquadV2Processor()
    data = processor.get_dev_examples(root_path)

    t3 = time.time()
    print(f'Loading examples: {(t3 - t2)}s')

    features = squad_convert_examples_to_features(
        examples=data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        threads=multiprocessing.cpu_count()
    )

    batch_indices = generate_batch_indices(features, batch_size)

    t4 = time.time()
    print(f'Pre-processing: {(t4 - t3)}s')

    # ========================
    # Forward pass
    # ========================

    all_answers = []
    forward_time = []
    postprocessing_time = []
    for (start, end) in tqdm(batch_indices):
        t_batch = time.time()
        batch_features = features[start: end]
        fw_args = inputs_for_model([f.__dict__ for f in batch_features])

        with torch.no_grad():
            # Retrieve the score for the context tokens only (removing question tokens)
            fw_args = {k: torch.tensor(v).cuda() for (k, v) in fw_args.items()}
            start, end = model(**fw_args)
            start, end = start.cpu().numpy(), end.cpu().numpy()
        t_forward = time.time()
        forward_time.append(t_forward - t_batch)

        example_to_feature_ids = dict()
        for feature_index, feature in enumerate(batch_features):
            example_to_feature_ids[feature.example_index] = feature_index + 1

        feature_id_start = 0

        for (example_id, max_feature_id) in example_to_feature_ids.items():
            answers = []
            for example_feature_id in range(feature_id_start, max_feature_id):
                feature = batch_features[example_feature_id]
                start_ = start[example_feature_id]
                end_ = end[example_feature_id]
                example = data[example_id]
                # Normalize logits and spans to retrieve the answer
                start_ = np.exp(start_) / np.sum(np.exp(start_))
                end_ = np.exp(end_) / np.sum(np.exp(end_))

                # Mask padding and question
                start_, end_ = (
                    start_ * np.abs(np.array(feature.p_mask) - 1),
                    end_ * np.abs(np.array(feature.p_mask) - 1),
                )

                # TODO : What happens if not possible
                # Mask CLS
                start_[0] = end_[0] = 0

                starts, ends, scores = decode(start_, end_, topk, max_answer_len)
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
                    for s, e, score in zip(starts, ends, scores)
                ]
                feature_id_start = max_feature_id
            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: topk]
            all_answers += answers
        postprocessing_time.append(time.time() - t_forward)

    t5 = time.time()
    print(f'Inference: {(t5 - t4)}s')
    print(f'Inference (forward): {sum(forward_time)}s')
    print(f'Inference (post-processing): {sum(postprocessing_time)}s')
    print(f'Total: {(t5 - t0)}s')
