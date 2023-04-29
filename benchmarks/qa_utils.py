from typing import Union, List, Dict, Tuple

import numpy as np


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
    topk = 1
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

    num_batches = len(features) // batch_size + 1
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
