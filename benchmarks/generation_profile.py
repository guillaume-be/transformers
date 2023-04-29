import time
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast, \
    PreTrainedTokenizer, PreTrainedTokenizerFast

# Select Fast or Slow tokenizer
# Run profiling:
# python -mcProfile -o sst2.prof question_answering_profile.py
# tuna sst2.prof


texts = [
    "Hello, I'm a language model,",
    "Sartre once said: ",
    "The majority of crustaceans are aquatic,",
]


def load_model(model_name: str) -> (GPT2LMHeadModel, GPT2Tokenizer):
    _tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    _model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    return _model, _tokenizer


def load_model_fast(model_name: str) -> (GPT2LMHeadModel, GPT2TokenizerFast):
    _tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    _model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    return _model, _tokenizer


def prepare_input(_texts: List[str]) -> Tuple[Tensor, Tensor]:
    _encoded_input = [tokenizer.encode(text, add_special_tokens=True) for text in _texts]
    _attention_masks = [[1] * len(sentence) for sentence in _encoded_input]
    _max_len = max([len(sentence) for sentence in _encoded_input])
    _encoded_input = [(_max_len - len(sequence)) * [tokenizer.eos_token_id] + sequence for sequence in _encoded_input]
    _attention_masks = [(_max_len - len(mask)) * [0] + mask for mask in _attention_masks]
    _encoded_input = torch.Tensor(_encoded_input).long().cuda()
    _attention_mask = torch.Tensor(_attention_masks).long().cuda()
    return _encoded_input, _attention_mask


def run_benchmark(_n_iter: int,
                  _data: List[str],
                  _tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                  _model: nn.Module,
                  _min_length: int,
                  _max_length: int,
                  _num_return_sequences: int,
                  _num_beams: int,
                  _do_sample: bool,
                  _top_k: int,
                  _top_p: float,
                  _no_repeat_ngram_size: int,
                  _early_stopping: bool):
    forward_times = []
    postprocessing_times = []

    for i in range(_n_iter):
        t1 = time.time()
        _all_answers = []
        t_tokenization = time.time()
        (input_ids, mask) = prepare_input(_data)
        feature_preparation_times.append(time.time() - t_tokenization)

        with torch.no_grad():
            outputs, forward_t, postprocessing_t = model.generate(input_ids,
                                                                  min_length=_min_length,
                                                                  max_length=_max_length,
                                                                  num_return_sequences=_num_return_sequences,
                                                                  num_beams=_num_beams,
                                                                  do_sample=_do_sample,
                                                                  top_k=_top_k,
                                                                  top_p=_top_p,
                                                                  no_repeat_ngram_size=_no_repeat_ngram_size,
                                                                  early_stopping=_early_stopping,
                                                                  attention_mask=mask
                                                                  )
            forward_times.extend(forward_t)
            postprocessing_times.extend(postprocessing_t)
        total_times.append(time.time() - t1)

        for output_sequence in outputs:
            tokenizer.decode(output_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print(f'Forward pass: {sum(forward_times) / len(total_times)}s')
    print(f'Feature generation: {sum(feature_preparation_times) / len(total_times)}s')
    print(f'Post-processing  {sum(postprocessing_times) / len(total_times)}s')
    print(f'Total: {sum(total_times) / len(total_times)}s')


if __name__ == '__main__':
    # ========================
    # Settings
    # ========================

    min_length = 64
    max_length = 64
    num_return_sequences = 5
    num_beams = 5
    do_sample = True
    top_k = 50
    top_p = 0.9
    no_repeat_ngram_size = 3
    early_stopping = True
    n_iter = 10
    fast_tokenizer = True
    total_times = []
    feature_preparation_times = []
    forward_pass_times = []

    # ========================
    # Initialization
    # ========================
    (model, tokenizer) = load_model("gpt2") if not fast_tokenizer else load_model_fast("gpt2")

    # ========================
    # Warmup
    # ========================
    encoded_input = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    attention_masks = [[1] * len(sentence) for sentence in encoded_input]
    max_len = max([len(sentence) for sentence in encoded_input])
    encoded_input = [(max_len - len(sequence)) * [tokenizer.eos_token_id] + sequence for sequence in encoded_input]
    attention_masks = [(max_len - len(mask)) * [0] + mask for mask in attention_masks]
    encoded_input = torch.Tensor(encoded_input).long().cuda()
    attention_mask = torch.Tensor(attention_masks).long().cuda()

    with torch.no_grad():
        outputs, _, _ = model.generate(encoded_input,
                                       min_length=min_length,
                                       max_length=max_length,
                                       num_return_sequences=num_return_sequences,
                                       num_beams=num_beams,
                                       do_sample=do_sample,
                                       top_k=top_k,
                                       top_p=top_p,
                                       no_repeat_ngram_size=no_repeat_ngram_size,
                                       early_stopping=early_stopping,
                                       attention_mask=attention_mask
                                       )

    for output_sequence in outputs:
        tokenizer.decode(output_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # ========================
    # Pipeline execution
    # ========================
    run_benchmark(n_iter, texts, tokenizer, model, min_length,
                  max_length, num_return_sequences, num_beams,
                  do_sample, top_k, top_p, no_repeat_ngram_size, early_stopping)
