import time
from typing import List

import torch
from torch import nn
from transformers.data.processors.glue import Sst2Processor
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, DistilBertTokenizer, \
    PreTrainedTokenizer
from pathlib import Path

# Run profiling:
# python -mcProfile -o sst2.prof classification_profile.py
# tuna sst2.prof


def load_model(model_name: str) -> (DistilBertForSequenceClassification, DistilBertTokenizerFast):
    _tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    _model = DistilBertForSequenceClassification.from_pretrained(model_name).cuda()
    return _model, _tokenizer


def load_model_fast(model_name: str) -> (DistilBertForSequenceClassification, DistilBertTokenizerFast):
    _tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    _model = DistilBertForSequenceClassification.from_pretrained(model_name).cuda()
    return _model, _tokenizer


def run_benchmark(_n_iter: int, _data: List, _tokenizer: PreTrainedTokenizer, _model: nn.Module):
    for i in range(_n_iter):
        t1 = time.time()
        _num_batches = len(_data) // batch_size + 1
        _outputs = []
        for _batch_index in range(_num_batches):
            t_tokenization = time.time()
            _batch = _data[_batch_index * batch_size:min((_batch_index + 1) * batch_size, len(_data))]
            _inputs = _tokenizer.batch_encode_plus(_batch, return_tensors='pt', padding=True)['input_ids'].cuda(0)
            feature_preparation_times.append(time.time() - t_tokenization)
            t_forward = time.time()
            with torch.no_grad():
                _model_outputs = _model(_inputs)[0].detach().cpu()
            torch.cuda.synchronize()
            _outputs.extend(_model_outputs.softmax(dim=-1))
            forward_pass_times.append(time.time() - t_forward)
        total_times.append(time.time() - t1)

    print(f'Forward pass: {sum(forward_pass_times) / len(total_times)}s')
    print(f'Feature generation: {sum(feature_preparation_times) / len(total_times)}s')
    print(f'Total: {sum(total_times) / len(total_times)}s')


if __name__ == '__main__':

    # ========================
    # Settings
    # ========================

    root_path = Path('E:/Coding/data-resources/sst2/')
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
        "distilbert-base-uncased-finetuned-sst-2-english") if not fast_tokenizer else load_model_fast(
        "distilbert-base-uncased-finetuned-sst-2-english")

    # ========================
    # Warmup
    # ========================
    processor = Sst2Processor()
    data = processor.get_train_examples(root_path)
    data = [example.text_a for example in data[:1000]]

    num_batches = len(data) // batch_size + 1
    outputs = []
    for batch_index in range(num_batches):
        batch = data[batch_index * batch_size:min((batch_index + 1) * batch_size, len(data))]
        inputs = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True)['input_ids'].cuda(0)
        with torch.no_grad():
            model_outputs = model(inputs)[0].detach().cpu()
        outputs.extend(model_outputs.softmax(dim=-1))

    # ========================
    # Pipeline execution
    # ========================
    run_benchmark(n_iter, data, tokenizer, model)
