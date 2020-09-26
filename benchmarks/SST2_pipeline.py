import math
import time

import torch

from transformers.data.processors.glue import Sst2Processor

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizerFast
from pathlib import Path


def stdev(series):
    mean = sum(forward_pass_times) / len(forward_pass_times)
    return math.sqrt(
        sum([(forward - mean) ** 2.0 for forward in series])
        / len(forward_pass_times))


if __name__ == '__main__':

    t0 = time.time()

    # ========================
    # Settings
    # ========================

    root_path = Path('E:/Coding/data-resources/sst2/')
    vocab_path = Path('E:/Coding/cache/rustbert/distilbert-sst2/vocab.txt')
    batch_size = 64
    n_iter = 10

    loading_times = []
    feature_preparation_times = []
    forward_pass_times = []

    # ========================
    # Initialization
    # ========================

    for i in range(n_iter):
        t1 = time.time()
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english").cuda()
        t2 = time.time()
        loading_times.append(t2 - t1)

    # ========================
    # Pre-processing
    # ========================

    processor = Sst2Processor()
    data = processor.get_train_examples(root_path)
    data = [example.text_a for example in data[:2000]]

    for i in range(n_iter):
        t1 = time.time()
        num_batches = len(data) // batch_size + 1
        outputs = []
        for batch_index in range(num_batches):
            batch = data[batch_index * batch_size:min((batch_index + 1) * batch_size, len(data))]
            inputs = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True)['input_ids'].cuda(0)
        t2 = time.time()
        feature_preparation_times.append(t2 - t1)

    # ========================
    # Pipeline execution
    # ========================
    for i in range(n_iter):
        t1 = time.time()
        num_batches = len(data) // batch_size + 1
        outputs = []
        for batch_index in range(num_batches):
            batch = data[batch_index * batch_size:min((batch_index + 1) * batch_size, len(data))]
            inputs = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True)['input_ids'].cuda(0)
            with torch.no_grad():
                model_outputs = model(inputs)[0].detach().cpu()
            outputs.extend(model_outputs.softmax(dim=-1))
        t2 = time.time()
        forward_pass_times.append(t2 - t1)

    print(f'Inference: {sum(forward_pass_times) / len(forward_pass_times)}s')
    print(f'Stddev: {stdev(forward_pass_times)}s')
    print(f'Loading: {sum(loading_times) / len(loading_times)}s')
    print(f'Stddev: {stdev(loading_times)}s')
