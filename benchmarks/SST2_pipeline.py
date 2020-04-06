import time

import torch

from transformers.data.processors.glue import Sst2Processor

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizerFast
from pathlib import Path

if __name__ == '__main__':

    t0 = time.time()

    # ========================
    # Settings
    # ========================

    root_path = Path('E:/Coding/data-resources/sst2/')
    vocab_path = Path('C:/Users/Guillaume/rustbert/distilbert_sst2/vocab.txt')
    batch_size = 64

    t1 = time.time()
    print(f'Set-up: {(t1 - t0) * 1000}ms')

    # ========================
    # Initialization
    # ========================

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(vocab_path))
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").cuda()
    t2 = time.time()
    print(f'initialization: {(t2 - t1)}s')

    # ========================
    # Pre-processing
    # ========================

    processor = Sst2Processor()
    data = processor.get_train_examples(root_path)
    data = [example.text_a for example in data]

    # ========================
    # Pipeline execution
    # ========================

    num_batches = len(data) // batch_size + 1
    outputs = []
    for batch_index in range(num_batches):
        batch = data[batch_index * batch_size:min((batch_index + 1) * batch_size, len(data))]
        inputs = tokenizer.batch_encode_plus(batch, return_tensors='pt')['input_ids'].cuda(0)
        with torch.no_grad():
            model_outputs = model(inputs)[0].detach().cpu()
        outputs.extend(model_outputs.softmax(dim=-1))

    print(len(outputs))
    t3 = time.time()
    print(f'Inference: {(t3 - t2)}s')
    print(f'Total: {(t3 - t0)}s')
