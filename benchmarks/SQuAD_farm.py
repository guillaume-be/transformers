import time
from farm.infer import Inferencer
from transformers.data.processors.squad import SquadV2Processor
from pathlib import Path
import torch

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
    n_iter = 10

    loading_times = []
    feature_preparation_times = []
    forward_pass_times = []

    # ========================
    # Initialization
    # ========================

    for i in range(n_iter):
        t1 = time.time()
        model = Inferencer.load("bert-large-cased-whole-word-masking-finetuned-squad",
                                task_type="question_answering",
                                gpu=True,
                                disable_tqdm=True)
        del model
        t2 = time.time()
        loading_times.append(t2 - t1)
    model = Inferencer.load("bert-large-cased-whole-word-masking-finetuned-squad",
                            task_type="question_answering",
                            gpu=True,
                            disable_tqdm=True)
    # ========================
    # Pre-processing
    # ========================

    processor = SquadV2Processor()
    data = processor.get_dev_examples(root_path)[:1000]

    qa_inputs = [{"qas": [example.question_text], "context": example.context_text} for example in data]

    # ========================
    # Pipeline execution
    # ========================
    model_outputs = []
    for i in range(n_iter):
        t1 = time.time()
        num_batches = len(qa_inputs) // batch_size + 1
        outputs = []
        for batch_index in range(num_batches):
            batch = qa_inputs[batch_index * batch_size:min((batch_index + 1) * batch_size, len(data))]
            with torch.no_grad():
                model_outputs.extend(model.inference_from_dicts(batch))
        t2 = time.time()
        forward_pass_times.append(t2 - t1)

    print(f'Inference: {sum(forward_pass_times) / len(forward_pass_times)}s')
    print(len(model_outputs))
    print(f'Loading: {sum(loading_times) / len(loading_times)}s')
