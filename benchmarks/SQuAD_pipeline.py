import time

from benchmarks.qa_utils import decode, generate_batch_indices, inputs_for_model
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline
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

    nlp_qa = pipeline('question-answering')

    t2 = time.time()
    print(f'initialization: {(t2 - t1)}s')

    # ========================
    # Pre-processing
    # ========================

    processor = SquadV2Processor()
    data = processor.get_dev_examples(root_path)

    # ========================
    # Pipeline execution
    # ========================

    nlp_qa(data)

    t3 = time.time()
    print(f'Inference: {(t3 - t2)}s')
    print(f'Total: {(t3 - t0)}s')
