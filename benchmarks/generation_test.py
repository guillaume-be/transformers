import time

import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

start = time.time()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
model = model.eval()

initialization = time.time()
print(f'Initialization: {initialization-start}')

PROMPTS = ["The dog "]

encoded_input = [tokenizer.encode(text, add_special_tokens=False) for text in PROMPTS]
max_len = max([len(sentence) for sentence in encoded_input])
encoded_input = [sequence + (max_len - len(sequence)) * [0] for sequence in encoded_input]
encoded_input = torch.Tensor(encoded_input).long().cuda()

num_iterations = 3
generation_times = []
for _ in range(num_iterations):
    iteration_start = time.time()
    outputs = model.generate(encoded_input,
                             num_beams=5,
                             num_return_sequences=3,
                             do_sample=True,
                             max_length=150,
                             temperature=1.1,
                             no_repeat_ngram_size=3)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    iteration_elapsed = time.time() - iteration_start
    generation_times.append(iteration_elapsed)
    print(f'Elapsed: {iteration_elapsed}')

print(f'Average iteration time: {sum(generation_times)/len(generation_times)}')
