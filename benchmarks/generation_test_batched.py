import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

start = time.time()
model = AutoModelForCausalLM.from_pretrained('gpt2').cuda()
tokenizer = AutoTokenizer.from_pretrained('gpt2')

initialization = time.time()
print(f'Initialization: {initialization - start}')

num_iterations = 10
generation_times = []

texts = [
    "Hello, I'm a language model,",
    "Sartre once said: ",
    "The majority of crustaceans are aquatic,",
]

for _ in range(num_iterations):
    iteration_start = time.time()

    encoded_input = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    attention_masks = [[1] * len(sentence) for sentence in encoded_input]
    max_len = max([len(sentence) for sentence in encoded_input])
    encoded_input = [(max_len - len(sequence)) * [tokenizer.eos_token_id] + sequence for sequence in encoded_input]
    attention_masks = [(max_len - len(mask)) * [0] + mask for mask in attention_masks]
    encoded_input = torch.Tensor(encoded_input).long().cuda()
    attention_mask = torch.Tensor(attention_masks).long().cuda()

    with torch.no_grad():
        outputs = model.generate(encoded_input,
                                 min_length=64,
                                 max_length=64,
                                 num_return_sequences=5,
                                 num_beams=5,
                                 do_sample=False,
                                 # do_sample=True,
                                 temperature=1.0,
                                 top_k=0,
                                 top_p=0.9,
                                 no_repeat_ngram_size=3,
                                 early_stopping=False,
                                 attention_mask=attention_mask
                                 )

        iteration_elapsed = time.time() - iteration_start
        generation_times.append(iteration_elapsed)
        print(f'Elapsed: {iteration_elapsed}')

print(f'Average iteration time: {sum(generation_times) / len(generation_times)}')
for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True))
