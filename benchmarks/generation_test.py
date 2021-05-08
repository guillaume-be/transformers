import time
from transformers import pipeline

start = time.time()
generator = pipeline('text-generation', model='gpt2', device=0)
initialization = time.time()
print(f'Initialization: {initialization - start}')

num_iterations = 10
generation_times = []
for _ in range(num_iterations):
    iteration_start = time.time()
    outputs = generator([
        "Hello, I'm a language model,",
        "Once upon a time, ",
        "The majority of crustaceans are aquatic,",
    ],
        min_length=64,
        max_length=64,
        num_return_sequences=5,
        num_beams=5,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    iteration_elapsed = time.time() - iteration_start
    generation_times.append(iteration_elapsed)
    print(f'Elapsed: {iteration_elapsed}')

print(f'Average iteration time: {sum(generation_times) / len(generation_times)}')
