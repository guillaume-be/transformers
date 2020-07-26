import time
import torch
from transformers import TranslationPipeline, AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained('Helsinki-NLP/opus-mt-en-ROMANCE').cuda()
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ROMANCE')

# model = AutoModelWithLMHead.from_pretrained('t5-base').cuda()
# tokenizer = AutoTokenizer.from_pretrained('t5-base')
pipeline = TranslationPipeline(model=model, tokenizer=tokenizer, device=0)
pipeline.model.config.prefix = ">>fr<<"
# pipeline.model.config.prefix = "translate English to French: "


TEXTS_TO_TRANSLATE = [
    "In findings published Tuesday in Cornell University's arXiv by a team of scientists from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, a planet circling a star in the constellation Leo. ",
    "This is the first such discovery in a planet in its star's habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, used data from the NASA\'s Hubble telescope to assess changes in the light coming from K2-18b's star as the planet passed between it and Earth.",
    "They found that certain wavelengths of light, which are usually absorbed by water, weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere contains water in vapour form.",
    "The team from UCL then analyzed the Montreal team's data using their own software and confirmed their conclusion.",
    "This was not the first time scientists have found signs of water on an exoplanet, but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth.",
    "This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
    said UCL astronomer Angelos Tsiaras. \".",
    "It's the best candidate for habitability right now.\" \"It's a good sign\", said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors.",
    "Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being a potentially habitable planet, but further observations will be required to say for sure. \"",
    "K2-18b was first identified in 2015 by the Kepler space telescope.",
    "It is about 110 light-years from Earth and larger but less dense.",
    "Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year on K2-18b lasts 33 Earth days.",
    "According to The Guardian, astronomers were optimistic that NASA's James Webb space telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more about exoplanets like K2-18b."]

generation_times = []
n_iter = 1
loading_times = []
feature_preparation_times = []
forward_pass_times = []

# for i in range(n_iter):
#     t1 = time.time()
#     model = AutoModelWithLMHead.from_pretrained('Helsinki-NLP/opus-mt-en-ROMANCE').cuda()
#     tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ROMANCE')
#     pipeline = TranslationPipeline(model=model, tokenizer=tokenizer, device=0)
#     t2 = time.time()
#     loading_times.append(t2 - t1)
#
# for _ in range(n_iter):
#     t1 = time.time()
#     with torch.no_grad():
#         outputs = pipeline._parse_and_tokenize(TEXTS_TO_TRANSLATE, padding=True)
#         t2 = time.time()
#         feature_preparation_times.append(t2 - t1)

for _ in range(n_iter):
    t1 = time.time()
    with torch.no_grad():
        outputs = pipeline(TEXTS_TO_TRANSLATE, num_beams=6, early_stopping=True, max_length=512)
        t2 = time.time()
        forward_pass_times.append(t2 - t1)
        
# print(f'Loading: {sum(loading_times) / len(loading_times)}s')
# print(f'Tokenization: {sum(feature_preparation_times) / len(feature_preparation_times)}s')
print(f'Inference: {sum(forward_pass_times) / len(forward_pass_times)}s')
for output in outputs:
    print(output)

