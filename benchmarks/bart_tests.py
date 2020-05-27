import time

import torch

from transformers.tokenization_bart import BartTokenizer
from transformers.modeling_bart import BartForConditionalGeneration, BartModel

tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn').cuda()
# model.model.output_attentions = True
# model.model.output_hidden_states = True
model = model.eval()

TEXT_TO_SUMMARIZE = ["""In findings published Tuesday in Cornell University's arXiv by a team of scientists \
from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
a potentially habitable planet, but further observations will be required to say for sure. \" \
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b."""]

num_iterations = 3
generation_times = []
for _ in range(num_iterations):
    iteration_start = time.time()
    encoded_input = [tokenizer.encode(text, add_special_tokens=True) for text in TEXT_TO_SUMMARIZE]
    max_len = max([len(sentence) for sentence in encoded_input])
    encoded_input = [sequence + (max_len - len(sequence)) * [0] for sequence in encoded_input]
    encoded_input = torch.Tensor(encoded_input).long().cuda()
    with torch.no_grad():
        outputs = model.generate(encoded_input,
                                 max_length=142,
                                 num_beams=3,
                                 temperature=1.0,
                                 top_k=50,
                                 top_p=1.0,
                                 length_penalty=1.0,
                                 min_length=56,
                                 do_sample=False)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print(f'Iteration time : {time.time() - iteration_start}')
