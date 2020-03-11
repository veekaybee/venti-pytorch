"""
    modified by @vboykis
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
"""
import os
import sys
import torch
import random
import numpy as np
from PIL import Image

import quotes
import json

from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

# Load Model
state_dict = torch.load(
    "gpt2-pytorch_model.bin",
    map_location="cpu" if not torch.cuda.is_available() else None,
)
enc = get_encoder()
config = GPT2Config()
model = GPT2LMHeadModel(config)
model = load_weight(model, state_dict)


def text_generator(model, text):
    nsamples = 1
    batch_size = -1
    length = 200
    temperature = .7
    top_k = 40
    unconditional = False

    if batch_size == -1:
        batch_size = 1
    assert nsamples % batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.to(device)
    model.eval()

    if length == -1:
        length = config.n_ctx // 2
    elif length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    context_tokens = enc.encode(text)

    generated = 0
    for _ in range(nsamples // batch_size):
        out = sample_sequence(
            model=model,
            length=length,
            context=context_tokens if not unconditional else None,
            start_token=enc.encoder["<|endoftext|>"] if unconditional else None,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        out = out[:, len(context_tokens):].tolist()

        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            return (text)


# Generate n random snippest of text to JSON file
with open("generated_phrases.json", "w") as outfile:
    quotes_dict = {}

    for i in range(50):
        input_phrase = random.choice(quotes.quote_list)
        text = text_generator(model,input_phrase)
        phrase_number = f'phrase_{i}'
        quotes_dict[phrase_number] = f'{input_phrase} {text}'

    json.dump(quotes_dict, outfile)

