# -*- coding: utf-8 -*-

"""decode.py: Various decoding procedures - for now just greedy decoding."""

### IMPORTS ###
import torch

from model.transformer import Transformer
from utils import subsequent_mask

### CONSTANTS ###

def greedy_decode(model: Transformer, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src))
        probabilities = model.generator(out[:, -1])
        _, next_word = torch.max(probabilities, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.zeros(1, 1).fill_(next_word).type_as(src)], dim=1)
    
    return ys