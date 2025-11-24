#!/usr/bin/env python
# -*- coding:UTF-8 -*-

"""main.py: Simple forward-pass and debug script."""


### IMPORTS ###
import torch
import torch.nn as nn

from model.transformer import Transformer
from model.utils import subsequent_mask

### CONSTANTS ###
NUM_TESTS = 100

def make_model(
    src_vocab, 
    tgt_vocab, 
    N=6, 
    d_model=512, 
    d_ff=2048, 
    h=8, 
    dropout=0.1
):
    model = Transformer(d_model, d_ff, h, dropout, src_vocab, tgt_vocab)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def inference_test():
    ### In the following: We try to make the model memorize the numbers from 1 to 10 ###
    test_model = make_model(11, 11, 2)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model(
            src, ys, src_mask, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # Generate output probabilities from decoder output
        probabilities = test_model.generate(out[:, -1])
        _, next_word  = torch.max(probabilities, dim=1)
        next_word = next_word.data[0]

        # Append the new_word to the output and continue feeding into the transformer.
        next_token = torch.tensor([[next_word]], dtype=src.dtype, device=src.device)
        ys = torch.cat([ys, next_token], dim=1)

    print(f'Untrained example model prediction: {ys}')

def run_tests():
    for _ in range(NUM_TESTS):
        inference_test()

if __name__ == '__main__':
    run_tests()
