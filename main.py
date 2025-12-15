#!/usr/bin/env python
# -*- coding:UTF-8 -*-

"""main.py: Simple forward-pass and debug script."""


### IMPORTS ###
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from data.synthetic_data import generate_data

from model.transformer import Transformer

from train.decode import greedy_decode
from train.label_smoothing import LabelSmoothing
from train.loss import SimpleLossCompute
from train.rate import rate
from train.train import run_epoch, load_trained_model

from utils import subsequent_mask, DummyScheduler, DummyOptimizer

### CONSTANTS ###
BATCH_SIZE_TRAIN = 80
BATCH_SIZE_EVAL = 5
MODE_TRAIN = "train"
MODE_EVAL = "eval"
NUM_BATCHES = 20
NUM_EPOCHS = 20
NUM_TESTS = 100
NUM_WARMUP_STEPS = 400
PADDING_IDX = 0
RATE_SCALING_FACTOR = 1.0
VOCAB_SIZE = 11

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
    test_model = make_model(VOCAB_SIZE, VOCAB_SIZE, 2)
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

def example_simple_model():
    criterion = LabelSmoothing(VOCAB_SIZE, PADDING_IDX, smoothing=0.0)
    model: Transformer = make_model(VOCAB_SIZE, VOCAB_SIZE, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model.src_embedding[0].d_model, factor=RATE_SCALING_FACTOR, warmup_steps=NUM_WARMUP_STEPS
        )
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        run_epoch(
            generate_data(VOCAB_SIZE, BATCH_SIZE_TRAIN, NUM_BATCHES),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode=MODE_TRAIN,
        )
        model.eval()
        run_epoch(
            generate_data(VOCAB_SIZE, BATCH_SIZE_EVAL, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode=MODE_EVAL,
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print("Decoding")
    print(greedy_decode(model, src, src_mask, max_len, start_symbol=0))

def run_tests():
    for _ in range(NUM_TESTS):
        inference_test()

def run_example_simple_model():
    example_simple_model()

if __name__ == '__main__':
    #run_example_simple_model()
    model = load_trained_model()