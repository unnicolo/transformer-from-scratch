#!/bin/env python
# -*- coding: UTF-8 -*-

"""
train.py: Script for training a model.
"""

### IMPORTS ###
import time
import torch
from torch.optim import Optimizer
from typing import Generator

from train.state import TrainState
from data.batch import Batch
from model.transformer import Transformer

def run_epoch(
    data_iter: Generator[Batch, None, None],
    model: Transformer,
    loss_compute,
    optimizer: Optimizer,
    scheduler: Optimizer,
    mode='train',
    accum_iter = 1,
    train_state=TrainState()
):
    """Train the model for a single epoch.
    
    Args:
        data_iter: Data-iterator that holds the training data
        model: The model to be trained
        loss_compute: Loss computation 
        optimizer: The optimizer being used 
        scheduler: The learning rate scheduler being used
        mode: Current training mode; either `train` or `train+log`
        accum_iter: The number of passes to perform after which we update the parameters
        train_state: The current state of the training process
    """
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    num_accumulated = 0
    for i, batch in enumerate(data_iter):
        # Step 1: Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Step 2: Forward-pass the training data through the model
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )

        # Step 3: Compute the loss
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.num_tokens)

        if mode == 'train' or mode == 'train+log':
            # Step 4: Backpropagate the gradients
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                # Update the parameters
                optimizer.step()
                num_accumulated += 1
                train_state.accum_step += 1
            scheduler.step()