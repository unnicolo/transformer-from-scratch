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

### CONSTANTS ###
STEPS_UNTIL_LOG_WITHIN_EPOCH = 40 # number of steps within an epoch until we log metrics
MODE_TRAIN = 'train'
MODE_TRAIN_AND_LOG = 'train+log'
LEARNING_RATE_PARAM_GROUP_SPECIFIER = 'lr'

def run_epoch(
    data_iter: Generator[Batch, None, None],
    model: Transformer,
    loss_compute,
    optimizer: Optimizer,
    scheduler: Optimizer,
    mode: str = MODE_TRAIN,
    accum_iter: int = 1,
    train_state: TrainState =TrainState()
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
    
    Returns:
        (Tuple): The overall loss and the final train state.
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

        if mode == MODE_TRAIN or mode == MODE_TRAIN_AND_LOG:
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

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % STEPS_UNTIL_LOG_WITHIN_EPOCH == 1 and (mode == MODE_TRAIN or mode == MODE_TRAIN_AND_LOG):
            lr = optimizer.param_groups[0][LEARNING_RATE_PARAM_GROUP_SPECIFIER]
            elapsed_time = time.time() - start_time
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, num_accumulated, (loss / batch.ntokens), (tokens / elapsed_time), lr)
            )
            start_time = time.time()
            del loss
            del loss_node
        return (total_loss / total_tokens), train_state