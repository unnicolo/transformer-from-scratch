#!/bin/env python
# -*- coding: UTF-8 -*-

"""
synthetic_data.py: Generation of random synthetic training data for training purposes.
"""

### IMPORTS ###
import torch

from data.batch import Batch

### CONSTANTS ###
PAD = 0
SEQ_LEN = 10

def generate_data(vocabulary_size, batch_size, num_batches):
    """Generate synthetic data for a src-tgt copy task.

    Args:
        vocabulary_size: The size of the vocabulary to draw from.
        batch_size: The size of a single batch.
        num_batches: The number of batches to generate.

    Returns:
        A generator, yielding :class:`Batch` objects that hold on to a single batch each.
    """
    for _ in range(num_batches):
        data = torch.randint(1, vocabulary_size, size=(batch_size, SEQ_LEN))
        # Each data sequence starts with the same symbol - much like a <start-of-sequence>-token.
        data[:, 0] = 1
        src = data.clone().detach()
        tgt = data.clone().detach()
        yield Batch(src, tgt, PAD)