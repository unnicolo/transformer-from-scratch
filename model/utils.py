#!/bin/env python
# -*- coding: UTF-8 -*-

"""
utils.py: Utility functions. 
"""

import torch
import torch.nn as nn
import numpy as np
import copy

def clones(module, N):
    """Make `N` identical layers (deepcopy).

         Args:
            module: The module to clone.
            N: The number of clones to produce.

        Returns:
            The cloned layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions.

    Args:
        size: The size of each dimension of the attention.

    Returns:
        The mask preventing leftward information flow, with zeros above the main diagonal and ones everywhere else.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)

    return subsequent_mask == 0
