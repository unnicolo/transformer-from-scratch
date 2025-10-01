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
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0
