#!/bin/env python
# -*- coding: UTF-8 -*-

"""
utils.py: Utility functions. 
"""

import torch.nn as nn
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