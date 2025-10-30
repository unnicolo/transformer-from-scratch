#!/bin/env python
# -*- coding: UTF-8 -*-

"""
batch.py: Defines the Batch class for storing source and target sentences
and constructing the corresponding attention masks.
"""

### Imports ###
import torch

class Batch:
    """Implements the Batch class, object of which hold on to a batch of training data, along with the corresponding attention masks."""
    def __init__(self, src, tgt=None, pad=2): # 2 = <blank>
        pass
    
    @staticmethod
    def make_std_mask(tgt, pad):
        pass