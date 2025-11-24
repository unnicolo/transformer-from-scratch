#!/usr/bin python
# -*- coding: UTF-8 -*-

"""
batch.py: Defines the Batch class for storing source and target sentences
and constructing the corresponding attention masks.
"""

### Imports ###
import torch
from model.utils import subsequent_mask

class Batch:
    """Implements the Batch class, objects of which hold on to a batch of training data, along with the corresponding attention masks."""
    def __init__(self, src, tgt=None, pad=2): # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]      # All tokens but the last one since we try to predict it
            self.tgt_y = tgt[:, 1:]     # All tokens but the first one
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        
        return tgt_mask