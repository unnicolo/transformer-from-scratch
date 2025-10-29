#!/bin/env python
# -*- coding: UTF-8 -*-

"""
feed_forward.py: Implementation of the position-wise feed-forward neural network used in the encoder/decoder architecture, 
as described in `The annotated transformer.`
"""

### Imports ###

import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """Construct a PositionWise feed-forward module which implements the FF equation.

        Args:
            d_model: The model input dimension size.
            d_ff: The inner-layer dimensionality.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout) 

    def forward(self, x):
        return self.linear_2(self.linear_1(x).relu())
