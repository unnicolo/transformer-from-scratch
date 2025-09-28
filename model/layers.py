#!/bin/env python
# -*- coding: UTF-8 -*-

"""
decoder.py: Implementation of a decoder module, as described in `The annotated transformer.`
"""

### Imports ###

import torch
import torch.nn as nn

class SublayerConnection(nn.Module):
    """Module to model the connection between the attention and feed-forward sublayers."""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply a residual connection and use pre layer normalization (for code simplicity)

        Args:
            x: The input to the layer.
            sublayer: The layer through which to the pass the input.

        Returns:
            The output value of the layer.
        """
        return x + self.dropout(x + sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    """A LayerNorm module."""

    def __init__(self, num_features, eps=1e-6):
        """Construct a LayerNorm module.

        Args:
            num_features: The number of output features.
            eps: Epsilon added to the denominator for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        """Apply layer normalization to the input.

        Args:
            x: The layer input to which apply normalization to.

        Returns:
            The normalized input values.
            """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return (self.a_2 * (x - mean)) / (std + self.eps) + self.b_2