#!/bin/env python
# -*- coding: UTF-8 -*-

"""
transformer.py: Implementation of a vanilla transformer, as described in `The annotated transformer.`
"""

# Imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        return (self.a_2 * (x - mean)) / (std + eps) + self.b_2

    class Encoder(nn.Module):
    def __init(self, layer, N):
        """Construct an Encoder unit of the transformer which is a stack of N layers.

        Args:
        N: The number of EncoderLayer clones in the transformer architecture.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer and norm the final result."""
        for layer in self.layers:
        x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init(self, self_attn, feed_forward, size, dropout):
        """Construct a layer of the encoder.

        Args:
        self_attn: Self-attention mechanism.
        feed_forwards: A feed-forward network to learn connections.
        size: The number of output features of the encoder layer.
        dropout: Dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_cnnts = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_cnnts[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_cnnts[1](x, lambda x: self.feed_forward)

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
        return x + self.dropout(x + sublayer(norm(x)))