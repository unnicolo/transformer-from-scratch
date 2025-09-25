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
    def __init__(self, layer, N):
        """Construct an Encoder unit of the transformer which is a stack of N layers.

        Args:
            layer: A single layer of the encoder.
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
    def __init__(self, self_attn, feed_forward, size, dropout):
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

class Decoder(nn.Module):
    """Decoder in the decoder stack consisting of `N` identical decoder modules."""
    def __init__(self, layer, N):
        """Construct a decoder module.

            Args:
                layer: A layer in the decoder.
                N: The number of identical layers to use in the decoder.
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        """Pass the input through the decoder module.
        
        Args:
            x:
            memory:
            src_mask:
            target_mask:
        
        Returns:
            The tensor obtained by passing the input through the decoder module.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """A layer in the decoder stack, consisting of two attention blocks and a feed-forward neural network."""
    def __init__(self, self_attn, feed_forward, size, dropout):
        """Construct a decoder layer.
        
        Args:
            self_attn: (Masked) self-attention mechanism.
            feed_forward: A feed-forward neural network that enable the transformer to learn.
            size: The number of output features of the transformer.
            dropout: Dropout probability at each layer.
        """
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_cnnts = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        """Pass the input (and the mask) through each layer and norm the result.
        
        Args:
            x: The input to pass through.
            memory: Output tensors of the encoder stack.
            src_mask: 
            tgt_mask:
        
        Returns: The output values after passing the input through each of the sublayers.
        """
        m = memory
        x = self.sublayer_cnnts[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer_cnnts[1](x, lambda x: self.self_attn(m, m, x, src_mask))
        return self.sublayer_cnnts[2](x, self.feed_forward)
