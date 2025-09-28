#!/bin/env python
# -*- coding: UTF-8 -*-

"""
decoder.py: Implementation of a decoder module, as described in `The annotated transformer.`
"""

### Imports ###

import torch.nn as nn
from model.layers import LayerNorm, SublayerConnection
from utils import clones

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

    def forward(self, x, memory, src_mask, tgt_mask):
        """Pass the input through the decoder module.
        
        Args:
            x:
            memory:
            src_mask:
            tgt_mask:
        
        Returns:
            The tensor obtained by passing the input through the decoder module.
        """
        for layer in self.layers:
            x = layer(x, tgt_mask)
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