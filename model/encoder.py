# -*- coding: UTF-8 -*-

"""
decoder.py: Implementation of a decoder module, as described in `The annotated transformer.`
"""

### Imports ###

import torch.nn as nn
from model.layers import LayerNorm, SublayerConnection
from model.utils import clones

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
        self.size = size

    def forward(self, x, mask):
        #print(f'x.shape is {x.shape}')
        x = self.sublayer_cnnts[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_cnnts[1](x, self.feed_forward)
