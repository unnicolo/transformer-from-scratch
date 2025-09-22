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
  """Make `N` identical layers.

    Args:
      module: The module to clone.
      N: The number of clones to produce.

    Returns:
      The clones layers.
  """
  pass

class LayerNorm(nn.Module):
  """A LayerNorm module."""

  def __init__(self, num_features, eps=1e-6):
    """Construct a LayerNorm module.

      Args:
        num_features: The number of input features.
        eps: Epsilon added to the denominator for numerical stability.
    """
    pass

  def forward(self, x):
    """Apply layer normalization to the input.

      Args:
        x: The layer input to which apply normalization to.

      Returns:
        The normalized input values.
    """
    pass

class Encoder(nn.Module):
  def __init(self, layer, N):
    """Construct an Encoder unit of the transformer which is a stack of N layers.

    Args:
      N: The number of EncoderLayer clones in the transformer architecture.
    """
    pass 

  def forward(self, x, mask):
    """Pass the input (and mask) through each layer and norm the final result."""
    pass

class EncoderLayer(nn.Module):
  def __init(self, self_attn, feed_forward, size, dropout):
    """Construct a layer of the encoder.

    Args:
      self_attn: Self-attention mechanism.
      feed_forwards: A feed-forward network to learn connections.
      size: The number of output features of the encoder layer.
      dropout: Dropout probability.
    """
    pass

  def forward(self, x, mask):
    pass

class SublayerConnection(nn.Module):
  def __init__(self, size, dropout):
    pass

  def forward(self, x, sublayer):
    """Apply a residual connection and use pre layer normalization (for code simplicity)

    Args:
      x: The input to the layer.
      sublayer: The layer through which to the pass the input.

    Returns:
      The output value of the layer.
    """
    pass