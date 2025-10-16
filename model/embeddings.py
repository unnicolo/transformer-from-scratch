#!/bin/env python
# -*- coding: UTF-8 -*-

"""
embeddings.py: Token and positional embeddings.
"""

### Imports ###

import torch
import torch.nn as nn
import math

DEFAULT_DROPOUT_PROBABILITY = 0.1
MAX_SEQ_LEN = 5000
FREQUENCY_BASE_VALUE = 10000.0

class Embeddings(nn.Module):
    """Token embeddings."""
    def __init__(self, d_model, vocab_size):
        """Construct a token embedding module.
        
        Args:
            d_model: The model embedding dimension.
            vocab_size: Size of the vocabulary.
        """
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.lookup_table = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        """Embed a token.
        
        Args:
            x: The input tensor to be embedded.

        Returns:
            The embedding of the token, a vector in a d_model-dimensional space.
        """
        return self.lookup_table(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Implements positional encoding as described in the paper."""
    def __init__(self, d_model, dropout=DEFAULT_DROPOUT_PROBABILITY, max_len=MAX_SEQ_LEN):
        """Construct a positional encoding module.
        
        Args:
            d_model: The dimension of the model embedding.
            dropout: Dropout probability.
            max_len: Maximum length of the input sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # We compute the positional encodings once and register them as a buffer for use in the `forward` method.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(FREQUENCY_BASE_VALUE) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Apply positional encoding to the input.
        Args:
            x: The input to the positional encoding layer.
        
        Returns:
            The sum of the input and the positional encoding for that input.
        """
        pe = self.get_buffer('pe')
        return x + pe
