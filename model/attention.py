#!/bin/env python
# -*- coding: UTF-8 -*-

"""
attention.py: Implementation of scaled dot-product attention and multi-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MASK_FILL_VALUE = -1e9

def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot-product attention."""
    keyspace_dim = key.size(-1)
    scale_factor = 1 / (math.sqrt(keyspace_dim))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if mask is not None:
        scores = scores.masked_fill(mask == 0, MASK_FILL_VALUE)
    attn_logits = F.softmax(scores, dim = -1)
    if dropout is not None:
        attn_logits = dropout(attn_logits)
    return torch.matmul(attn_logits, value), attn_logits

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Construct a multi-head attention module.

        Args:
            h: The number of attention heads to use in parallel.
            d_model: The model input dimension.
            dropout: The dropout rate to use.
        """
        super(MultiHeadAttention, self).__init__()
        # Ensure that d_model is a multiple of h, such that we can
        # evenly distribute the input among h heads.
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Push query, key and value sequences through the multi-head attention block.
        Note that it is assume that d_k = d_v in the paper.

        Args:
            query: The query sequence tensor of shape (batch_size, seq_length, d_model).
            key: The key sequence tensor of shape (batch_size, seq_length, d_model).
            value: The value sequence tensor of shape (batch_size, seq_length, d_model).
            mask: The mask to apply (if any) to prevent future positions to attend to earlier positions.

        Returns:
            The output tensor that results after applying multi-head attetion to query, key and value.
        """
        if mask is not None:
            # Insert extra dimension to account for batch_size dimension
            mask = torch.unsqueeze(mask, 1)
        num_batches = query.size(0)
      
        # Step 1: Do batched linear projections, d_model => h x d_k
        query, key, value = \
            [linear(x).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linears, (query, key, value))]
        
        # Step 2: Apply attention to the batches
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Step 3: Concat the tensors using a view and apply the final linear layer.
        # h x d_k => d_model
        x = x.transpose(1, 2).contiguous() \
             .view(num_batches, -1, self.h * self.d_model)
        
        return self.linears[-1](x)