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
    pass