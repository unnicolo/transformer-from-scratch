#!/bin/env python
# -*- coding: UTF-8 -*-

"""
transformer.py: Implementation of a vanilla transformer, as described in `The annotated transformer.`
"""

### Imports ###

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from decoder import DecoderLayer, Decoder
from embeddings import Embeddings, PositionalEncoding
from encoder import EncoderLayer, Encoder
from feed_forward import PositionWiseFeedForward

### Constants ###
DROPOUT = 0.1
MODEL_DIMENSION = 512
INNER_LAYER_DIM = 2048
NUM_ATTENTION_HEADS = 6
NUM_ENCODER_DECODER_BLOCKS = 8
SOURCE_VOCAB_SIZE = 32000
TARGET_VOCAB_SIZE = 32000

class Transformer(nn.Module):
    def __init__(
            self, 
            d_model=MODEL_DIMENSION, 
            d_ff=INNER_LAYER_DIM, 
            num_heads=NUM_ATTENTION_HEADS, 
            dropout=DROPOUT, 
            src_vocab=SOURCE_VOCAB_SIZE, 
            tgt_vocab=TARGET_VOCAB_SIZE
    ):
        """Construct a transformer.
        
        Args:
            d_model: The model embedding dimension.
            d_ff: The dimension of the position-wise feed-forward neural network.
            num_heads: The number of parallel attention heads.
            dropout: Dropout probability used.
            src_vocab: Size of the source vocabulary used.
            tgt_vocab: Size of the target vocabulary used.
        """
        c = copy.deepcopy
        ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        attn = MultiHeadAttention(num_heads, d_model, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.src_embedding = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.tgt_embedding = nn.Sequential(Embeddings(d_model, tgt_vocab) c(position))
        self.encoder = Encoder(EncoderLayer(c(attn), c(ff), d_model, dropout), NUM_ENCODER_DECODER_BLOCKS)
        self.decoder = Decoder(DecoderLayer(c(attn), c(ff), d_model, dropout), NUM_ENCODER_DECODER_BLOCKS)

    def forward(self, x):
        pass
