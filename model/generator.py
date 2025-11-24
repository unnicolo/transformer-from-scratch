# -*- coding: UTF-8 -*-

"""
generator.py: Linear layer plus softmax step to generate the output probabilities.`
"""

### Imports ###
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Standard linear layer plus softmax generation step."""
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)
