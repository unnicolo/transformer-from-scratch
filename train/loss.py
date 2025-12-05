# -*- coding:UTF-8 -*-

"""
loss.py: Loss computation module
"""

### IMPORTS ###
import torch
import torch.nn as nn

class SimpleLossCompute(nn.Module):
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion):
        """
        Initialize the loss compute module.

        Args:
            generator: Probability output generator.
            criterion: Loss criterion.
        """
        super(SimpleLossCompute, self).__init__()
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        """
        
        """
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss