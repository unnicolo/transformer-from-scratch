# -*- coding:UTF-8 -*-

"""
label_smoothing.py: Implement label smoothing to mitigate the effects of overfitting and over-confidence.
"""

### IMPORTS ###
import torch
import torch.nn as nn

### CONSTANTS ###
KLDIVLOSS_REDUCTION_METHOD = 'sum'

class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        Initialize the label smoothing module.

        Args:
            size: The size of the vocabulary.
            padding_idx: The index that identifies a padding character.
            smoothing: How much smoothing to apply.
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction=KLDIVLOSS_REDUCTION_METHOD)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """Apply label smoothing to the input.

        Args:
            x: The input to smooth.
            target: The ground truth target values.

        Returns:
            A smoothed version of the input.
        """
        assert x.size(1) == self.size
        true_dist = torch.full_like(x, self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())