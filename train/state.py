#!/bin/env python
# -*- coding: UTF-8 -*-

"""
state.py: Keep track of different training metrics.
"""

class TrainState:
    """Track number of steps, examples and tokens processed."""

    step: int = 0          # Steps in the current epoch.
    accum_step: int = 0     # Number of gradient calculation steps.
    tokens: int = 0         # Total number of tokens processed.
    samples: int = 0        # Total number of examples used.