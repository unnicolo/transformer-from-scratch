#!/bin/env python
# -*- coding: UTF-8 -*-

"""
rate.py: Implementation of the Noam learning rate scheduler, adjusting the learning rate according to the given formula.
"""

def rate(step: int, model_dim: int, factor: float, warmup_steps: int):
    """
    Implement the Noam learning rate scheduler.

    Args:
        step: Current step within the epoch
        model_dim: The model embedding dimension
        factor: Scaling factor
        warmup_steps: Number of steps to use for warmup
    
    Returns:
        float: The learning rate for the current step
    """
    # Default the step to 1 to avoid raising zero to a negative power
    if step == 0:
        step = 1
    
    return factor * (
        model_dim ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )