#!/usr/bin python
# -*- coding: UTF-8 -*-

"""
example_learning_schedule.py: Visualize the scheduled learning rate for three different exemplatory configurations.
"""

### IMPORTS ###
import altair as alt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from train.rate import rate

import sys

### CONSTANTS ###
FIGURE_WIDTH = 600
STEPS_PER_EPOCH = 20000
BETAS = (0.9, 0.98)
EPSILON = 1e-9
LEARNING_RATE_KEY = 'lr'

def example_learning_schedule():
    """Output a figure, showing how the learning rate is scheduled in each scenario."""
    examples = [
        # [d_model, factor, warmup_steps]
        [512, 1, 4000], # example 1
        [512, 1, 8000], # example 2
        [256, 1, 4000], # example 3
    ]
    learning_rates = []
    dummy_model = nn.Linear(1, 1)

    # We have 3 examples
    for index, example in enumerate(examples):
        optimizer = Adam(
            dummy_model.parameters(), betas=BETAS, eps=EPSILON
        ) 
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step, *example))

        history = []
        for step in range(STEPS_PER_EPOCH):
            history.append(optimizer.param_groups[0][LEARNING_RATE_KEY])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(history)
    
    learning_rates = torch.tensor(learning_rates)

    # Disable the row limit in Altair
    alt.data_transformers.disable_max_rows()

    examples_data = pd.concat(
        [
            pd.DataFrame(
                {
                    'step': range(STEPS_PER_EPOCH),
                    'Learning Rate': learning_rates[warmup_index, :],
                    'info': ["512:4000", "512:8000", "256:4000"][warmup_index],
                }
            )
            for warmup_index in range(len(examples))
        ]
    )

    return (
        alt.Chart(examples_data)
        .mark_line()
        .properties(width=FIGURE_WIDTH)
        .encode(x='step', y='Learning Rate', color='info')
        .interactive()
    )

if __name__ == '__main__':
    chart = example_learning_schedule()
    chart.save("learning_schedule_chart.html")