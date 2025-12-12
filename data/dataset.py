# -*- coding: utf-8 -*-

"""
dataset.py: Load the dataset used for model training.
"""

### IMPORTS ###
from torch.utils.data import Dataset
from datasets import load_dataset

### CONSTANTS ###
MULTI30K_URI = "bentrevett/multi30k"
TRAIN_KEY = "train"
VALID_KEY = "validation"

def load_multi30k() -> tuple[Dataset, Dataset]:
    """
    Load the Multi30k training and validation split.

    Args:
        No arguments.
    
    Returns:
        A tuple ``(train_ds, valid_ds)`` containing the training and validation splits of the Multi30k dataset, respectively.
    """
    ds = load_dataset(MULTI30K_URI)
    train_ds, valid_ds = ds[TRAIN_KEY], ds[VALID_KEY]
    
    return train_ds, valid_ds