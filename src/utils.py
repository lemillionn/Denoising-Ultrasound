# src/utils.py

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, path):
    """
    Save a model state dict to the given file path.
    """
    # ensure the directory exists
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
