# src/utils.py

import numpy as np

def discount_curve(r, time_grid):
    return np.exp(-r * time_grid)
