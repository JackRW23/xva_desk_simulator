# src/portfolio.py
import numpy as np

class Portfolio:
    def __init__(self, trades, counterparty):
        self.trades = trades
        self.counterparty = counterparty

    def aggregate_exposure(self, spot_paths, time_grid):
        exposures = np.zeros_like(spot_paths)
        n_steps = spot_paths.shape[1]
        for trade in self.trades:
            for i in range(n_steps):
                exposures[:, i] += trade.mtm(spot_paths[:, i], time_grid[i])
        return exposures


