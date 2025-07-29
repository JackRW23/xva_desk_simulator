# src/counterparty.py
import numpy as np

class Counterparty:
    def __init__(self, name, hazard_rate=0.02, recovery_rate=0.4):
        self.name = name
        self.hazard_rate = hazard_rate      # Annualized default intensity (lambda)
        self.recovery_rate = recovery_rate  # Recovery on default

    def pd_curve(self, times):
        # Survival probability to each time
        return [1 - self.survival_probability(t) for t in times]

    def survival_probability(self, t):
        # Exponential default model
        return np.exp(-self.hazard_rate * t)
