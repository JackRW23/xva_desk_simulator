# src/products.py
import numpy as np

class Derivative:
    def __init__(self, notional, maturity, product_type, direction="payer", strike=1.0):
        self.notional = notional
        self.maturity = maturity
        self.product_type = product_type
        self.direction = direction
        self.strike = strike

    def mtm(self, spot, t):
        """
        spot: np.array (n_paths,)
        t: scalar (time)
        Returns: np.array (n_paths,)
        """
        if self.product_type == 'swap':
            direction_sign = 1 if self.direction == "receiver" else -1
            return self.notional * (0.05 * (self.maturity - t)) * direction_sign * np.ones_like(spot)
        elif self.product_type == 'fx':
            direction_sign = 1 if self.direction == "long" else -1
            return self.notional * (spot - 1.0) * direction_sign
        elif self.product_type == 'option':
            return np.maximum(spot - self.strike, 0) * self.notional
        else:
            return np.zeros_like(spot)
