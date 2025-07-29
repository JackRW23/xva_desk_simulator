# app.py

import numpy as np
from src.products import Derivative
from src.counterparty import Counterparty
from src.portfolio import Portfolio
from src.engine import simulate_spot_paths, expected_exposure_matrix, cva, dva
from src.utils import discount_curve
from src.report import plot_exposure

def main():
    # Parameters (can make user-configurable later)
    S0 = 1.0
    mu = 0.00
    sigma = 0.2
    T = 1.0
    n_steps = 20
    n_paths = 10000
    r = 0.03
    dt = T / n_steps

    # Setup trades and counterparties
    trade1 = Derivative(1_000_000, T, "swap", direction="receiver")
    counterparty = Counterparty("SocGen", hazard_rate=0.02, recovery_rate=0.4)

    # Simulate
    spot_paths, time_grid = simulate_spot_paths(S0, mu, sigma, T, n_steps, n_paths)
    pf = Portfolio([trade1], counterparty)
    exposures = pf.aggregate_exposure(spot_paths, time_grid)

    EPE, ENE = expected_exposure_matrix(exposures)
    discount_factors = discount_curve(r, time_grid)
    cva_val = cva(EPE, counterparty.hazard_rate, counterparty.recovery_rate, discount_factors, dt)
    dva_val = dva(ENE, 0.015, counterparty.recovery_rate, discount_factors, dt)

    # Report
    print(f"CVA: {cva_val:,.2f}")
    print(f"DVA: {dva_val:,.2f}")
    plot_exposure(time_grid, EPE, ENE)

if __name__ == "__main__":
    main()
