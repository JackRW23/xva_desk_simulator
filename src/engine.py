# src/engine.py

import numpy as np

def simulate_spot_paths(S0, mu, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return S, times

def expected_exposure_matrix(exposures):
    positive_exposure = np.maximum(exposures, 0)
    negative_exposure = np.minimum(exposures, 0)
    return positive_exposure.mean(axis=0), -negative_exposure.mean(axis=0)

def cva(EPE, hazard_rate, recovery_rate, discount_factors, dt):
    # Discrete time CVA with constant hazard rate and flat discounting
    cva_val = 0.0
    for i in range(1, len(EPE)):
        dp = np.exp(-hazard_rate * (i-1)*dt) - np.exp(-hazard_rate * i*dt)
        cva_val += (1 - recovery_rate) * EPE[i] * dp * discount_factors[i]
    return cva_val

def dva(ENE, own_hazard_rate, recovery_rate, discount_factors, dt):
    dva_val = 0.0
    for i in range(1, len(ENE)):
        dp = np.exp(-own_hazard_rate * (i-1)*dt) - np.exp(-own_hazard_rate * i*dt)
        dva_val += (1 - recovery_rate) * ENE[i] * dp * discount_factors[i]
    return dva_val
