# src/scenarios.py

def shock_hazard_rate(counterparty, factor):
    old = counterparty.hazard_rate
    counterparty.hazard_rate *= factor
    return old

def reset_hazard_rate(counterparty, old_rate):
    counterparty.hazard_rate = old_rate
