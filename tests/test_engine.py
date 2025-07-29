# tests/test_engine.py

from src.engine import discount_curve

def test_discount_curve():
    import numpy as np
    times = np.array([0, 1, 2])
    r = 0.05
    expected = np.exp(-r * times)
    assert np.allclose(discount_curve(r, times), expected)
