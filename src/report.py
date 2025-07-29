# src/report.py

import matplotlib.pyplot as plt

def plot_exposure(time_grid, EPE, ENE):
    plt.plot(time_grid, EPE, label="EPE (Expected Positive Exposure)")
    plt.plot(time_grid, ENE, label="ENE (Expected Negative Exposure)")
    plt.xlabel("Time")
    plt.ylabel("Exposure")
    plt.legend()
    plt.title("Expected Exposure Profiles")
    plt.grid(True)
    plt.show()
