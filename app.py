# app.py

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from src.products import Derivative
from src.counterparty import Counterparty
from src.portfolio import Portfolio
from src.engine import simulate_spot_paths, expected_exposure_matrix, cva, dva
from src.utils import discount_curve

class XVAGUI:
    def __init__(self, root):
        self.root = root
        root.title("xVA Desk Simulator")

        params_fr = tk.LabelFrame(root, text="Simulation Parameters", padx=10, pady=8)
        params_fr.pack(fill=tk.X, expand=False)

        self.S0_var      = tk.DoubleVar(value=1.0)
        self.mu_var      = tk.DoubleVar(value=0.00)
        self.sigma_var   = tk.DoubleVar(value=0.2)
        self.T_var       = tk.DoubleVar(value=1.0)
        self.n_steps_var = tk.IntVar(value=20)
        self.n_paths_var = tk.IntVar(value=10000)
        self.r_var       = tk.DoubleVar(value=0.03)

        self.counterparty_hazard_var  = tk.DoubleVar(value=0.02)
        self.counterparty_recovery_var= tk.DoubleVar(value=0.4)
        self.own_hazard_var           = tk.DoubleVar(value=0.015)

        self.has_fx_var     = tk.IntVar(value=1)
        self.has_swap_var   = tk.IntVar(value=1)
        self.has_option_var = tk.IntVar(value=1)
        self.swap_notional_var   = tk.DoubleVar(value=1_000_000)
        self.fx_notional_var     = tk.DoubleVar(value=1_000_000)
        self.option_notional_var = tk.DoubleVar(value=1_000_000)
        self.option_strike_var   = tk.DoubleVar(value=1.0)

        row = 0
        tk.Label(params_fr, text="S₀ (spot):").grid(row=row, column=0, sticky="w")
        tk.Entry(params_fr, textvariable=self.S0_var, width=8).grid(row=row, column=1)
        tk.Label(params_fr, text="μ:").grid(row=row, column=2, sticky="w")
        tk.Entry(params_fr, textvariable=self.mu_var, width=8).grid(row=row, column=3)
        tk.Label(params_fr, text="σ:").grid(row=row, column=4, sticky="w")
        tk.Entry(params_fr, textvariable=self.sigma_var, width=8).grid(row=row, column=5)
        tk.Label(params_fr, text="T:").grid(row=row, column=6, sticky="w")
        tk.Entry(params_fr, textvariable=self.T_var, width=8).grid(row=row, column=7)
        tk.Label(params_fr, text="n_steps:").grid(row=row, column=8, sticky="w")
        tk.Entry(params_fr, textvariable=self.n_steps_var, width=8).grid(row=row, column=9)
        tk.Label(params_fr, text="n_paths:").grid(row=row, column=10, sticky="w")
        tk.Entry(params_fr, textvariable=self.n_paths_var, width=8).grid(row=row, column=11)
        tk.Label(params_fr, text="r (discount):").grid(row=row, column=12, sticky="w")
        tk.Entry(params_fr, textvariable=self.r_var, width=8).grid(row=row, column=13)

        row += 1
        tk.Label(params_fr, text="Counterparty λ (hazard):").grid(row=row, column=0, sticky="w")
        tk.Entry(params_fr, textvariable=self.counterparty_hazard_var, width=8).grid(row=row, column=1)
        tk.Label(params_fr, text="Counterparty recovery:").grid(row=row, column=2, sticky="w")
        tk.Entry(params_fr, textvariable=self.counterparty_recovery_var, width=8).grid(row=row, column=3)
        tk.Label(params_fr, text="Bank λ (own hazard):").grid(row=row, column=4, sticky="w")
        tk.Entry(params_fr, textvariable=self.own_hazard_var, width=8).grid(row=row, column=5)

        row += 1
        tk.Checkbutton(params_fr, text="Include Swap", variable=self.has_swap_var).grid(row=row, column=0)
        tk.Label(params_fr, text="Notional:").grid(row=row, column=1)
        tk.Entry(params_fr, textvariable=self.swap_notional_var, width=10).grid(row=row, column=2)
        tk.Checkbutton(params_fr, text="Include FX", variable=self.has_fx_var).grid(row=row, column=3)
        tk.Label(params_fr, text="Notional:").grid(row=row, column=4)
        tk.Entry(params_fr, textvariable=self.fx_notional_var, width=10).grid(row=row, column=5)
        tk.Checkbutton(params_fr, text="Include Option", variable=self.has_option_var).grid(row=row, column=6)
        tk.Label(params_fr, text="Notional:").grid(row=row, column=7)
        tk.Entry(params_fr, textvariable=self.option_notional_var, width=10).grid(row=row, column=8)
        tk.Label(params_fr, text="Strike:").grid(row=row, column=9)
        tk.Entry(params_fr, textvariable=self.option_strike_var, width=10).grid(row=row, column=10)

        tk.Button(root, text="Run Simulation", command=self.run_simulation, bg="lightblue").pack(pady=10)

        self.output_text = tk.Text(root, height=8, width=90, font=("Consolas", 10))
        self.output_text.pack(padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)

        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(padx=10, pady=5)
        self.current_canvas = None

    def run_simulation(self):
        S0 = self.S0_var.get()
        mu = self.mu_var.get()
        sigma = self.sigma_var.get()
        T = self.T_var.get()
        n_steps = self.n_steps_var.get()
        n_paths = self.n_paths_var.get()
        r = self.r_var.get()
        dt = T / n_steps
        cp_hazard = self.counterparty_hazard_var.get()
        cp_recovery = self.counterparty_recovery_var.get()
        own_hazard = self.own_hazard_var.get()

        trades = []
        if self.has_swap_var.get():
            trades.append(Derivative(self.swap_notional_var.get(), T, "swap", direction="receiver"))
        if self.has_fx_var.get():
            trades.append(Derivative(self.fx_notional_var.get(), T, "fx", direction="long"))
        if self.has_option_var.get():
            trades.append(Derivative(self.option_notional_var.get(), T, "option", direction="receiver", strike=self.option_strike_var.get()))

        if not trades:
            messagebox.showerror("Input Error", "Select at least one trade type.")
            return

        counterparty = Counterparty("Counterparty", hazard_rate=cp_hazard, recovery_rate=cp_recovery)
        spot_paths, time_grid = simulate_spot_paths(S0, mu, sigma, T, n_steps, n_paths)
        pf = Portfolio(trades, counterparty)
        exposures = pf.aggregate_exposure(spot_paths, time_grid)
        EPE, ENE = expected_exposure_matrix(exposures)
        discount_factors = discount_curve(r, time_grid)
        cva_val = cva(EPE, cp_hazard, cp_recovery, discount_factors, dt)
        dva_val = dva(ENE, own_hazard, cp_recovery, discount_factors, dt)

        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"CVA: ${cva_val:,.2f}\n")
        self.output_text.insert(tk.END, f"DVA: ${dva_val:,.2f}\n")
        self.output_text.insert(tk.END, f"Peak EPE: {EPE.max():,.2f}\n")
        self.output_text.insert(tk.END, f"Peak ENE: {ENE.max():,.2f}\n")
        self.output_text.config(state=tk.DISABLED)

        # ---- Embedded matplotlib plot with tight_layout ----
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(time_grid, EPE, label="EPE (Expected Positive Exposure)")
        ax.plot(time_grid, ENE, label="ENE (Expected Negative Exposure)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Exposure")
        ax.legend()
        ax.set_title("Expected Exposure Profiles")
        ax.grid(True)
        fig.tight_layout()  # This is the critical addition

        os.makedirs("outputs", exist_ok=True)
        fig.savefig(os.path.join("outputs", "exposure_profile.png"), bbox_inches="tight")

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plt.close(fig)

if __name__ == "__main__":
    tk.Tk.report_callback_exception = lambda *exc: messagebox.showerror("Error", f"Uncaught exception:\n{exc[1]}")
    root = tk.Tk()
    XVAGUI(root)
    root.mainloop()
