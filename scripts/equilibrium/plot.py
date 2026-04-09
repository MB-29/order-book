"""
Plot results from the unbiased equilibrium experiment.

Produces two figures:
  1. Price variance over time (log-log), compared to t^{2H-1}
  2. Average orderbook density profiles at selected times
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

from scripts.equilibrium.config import (
    HURST, L, J, XMAX, DURATION, N_FRAMES,
    M1_VALUES, PROFILE_INDICES, RESULTS_DIR,
)

# --- Load ---
with open(RESULTS_DIR / "equilibrium_results.pkl", "rb") as f:
    all_results = pickle.load(f)

dt = DURATION / N_FRAMES

# --- Figure 1: Price variance ---
fig1, ax = plt.subplots(figsize=(8, 5))

for m1 in M1_VALUES:
    res = all_results[m1]
    time = res["time"][1:]
    ax.plot(time, res["price_variance"][1:], label=f"$m_1/J = {m1/J:.1f}$")

slope = 2 * HURST - 1
t_ref = np.array(all_results[M1_VALUES[0]]["time"][1:])
ax.plot(t_ref, t_ref**slope * 0.3, "k--", lw=1,
        label=f"$\\propto t^{{{slope:.1f}}}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$t$")
ax.set_ylabel("$\\mathrm{Var}(p_t)$")
ax.set_title("Price variance — unbiased regime ($m_0=0$)")
ax.legend()
fig1.tight_layout()

# --- Figure 2: Average orderbook profiles at different times ---
n_rates = len(M1_VALUES)
n_snaps = len(PROFILE_INDICES)
fig2, axes = plt.subplots(1, n_rates, figsize=(6 * n_rates, 5), squeeze=False)

alphas_transparency = np.linspace(0.3, 1.0, n_snaps)
color_book = "#465987"

for i, m1 in enumerate(M1_VALUES):
    ax = axes[0, i]
    res = all_results[m1]
    X = res["X"]
    dx = X[1] - X[0]
    r1 = m1 / J

    for j, frame_idx in enumerate(PROFILE_INDICES):
        ask_density = res["ask_volumes_mean"][j] / dx
        bid_density = res["bid_volumes_mean"][j] / dx
        a = alphas_transparency[j]

        ax.plot(X, ask_density, color=color_book, lw=2, alpha=a,
                label=f"$t = {frame_idx * dt:.0f}$")
        ax.plot(X, bid_density, color=color_book, lw=2, alpha=a)

    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.set_title(f"$m_1/J = {r1:.1f}$")
    ax.legend(fontsize=9)

fig2.suptitle(
    "Average orderbook density — unbiased regime ($m_0 = 0$)", y=1.01)
fig2.tight_layout()

plt.show()
