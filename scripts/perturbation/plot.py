"""
Plot results from the perturbation experiment.

Produces three figures:
  1. Low participation: E[p_t] vs r0 * sqrt(Dt/pi)
  2. Weak noise: relative correction (E[p_t] - I_t) / I_t
  3. Strong noise: E[p_t] vs sqrt(m0/m1) * I_t
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

from config import (
    HURST, D, J, REGIMES, RESULTS_DIR,
)

# --- Load ---
with open(RESULTS_DIR / "perturbation_results.pkl", "rb") as f:
    all_results = pickle.load(f)


def noiseless_impact(r0, t):
    """Square-root impact I_t = r0 * sqrt(D t / pi)."""
    return r0 * np.sqrt(D * t / np.pi)


# --- Figure 1: Low participation ---
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

res = all_results["low_participation"]
rates = REGIMES["low_participation"]
r0, r1 = rates["r0"], rates["r1"]
time = np.array(res["time"][1:])

I_t = noiseless_impact(r0, time)
ax1a.plot(time, res["price_mean"][1:], label=r"$\mathbb{E}[p_t]$")
ax1a.plot(time, I_t, "k--", lw=1, label=r"$r_0\sqrt{Dt/\pi}$")
ax1a.set_xlabel("$t$")
ax1a.set_ylabel("Price impact")
ax1a.set_title(f"Mean impact — low participation ($r_0={r0}$, $r_1={r1}$)")
ax1a.legend()

slope = 2 * HURST - 1
ax1b.plot(time, res["price_variance"][1:], label=r"$\mathrm{Var}(p_t)$")
ax1b.plot(time, r1**2 * D * time**slope, "k--", lw=1,
          label=f"$r_1^2 D \\, t^{{{slope:.1f}}}$")
ax1b.set_xscale("log")
ax1b.set_yscale("log")
ax1b.set_xlabel("$t$")
ax1b.set_ylabel("Variance")
ax1b.set_title(f"Price variance — low participation")
ax1b.legend()
fig1.tight_layout()

# # --- Figure 2: Weak noise ---
# fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

# res = all_results["weak_noise"]
# rates = REGIMES["weak_noise"]
# r0, r1 = rates["r0"], rates["r1"]
# m0, m1 = r0 * J, r1 * J
# time = np.array(res["time"][1:])

# I_t = noiseless_impact(r0, time)
# ax2a.plot(time, res["price_mean"][1:], label=r"$\mathbb{E}[p_t]$")
# ax2a.plot(time, I_t, "k--", lw=1, label=r"$I_t$")
# ax2a.set_xlabel("$t$")
# ax2a.set_ylabel("Price impact")
# ax2a.set_title(f"Mean impact — weak noise ($r_0={r0}$, $r_1={r1}$)")
# ax2a.legend()

# relative_correction = (res["price_mean"][1:] - I_t) / I_t
# slope = 2 * HURST - 1
# ax2b.plot(time, -relative_correction,
#           label=r"$-({\mathbb{E}[p_t] - I_t})/{I_t}$")
# ax2b.plot(time, (m1 / m0)**2 * time**slope, "k--", lw=1,
#           label=f"$(m_1/m_0)^2 \\, t^{{{slope:.1f}}}$")
# ax2b.set_xscale("log")
# ax2b.set_yscale("log")
# ax2b.set_xlabel("$t$")
# ax2b.set_ylabel("Relative correction")
# ax2b.set_title("Negative correction to impact — weak noise")
# ax2b.legend()
# fig2.tight_layout()

# # --- Figure 3: Strong noise ---
# fig3, ax3 = plt.subplots(figsize=(7, 5))

# res = all_results["strong_noise"]
# rates = REGIMES["strong_noise"]
# r0, r1 = rates["r0"], rates["r1"]
# m0, m1 = r0 * J, r1 * J
# time = np.array(res["time"][1:])

# I_t = noiseless_impact(r0, time)
# ax3.plot(time, res["price_mean"][1:], label=r"$\mathbb{E}[p_t]$")
# ax3.plot(time, np.sqrt(m0 / m1) * I_t, "k--", lw=1,
#          label=r"$\sqrt{m_0/m_1} \, I_t$")
# ax3.set_xlabel("$t$")
# ax3.set_ylabel("Price impact")
# ax3.set_title(f"Mean impact — strong noise ($r_0={r0}$, $r_1={r1}$)")
# ax3.legend()
# fig3.tight_layout()

plt.show()
