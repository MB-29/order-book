"""
Configuration for the perturbation experiment.

Regime: m0 > 0, m1 > 0 (small deterministic bias + noise).
We sweep over (m0/J, m1/J) pairs to cover:
  - Low participation: m0, m1 << J
  - Weak noise: m0 >> J, m1 << m0
  - Strong noise: m1 >> J, m0 << m1

Expected results (CLAUDE.md):
  - Low participation: E[p_t] ~ r0 sqrt(Dt/pi), Var(p_t) ~ r1^2 D t^{2H-1}
  - Weak noise: (E[p_t] - I_t)/I_t ~ -(m1/m0)^2 t^{2H-1}
  - Strong noise: E[p_t] ~ sqrt(m0/m1) I_t
"""

from pathlib import Path

# Physics
HURST = 0.75
D = 0.5
L = 10.0
NU = 0.
# ALPHA = 0.5
ALPHA = 0.
J = D * L

# Participation rate pairs (r0 = m0/J, r1 = m1/J)
REGIMES = {
    "low_participation": {"r0": 0.1, "r1": 0.01},
    # "weak_noise": {"r0": 5.0, "r1": 0.5},
    # "strong_noise": {"r0": 0.5, "r1": 5.0},
}

# Simulation grid
DURATION = 200.0
N_FRAMES = 50
N_GRID = 200
XMIN, XMAX = -100.0, 100.0

# Ensemble
N_SAMPLES = 500

SIM_PARAMS = {
    "model_type": "discrete",
    "D": D,
    "L": L,
    "nu": NU,
    "alpha": ALPHA,
    "duration": DURATION,
    "n_frames": N_FRAMES,
    "n_grid": N_GRID,
    "xmin": XMIN,
    "xmax": XMAX,
}

# Output
RESULTS_DIR = Path(__file__).parent / "results"
