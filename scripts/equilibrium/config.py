"""
Configuration for the unbiased equilibrium experiment.

Regime: m0=0, m1>0, varying m1/J.
Expected results (CLAUDE.md):
  - E[p_t] = 0
  - Var(p_t) ~ t^{2H-1}
"""

from pathlib import Path

# Physics
HURST = 0.75
D = 0.5
L = 10.0
NU = 0.1
ALPHA = 0.5
J = D * L

# Participation rates m1/J
M1_OVER_J = [.5, 1.]
# M1_OVER_J = [0.1, 0.5, 1.0]
M1_VALUES = [r * J for r in M1_OVER_J]

# Simulation grid
DURATION = 100.0
N_FRAMES = 50
N_GRID = 200
XMIN, XMAX = -100.0, 100.0

# Ensemble
N_SAMPLES = 500

# Snapshot frames for orderbook profiles
PROFILE_INDICES = [0, N_FRAMES // 4, N_FRAMES //
                   2, 3 * N_FRAMES // 4, N_FRAMES - 1]

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
