"""
Configuration for the unbiased equilibrium experiment.

Regime: m0=0, m1>0, varying m1/J.
Expected results (CLAUDE.md):
  - E[p_t] = 0
  - Var(p_t) ~ t^{2H-1}

Parameters come from named presets in :mod:`scripts.presets`; this file
keeps the experiment-specific bits (orderbook-profile snapshots, results
directory) and re-exports the public constants downstream scripts use.
"""

from pathlib import Path

from scripts.presets import get_preset

# Which presets to sweep over (one MC run per preset).
PRESET_NAMES = ["equilibrium_m1_half_J", "equilibrium_m1_J"]
PRESETS = [get_preset(name) for name in PRESET_NAMES]

# Shared physics / grid — all presets in this sweep agree on these.
_base = PRESETS[0]
HURST = _base.hurst
D = _base.D
L = _base.L
NU = _base.nu
ALPHA = _base.alpha
J = _base.J
DURATION = _base.duration
N_FRAMES = _base.n_frames
N_GRID = _base.n_grid
XMIN, XMAX = _base.xmin, _base.xmax
DX = _base.dx
DT_STEP = _base.dt_step

# Sweep axis
M1_VALUES = [p.m1 for p in PRESETS]
M1_OVER_J = [p.r1 for p in PRESETS]

# Ensemble (override on the run-script CLI with --n-samples)
N_SAMPLES = 500

# Snapshot frames for orderbook profiles (experiment-specific)
PROFILE_INDICES = [0, N_FRAMES // 4, N_FRAMES // 2, 3 * N_FRAMES // 4, N_FRAMES - 1]

SIM_PARAMS = _base.sim_params(
    measured_quantities=["ask_volumes", "bid_volumes"],
    measurement_indices=PROFILE_INDICES,
)

# Output
RESULTS_DIR = Path(__file__).parent / "results"
