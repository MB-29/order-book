"""
Configuration for the perturbation experiment.

Regime: m0 > 0, m1 > 0 (small deterministic bias + noise).
Parameters come from named presets in :mod:`scripts.presets`.
"""

from pathlib import Path

from scripts.presets import get_preset

# Local short name (key) -> canonical preset name in scripts.presets.
# plot.py / run.py carry the short name; add e.g.
#   "weak_noise": "impact_weak_noise" to extend the sweep.
PRESET_NAMES = {
    "low_participation": "perturbation_low_participation",
}
PRESETS = {short: get_preset(full) for short, full in PRESET_NAMES.items()}

# Shared physics / grid (all presets in this sweep must agree)
_base = next(iter(PRESETS.values()))
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

# Regimes (kept under their preset names, mapping to {r0, r1})
REGIMES = {name: {"r0": p.r0, "r1": p.r1} for name, p in PRESETS.items()}

# Ensemble (override on the run-script CLI with --n-samples)
N_SAMPLES = 500

SIM_PARAMS = _base.sim_params()

# Output
RESULTS_DIR = Path(__file__).parent / "results"
