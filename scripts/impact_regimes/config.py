"""
Configuration for the impact-regime sweep.

We scan the (r0, r1) plane where r0 = m0/J and r1 = m1/J. Regimes are named
after the qualitative theoretical predictions in CLAUDE.md:

  - low_participation     : r0, r1 << 1            E[p_t] ~ r0 sqrt(Dt/pi)
  - weak_noise            : r0 >> 1, r1 << r0      E[p_t] ~ I_t with -(m1/m0)^2 correction
  - strong_noise          : r1 >> 1, r0 << r1      E[p_t] ~ sqrt(m0/m1) I_t
  - high_both             : r0, r1 >> 1            (m0, m1 >> J: high-participation, general case)
  - pure_noise            : r0 = 0, r1 > 0         E[p_t] = 0, Var ~ t^{2H-1}

Parameters come from named presets in :mod:`scripts.presets`.
"""

from pathlib import Path

from scripts.presets import get_preset

# Presets to sweep. The local short name (key) is what plot.py / run.py
# carry through; it maps to the canonical preset in scripts.presets.
PRESET_NAMES = {
    "low_participation":   "impact_low_participation",
    "low_p_strong_noise":  "impact_low_p_strong_noise",
    "balanced":            "impact_balanced",
    "weak_noise":          "impact_weak_noise",
    "strong_noise":        "impact_strong_noise",
    "high_both":           "impact_high_both",
    "pure_noise":          "impact_pure_noise",
}
PRESETS = {short: get_preset(full) for short, full in PRESET_NAMES.items()}

# Shared physics / grid (all impact presets agree on these)
_base = next(iter(PRESETS.values()))
HURST = _base.hurst
D = _base.D
L = _base.L
NU = _base.nu
J = _base.J
DURATION = _base.duration
N_FRAMES = _base.n_frames
N_GRID = _base.n_grid
XMIN, XMAX = _base.xmin, _base.xmax
DX = _base.dx
DT_STEP = _base.dt_step

REGIMES = {name: {"r0": p.r0, "r1": p.r1} for name, p in PRESETS.items()}

# Ensemble (override on the run-script CLI with --n-samples)
N_SAMPLES = 1_000

SIM_PARAMS = _base.sim_params()

# Output
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "impact_regimes.pkl"
