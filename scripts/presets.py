"""
Named parameter presets for LLOB experiments.

A ``Preset`` is a fully-specified runnable scenario: book parameters,
discretization grid, simulation horizon, and noise (``m0``, ``m1``, ``H``).
Monte-Carlo sample count is *not* part of a preset — it is an experiment-
level knob set in each experiment's ``config.py`` (and overridable on
the run-script CLI). Experiment ``config.py`` modules and the
``execution_animation.py`` script both consume presets, so a regime can
be examined either as a Monte-Carlo ensemble or as a single animated run
without re-typing parameters.

Noise is expressed as ``r0 = m0/J`` and ``r1 = m1/J`` (with ``J = D L``),
matching the parameterization used in CLAUDE.md and the original configs.

Naming convention: ``<family>_<regime>`` — e.g. ``impact_weak_noise``.
"""

from dataclasses import dataclass
from typing import Any

from llob import courant_dt


@dataclass(frozen=True)
class Preset:
    """A single runnable scenario."""

    description: str
    # Book
    D: float = 0.5
    L: float = 10.0
    nu: float = 0.0
    alpha: float = 0.0
    # Grid
    xmin: float = -100.0
    xmax: float = 100.0
    n_grid: int = 200
    # Simulation horizon
    duration: float = 100.0
    n_frames: int = 50
    model_type: str = "discrete"
    # Noise (expressed relative to J = D * L)
    r0: float = 0.0
    r1: float = 0.0
    hurst: float = 0.75

    @property
    def J(self) -> float:
        return self.D * self.L

    @property
    def m0(self) -> float:
        return self.r0 * self.J

    @property
    def m1(self) -> float:
        return self.r1 * self.J

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / self.n_grid

    @property
    def dt_step(self) -> float:
        return courant_dt(self.dx, self.D)

    def sim_params(self, **extra: Any) -> dict[str, Any]:
        """Kwargs for ``Simulation.from_params`` / ``MonteCarlo`` simulation_args."""
        params: dict[str, Any] = dict(
            model_type=self.model_type,
            D=self.D, L=self.L, nu=self.nu, alpha=self.alpha,
            duration=self.duration, n_frames=self.n_frames,
            n_grid=self.n_grid, xmin=self.xmin, xmax=self.xmax,
            dt_step=self.dt_step,
        )
        params.update(extra)
        return params

    def noise_args(self) -> dict[str, float]:
        """Kwargs for ``MonteCarlo.from_params(noise_args=...)``."""
        return {"m0": self.m0, "m1": self.m1, "hurst": self.hurst}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRESETS: dict[str, Preset] = {
    # --- Equilibrium family: unbiased noise (m0=0), sweep m1/J --------------
    "equilibrium_m1_half_J": Preset(
        description="Unbiased noise, m1 = J/2; tests pure-noise variance growth.",
        D=0.5, L=10.0, nu=0.1, alpha=0.5,
        xmin=-100.0, xmax=100.0, n_grid=200,
        duration=100.0, n_frames=50,
        r0=0.0, r1=0.5, hurst=0.75,
    ),
    "equilibrium_m1_J": Preset(
        description="Unbiased noise, m1 = J; tests pure-noise variance growth.",
        D=0.5, L=10.0, nu=0.1, alpha=0.5,
        xmin=-100.0, xmax=100.0, n_grid=200,
        duration=100.0, n_frames=50,
        r0=0.0, r1=1.0, hurst=0.75,
    ),

    # --- Perturbation family: low-participation sanity check ---------------
    "perturbation_low_participation": Preset(
        description="Low participation (r0=0.1, r1=0.01) over a long horizon.",
        D=0.5, L=10.0, nu=0.0, alpha=0.0,
        xmin=-100.0, xmax=100.0, n_grid=200,
        duration=200.0, n_frames=50,
        r0=0.1, r1=0.01, hurst=0.75,
    ),

    # --- Impact-regime sweep -----------------------------------------------
    # Shared grid: wide enough for r0=5 over T=100 (I_t ~ 22 << xmax).
    # L bumped to 100 to lift dq/n_steps above the integer floor.
    "impact_low_participation": Preset(
        description="r0=0.1, r1=0.1 — both small; diffusion dominates.",
        D=0.5, L=100.0, nu=0.0, alpha=0.0,
        xmin=-150.0, xmax=150.0, n_grid=400,
        duration=100.0, n_frames=100,
        r0=0.1, r1=0.1, hurst=0.75,
    ),
    "impact_low_p_strong_noise": Preset(
        description="r0=0.1, r1=1.0 — small bias, moderate noise.",
        D=0.5, L=100.0, nu=0.0, alpha=0.0,
        xmin=-150.0, xmax=150.0, n_grid=400,
        duration=100.0, n_frames=100,
        r0=0.1, r1=1.0, hurst=0.75,
    ),
    "impact_balanced": Preset(
        description="r0=1.0, r1=1.0 — bias and noise both at J.",
        D=0.5, L=100.0, nu=0.0, alpha=0.0,
        xmin=-150.0, xmax=150.0, n_grid=400,
        duration=100.0, n_frames=100,
        r0=1.0, r1=1.0, hurst=0.75,
    ),
    "impact_weak_noise": Preset(
        description="r0=5, r1=0.5 — high participation, weak noise correction.",
        D=0.5, L=100.0, nu=0.0, alpha=0.0,
        xmin=-150.0, xmax=150.0, n_grid=400,
        duration=100.0, n_frames=100,
        r0=5.0, r1=0.5, hurst=0.75,
    ),
    "impact_strong_noise": Preset(
        description="r0=0.5, r1=5 — bias diluted by strong noise; E[p] ~ sqrt(m0/m1) I_t.",
        D=0.5, L=100.0, nu=0.0, alpha=0.0,
        xmin=-150.0, xmax=150.0, n_grid=400,
        duration=100.0, n_frames=100,
        r0=0.5, r1=5.0, hurst=0.75,
    ),
    "impact_high_both": Preset(
        description="r0=5, r1=5 — both well above J, general high-participation case.",
        D=0.5, L=100.0, nu=0.0, alpha=0.0,
        xmin=-150.0, xmax=150.0, n_grid=400,
        duration=100.0, n_frames=100,
        r0=5.0, r1=5.0, hurst=0.75,
    ),
    "impact_pure_noise": Preset(
        description="r0=0, r1=1 — competing-metaorder symmetry: E[p] = 0.",
        D=0.5, L=100.0, nu=0.0, alpha=0.0,
        xmin=-150.0, xmax=150.0, n_grid=400,
        duration=100.0, n_frames=100,
        r0=0.0, r1=1.0, hurst=0.75,
    ),
}


def get_preset(name: str) -> Preset:
    """Look up a preset by name; raise with the available list on miss."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {sorted(PRESETS)}"
        ) from None


def list_presets() -> list[str]:
    return sorted(PRESETS)
