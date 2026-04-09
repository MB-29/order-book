"""
Equilibrium ensemble simulations for the unbiased LLOB.

Runs Monte Carlo simulations in the unbiased regime (m0=0, m1>0)
with fractional Gaussian noise metaorders.
"""

from typing import Any

import numpy as np

from .monte_carlo import MonteCarlo


def make_unbiased_mc(
    m1: float,
    hurst: float,
    n_samples: int,
    simulation_params: dict[str, Any],
    profile_indices: list[int] | None = None,
) -> MonteCarlo:
    """
    Build a MonteCarlo instance for the unbiased regime (m0=0).

    Args:
        m1: Noise magnitude for the fractional Gaussian metaorder.
        hurst: Hurst exponent (typically 0.75).
        n_samples: Number of Monte Carlo samples.
        simulation_params: Base simulation parameters (model_type, D, L, nu,
            duration, n_frames, xmin, xmax, n_grid, alpha, ...).
        profile_indices: Frame indices at which to snapshot the orderbook.
            If None, uses 10 evenly spaced frames.

    Returns:
        Configured MonteCarlo instance (call .run() to execute).
    """
    params = dict(simulation_params)
    n_frames = params["n_frames"]

    if profile_indices is None:
        profile_indices = np.linspace(0, n_frames - 1, 10, dtype=int).tolist()

    params["measured_quantities"] = ["ask_volumes", "bid_volumes"]
    params["measurement_indices"] = profile_indices

    noise_args = {"m0": 0.0, "m1": m1, "hurst": hurst}

    return MonteCarlo.from_params(
        N_samples=n_samples,
        noise_args=noise_args,
        simulation_args=params,
    )
