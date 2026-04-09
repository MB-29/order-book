"""
Perturbation ensemble simulations for the biased LLOB.

Runs Monte Carlo simulations with a small deterministic metaorder m0
on top of fractional Gaussian noise m1, measuring the average price
shift and its variance.
"""

from typing import Any

from .monte_carlo import MonteCarlo


def make_perturbation_mc(
    m0: float,
    m1: float,
    hurst: float,
    n_samples: int,
    simulation_params: dict[str, Any],
) -> MonteCarlo:
    """
    Build a MonteCarlo instance for the perturbation regime (m0 > 0, m1 > 0).

    Args:
        m0: Deterministic metaorder rate.
        m1: Noise magnitude for the fractional Gaussian metaorder.
        hurst: Hurst exponent (typically 0.75).
        n_samples: Number of Monte Carlo samples.
        simulation_params: Base simulation parameters.

    Returns:
        Configured MonteCarlo instance (call .run() to execute).
    """
    params = dict(simulation_params)
    noise_args = {"m0": m0, "m1": m1, "hurst": hurst}

    return MonteCarlo.from_params(
        N_samples=n_samples,
        noise_args=noise_args,
        simulation_args=params,
    )
