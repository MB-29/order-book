"""
Monte Carlo simulations of noisy LLOB models.
"""

from typing import Any, Optional, Self

import numpy as np
import numpy.typing as npt
from fbm import fgn
from tqdm.auto import tqdm

from .simulation import Simulation


class MonteCarlo:
    """
    Ensemble of LLOB simulations driven by fractional Gaussian noise metaorders.

    The per-frame metaorder intensity is ``m0 + m1 * fgn(hurst)``. After
    running, ensemble statistics are stored on the instance and returned by
    ``gather_results()``.
    """

    def __init__(
        self,
        N_samples: int,
        m0: float,
        m1: float,
        hurst: float,
        simulation_args: dict[str, Any],
        measurement_slice: int = 1,
        sample_measurements: Optional[list[str]] = None,
    ) -> None:
        self.N_samples = N_samples
        self.m0 = m0
        self.m1 = m1
        self.hurst = hurst
        self.simulation_args = dict(simulation_args)
        self.duration = self.simulation_args["duration"]
        self.n_frames = self.simulation_args["n_frames"]
        self.measurement_slice = measurement_slice
        self.sample_measurements = sample_measurements or []

        self.measured_quantities = self.simulation_args.get("measured_quantities", [])
        self.measurement_indices = self.simulation_args.get("measurement_indices", [])

        self.noisy_metaorders = np.zeros((self.n_frames, N_samples))
        self.price_samples = np.zeros((self.n_frames, N_samples))
        self.ask_samples = np.zeros((self.n_frames, N_samples))
        self.bid_samples = np.zeros((self.n_frames, N_samples))
        self.measured_samples: dict[str, list[Any]] = {
            q: [] for q in self.measured_quantities
        }

        # Statistics (populated by compute_statistics)
        self.price_mean: npt.NDArray[np.float64] = np.zeros(self.n_frames)
        self.price_variance: npt.NDArray[np.float64] = np.zeros(self.n_frames)
        self.ask_mean: npt.NDArray[np.float64] = np.zeros(self.n_frames)
        self.ask_variance: npt.NDArray[np.float64] = np.zeros(self.n_frames)
        self.bid_mean: npt.NDArray[np.float64] = np.zeros(self.n_frames)
        self.bid_variance: npt.NDArray[np.float64] = np.zeros(self.n_frames)
        self.measurement_means: dict[str, npt.NDArray[np.float64]] = {}
        self.measurement_vars: dict[str, npt.NDArray[np.float64]] = {}

        self.simulation: Optional[Simulation] = None

    @classmethod
    def from_params(
        cls,
        N_samples: int,
        noise_args: dict[str, Any],
        simulation_args: dict[str, Any],
    ) -> Self:
        return cls(
            N_samples=N_samples,
            m0=noise_args.get("m0", 0.0),
            m1=noise_args.get("m1", 0.0),
            hurst=noise_args["hurst"],
            simulation_args=simulation_args,
            measurement_slice=simulation_args.get("measurement_slice", 1),
            sample_measurements=simulation_args.get("sample_measurements", []),
        )

    def generate_noise(self) -> None:
        """Generate fractional Gaussian noise metaorder samples."""
        self.noisy_metaorders = np.full(
            (self.n_frames, self.N_samples), self.m0, dtype=float
        )
        if self.m1 == 0:
            return
        for k in range(self.N_samples):
            self.noisy_metaorders[:, k] += self.m1 * fgn(
                n=self.n_frames, hurst=self.hurst, length=self.duration
            )

    def run(self) -> None:
        """Run all Monte Carlo samples."""
        self.generate_noise()

        args = dict(self.simulation_args)
        self.simulation = Simulation.from_params(**args)
        print(self.simulation)

        for k in tqdm(range(self.N_samples)):
            args["metaorder"] = self.noisy_metaorders[:, k]
            self.simulation = Simulation.from_params(**args)
            self.simulation.run()

            self.price_samples[:, k] = self.simulation.prices
            self.ask_samples[:, k] = self.simulation.asks
            self.bid_samples[:, k] = self.simulation.bids

            for quantity in self.measured_quantities:
                self.measured_samples[quantity].append(
                    np.copy(self.simulation.measurements[quantity])
                )

        self.compute_statistics()

    def compute_statistics(self) -> None:
        self.price_mean = self.price_samples.mean(axis=1)
        self.ask_mean = self.ask_samples.mean(axis=1)
        self.bid_mean = self.bid_samples.mean(axis=1)
        self.price_variance = self.price_samples.var(axis=1)
        self.ask_variance = self.ask_samples.var(axis=1)
        self.bid_variance = self.bid_samples.var(axis=1)

        for quantity in self.measured_quantities:
            samples = np.array(self.measured_samples[quantity])
            self.measurement_means[quantity] = np.mean(samples, axis=0)
            self.measurement_vars[quantity] = np.var(samples, axis=0)

    def gather_results(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "price_mean": self.price_mean,
            "price_variance": self.price_variance,
            "ask_mean": self.ask_mean,
            "ask_variance": self.ask_variance,
            "bid_mean": self.bid_mean,
            "bid_variance": self.bid_variance,
            "m0": self.m0,
            "m1": self.m1,
            "hurst": self.hurst,
            "N_samples": self.N_samples,
            "params": self.simulation_args,
        }
        for quantity in self.measured_quantities:
            result[f"{quantity}_mean"] = self.measurement_means[quantity]
            result[f"{quantity}_variance"] = self.measurement_vars[quantity]

        for sample_measurement in self.sample_measurements:
            samples = getattr(self, f"{sample_measurement}_samples")
            result[sample_measurement] = [
                samples[t : t + self.measurement_slice, :]
                for t in self.measurement_indices
            ]
        return result
