"""
Monte Carlo simulations of noisy LLOB models.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from fbm import fgn
from retry import retry
from tqdm.auto import tqdm

from .simulation import Simulation


class MonteCarlo:
    """
    Monte Carlo simulation with fractional Gaussian noise metaorders.

    Runs an ensemble of simulations with stochastic metaorder processes,
    computing statistics over the ensemble.

    Attributes:
        N_samples: Number of Monte Carlo samples.
        price_mean: Mean price trajectory over ensemble.
        price_variance: Variance of price trajectory over ensemble.
        ask_mean: Mean ask price trajectory.
        bid_mean: Mean bid price trajectory.
    """

    def __init__(
        self,
        N_samples: int,
        T: int,
        Nt: int,
        m0: float,
        m1: float,
        hurst: float,
        simulation_args: dict[str, Any],
        measured_quantities: list[str],
        measurement_indices: list[int],
        measurement_slice: int,
        sample_measurements: list[str],
    ) -> None:
        """
        Initialize a MonteCarlo simulation.

        Use `from_params` classmethod for convenient construction.

        Args:
            N_samples: Number of Monte Carlo samples.
            T: Total time steps.
            Nt: Number of output time points.
            m0: Mean metaorder intensity.
            m1: Metaorder noise standard deviation.
            hurst: Hurst exponent for fractional Gaussian noise.
            simulation_args: Arguments passed to Simulation.from_params.
            measured_quantities: Quantities to measure at each sample.
            measurement_indices: Time indices for measurements.
            measurement_slice: Slice size for sample measurements.
            sample_measurements: Names of sample-level measurements.
        """
        self.N_samples = N_samples
        self.T = T
        self.Nt = Nt
        self.m0 = m0
        self.m1 = m1
        self.hurst = hurst
        self.simulation_args = simulation_args
        self.measured_quantities = measured_quantities
        self.measurement_indices = measurement_indices
        self.measurement_slice = measurement_slice
        self.sample_measurements = sample_measurements

        # Noise arrays
        self.noise = np.zeros((T, N_samples))
        self.noisy_metaorders = np.zeros((T, N_samples))
        self.scale = m1

        # Output arrays
        self.price_samples = np.zeros((T, N_samples))
        self.ask_samples = np.zeros((T, N_samples))
        self.bid_samples = np.zeros((T, N_samples))

        # Measurement storage
        self.measured_samples: dict[str, list[Any]] = {
            q: [] for q in measured_quantities
        }

        # Statistics (computed after run)
        self.price_mean: npt.NDArray[np.float64] = np.zeros(T)
        self.price_variance: npt.NDArray[np.float64] = np.zeros(T)
        self.ask_mean: npt.NDArray[np.float64] = np.zeros(T)
        self.ask_variance: npt.NDArray[np.float64] = np.zeros(T)
        self.bid_mean: npt.NDArray[np.float64] = np.zeros(T)
        self.bid_variance: npt.NDArray[np.float64] = np.zeros(T)
        self.measurement_means: dict[str, npt.NDArray[np.float64]] = {}
        self.measurement_vars: dict[str, npt.NDArray[np.float64]] = {}

        # Reference simulation (set during run)
        self.simulation: Simulation | None = None

    @classmethod
    def from_params(
        cls,
        N_samples: int,
        noise_args: dict[str, Any],
        simulation_args: dict[str, Any],
    ) -> MonteCarlo:
        """
        Create a MonteCarlo simulation from parameters.

        This maintains backward compatibility with the original API.

        Args:
            N_samples: Number of Monte Carlo samples.
            noise_args: Dictionary with keys:
                - m0: Mean metaorder intensity (default: 0)
                - m1: Metaorder noise std (default: 0)
                - hurst: Hurst exponent (required)
            simulation_args: Arguments passed to Simulation.from_params.
                Must include T and Nt.

        Returns:
            Configured MonteCarlo instance.
        """
        T = simulation_args["T"]
        Nt = simulation_args["Nt"]
        m0 = noise_args.get("m0", 0.0)
        m1 = noise_args.get("m1", 0.0)
        hurst = noise_args["hurst"]

        measured_quantities = simulation_args.get("measured_quantities", [])
        measurement_indices = simulation_args.get("measurement_indices", [])
        measurement_slice = simulation_args.get("measurement_slice", 1)
        sample_measurements = simulation_args.get("sample_measurements", [])

        return cls(
            N_samples=N_samples,
            T=T,
            Nt=Nt,
            m0=m0,
            m1=m1,
            hurst=hurst,
            simulation_args=simulation_args,
            measured_quantities=measured_quantities,
            measurement_indices=measurement_indices,
            measurement_slice=measurement_slice,
            sample_measurements=sample_measurements,
        )

    def generate_noise(self) -> None:
        """Generate fractional Gaussian noise metaorder samples."""
        self.noisy_metaorders = np.full((self.T, self.N_samples), self.m0, dtype=float)

        if self.m1 == 0:
            return

        # Generate standard fractional Gaussian noise for each sample
        for sample_index in range(self.N_samples):
            self.noise[:, sample_index] = fgn(n=self.T, hurst=self.hurst, length=self.T)

        # Scale and add to mean
        self.scale = self.m1
        self.noisy_metaorders += self.scale * self.noise

        order_mean = float(self.noisy_metaorders.mean())
        order_var = float(self.noisy_metaorders.var(axis=1).mean())
        print(
            f"Generated meta-order has mean {order_mean:.2f} and variance {order_var:.2f}"
        )

    def run(self) -> None:
        """Run all Monte Carlo samples."""
        self.generate_noise()

        # Create reference simulation for printing
        args = dict(self.simulation_args)
        self.simulation = Simulation.from_params(**args)
        print(self.simulation)

        for k in tqdm(range(self.N_samples)):
            args["metaorder"] = self.noisy_metaorders[:, k]
            self._try_running(args)

            if self.simulation is not None:
                self.price_samples[:, k] = self.simulation.prices
                self.ask_samples[:, k] = self.simulation.asks
                self.bid_samples[:, k] = self.simulation.bids

                for quantity in self.measured_quantities:
                    self.measured_samples[quantity].append(
                        np.copy(self.simulation.measurements[quantity])
                    )

        self.compute_statistics()

    def compute_statistics(self) -> None:
        """Compute ensemble statistics from samples."""
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
        """
        Gather all results into a dictionary.

        Returns:
            Dictionary containing means, variances, and metadata.
        """
        result: dict[str, Any] = {
            "price_mean": self.price_mean,
            "price_variance": self.price_variance,
            "ask_mean": self.ask_mean,
            "ask_variance": self.ask_variance,
            "bid_mean": self.bid_mean,
            "bid_variance": self.bid_variance,
            "m1": self.m1,
            "N_samples": self.N_samples,
            "m0": self.m0,
            "hurst": self.hurst,
            "params": self.simulation_args,
        }

        for quantity in self.measured_quantities:
            result[f"{quantity}_mean"] = self.measurement_means[quantity]
            result[f"{quantity}_variance"] = self.measurement_vars[quantity]

        for sample_measurement in self.sample_measurements:
            measurement = [
                getattr(self, f"{sample_measurement}_samples")[
                    time_index : time_index + self.measurement_slice, :
                ]
                for time_index in self.measurement_indices
            ]
            result[sample_measurement] = measurement

        return result

    @retry((ValueError, IndexError), tries=1)
    def _try_running(self, args: dict[str, Any]) -> None:
        """Run a simulation with retry on failure."""
        self.simulation = Simulation.from_params(**args)
        self.simulation.run()
