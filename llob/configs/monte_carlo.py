"""Monte Carlo configuration models."""

from pydantic import BaseModel, Field, model_validator

from llob.configs.simulation import SimulationConfig


class NoiseConfig(BaseModel):
    """Configuration for fractional Gaussian noise metaorder generation.

    Defines the stochastic process parameters for noisy metaorders.
    """

    m0: float = Field(default=0.0, description="Mean metaorder intensity")
    m1: float = Field(default=0.0, ge=0, description="Metaorder noise standard deviation")
    hurst: float = Field(
        ge=0.0,
        le=1.0,
        description="Hurst exponent for fractional Gaussian noise (0.5 = Brownian)",
    )

    model_config = {"frozen": True}

    @property
    def is_deterministic(self) -> bool:
        """Whether the noise is effectively zero (deterministic metaorder)."""
        return self.m1 == 0.0


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo ensemble simulations.

    Runs multiple simulations with stochastic metaorder processes.
    """

    N_samples: int = Field(gt=0, description="Number of Monte Carlo samples")
    noise: NoiseConfig = Field(description="Noise generation configuration")
    simulation: SimulationConfig = Field(description="Base simulation configuration")
    measurement_slice: int = Field(
        default=1,
        gt=0,
        description="Slice size for sample measurements",
    )
    sample_measurements: list[str] = Field(
        default_factory=list,
        description="Names of sample-level measurements to collect",
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_metaorder_override(self) -> "MonteCarloConfig":
        # Monte Carlo generates its own metaorders from noise config,
        # so we warn if simulation config has a non-trivial metaorder
        if self.simulation.metaorder is not None and len(self.simulation.metaorder) > 1:
            import warnings

            warnings.warn(
                "MonteCarloConfig simulation has explicit metaorder which will be "
                "overridden by noise-generated metaorders",
                stacklevel=2,
            )
        return self

    @property
    def T(self) -> int:
        """Total time steps (from simulation config)."""
        return self.simulation.T

    @property
    def Nt(self) -> int:
        """Number of output time points (from simulation config)."""
        return self.simulation.Nt

    def to_noise_args(self) -> dict:
        """Convert noise config to legacy noise_args dict."""
        return {
            "m0": self.noise.m0,
            "m1": self.noise.m1,
            "hurst": self.noise.hurst,
        }

    def to_simulation_args(self) -> dict:
        """Convert simulation config to legacy simulation_args dict.

        Returns a dict compatible with the original MonteCarlo.from_params API.
        """
        return {
            "model_type": self.simulation.model_type,
            "T": self.simulation.T,
            "Nt": self.simulation.Nt,
            "xmin": self.simulation.grid.xmin,
            "xmax": self.simulation.grid.xmax,
            "Nx": self.simulation.grid.Nx,
            "D": self.simulation.D,
            "L": self.simulation.L,
            "nu": self.simulation.nu,
            "n_start": self.simulation.effective_n_start,
            "n_end": self.simulation.effective_n_end,
            "price_formula": self.simulation.price_formula,
            "measured_quantities": self.simulation.measured_quantities,
            "measurement_indices": self.simulation.measurement_indices,
            "measurement_slice": self.measurement_slice,
            "sample_measurements": self.sample_measurements,
        }
