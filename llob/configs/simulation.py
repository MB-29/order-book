"""Simulation configuration model."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator, model_validator

from llob.configs.grid import GridConfig


class SimulationConfig(BaseModel):
    """Configuration for LLOB simulation.

    Orchestrates order book simulation with metaorder execution and price tracking.

    Time convention:
        - T: Physical simulation time (in arbitrary units)
        - Nt: Number of output frames/measurements
        - dt = T / Nt: Time between output frames
        - dt_diff = dx² / (2D): Elementary diffusion timestep
        - n_diff = int(dt / dt_diff): Diffusion steps per output frame
    """

    model_type: Literal["discrete", "continuous"] = Field(
        description="Type of order book model to use"
    )
    grid: GridConfig = Field(description="Spatial grid configuration")
    D: float = Field(gt=0, description="Diffusion constant")
    L: float | list[float] = Field(
        description="Latent liquidity. Scalar for single book, list for multi-actor"
    )
    nu: float | list[float] = Field(
        default=0.0,
        description="Cancellation rate. Scalar or list matching L for multi-actor",
    )
    T: float = Field(default=100.0, gt=0, description="Total physical simulation time")
    Nt: int = Field(default=100, gt=0, description="Number of output frames")
    metaorder: list[float] | None = Field(
        default=None,
        description="Metaorder intensity over time. If length 1, expanded to Nt with n_start/n_end",
    )
    n_start: int | None = Field(
        default=None,
        description="Frame index when metaorder starts. Defaults to 0",
    )
    n_end: int | None = Field(
        default=None,
        description="Frame index when metaorder ends. Defaults to Nt",
    )
    price_formula: Literal["middle", "best_ask", "best_bid", "vwap"] = Field(
        default="middle",
        description="Formula for computing price from bid/ask",
    )
    measured_quantities: list[str] = Field(
        default_factory=list,
        description="Quantities to measure during simulation",
    )
    measurement_indices: list[int] = Field(
        default_factory=list,
        description="Time indices at which to record measurements",
    )

    model_config = {"frozen": True}

    @field_validator("metaorder", mode="before")
    @classmethod
    def convert_metaorder(cls, v: list[float] | npt.NDArray | None) -> list[float] | None:
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @model_validator(mode="after")
    def set_defaults_and_validate(self) -> "SimulationConfig":
        # Validate n_start and n_end are within bounds
        n_start = self.n_start if self.n_start is not None else 0
        n_end = self.n_end if self.n_end is not None else self.Nt

        if n_start < 0:
            raise ValueError(f"n_start ({n_start}) must be >= 0")
        if n_end > self.Nt:
            raise ValueError(f"n_end ({n_end}) must be <= Nt ({self.Nt})")
        if n_start >= n_end:
            raise ValueError(f"n_start ({n_start}) must be < n_end ({n_end})")

        # Validate metaorder length if provided and not length 1
        if self.metaorder is not None and len(self.metaorder) > 1:
            if len(self.metaorder) != self.Nt:
                raise ValueError(
                    f"metaorder length ({len(self.metaorder)}) must equal Nt ({self.Nt}) "
                    "or be length 1"
                )

        # Validate L and nu consistency for multi-actor
        if isinstance(self.L, list):
            if isinstance(self.nu, list) and len(self.nu) != len(self.L):
                raise ValueError(
                    f"nu list length ({len(self.nu)}) must match L list length ({len(self.L)})"
                )

        return self

    @property
    def effective_n_start(self) -> int:
        """Get effective n_start (0 if not set)."""
        return self.n_start if self.n_start is not None else 0

    @property
    def effective_n_end(self) -> int:
        """Get effective n_end (Nt if not set)."""
        return self.n_end if self.n_end is not None else self.Nt

    @property
    def is_multi_book(self) -> bool:
        """Whether this is a multi-actor configuration."""
        return isinstance(self.L, list)

    @property
    def L_max(self) -> float:
        """Maximum latent liquidity value."""
        if isinstance(self.L, list):
            return max(self.L)
        return self.L

    @property
    def dt(self) -> float:
        """Time between output frames (physical time / number of frames)."""
        return self.T / self.Nt

    @property
    def dt_diff(self) -> float:
        """Elementary diffusion timestep for numerical stability."""
        return self.grid.dx**2 / (2 * self.D)

    @property
    def n_diff(self) -> int:
        """Number of diffusion steps per output frame."""
        return int(self.dt / self.dt_diff)

    @property
    def J(self) -> float:
        """Steady-state order flow (D * L_max)."""
        return self.D * self.L_max

    def get_full_metaorder(self) -> npt.NDArray[np.float64]:
        """Expand metaorder to full length Nt array."""
        if self.metaorder is None:
            return np.zeros(self.Nt, dtype=np.float64)

        metaorder_arr = np.asarray(self.metaorder, dtype=np.float64)
        if len(metaorder_arr) == 1:
            full = np.zeros(self.Nt, dtype=np.float64)
            full[self.effective_n_start : self.effective_n_end] = metaorder_arr[0]
            return full
        return metaorder_arr

    @classmethod
    def from_standard_parameters(
        cls,
        participation_rate: float,
        model_type: Literal["discrete", "continuous"],
        xmin: float | None = None,
        xmax: float | None = None,
        Nt: int | None = None,
        T: float | None = None,
        **kwargs,
    ) -> "SimulationConfig":
        """Create config from a participation rate.

        This is a convenience method that computes appropriate parameters
        from the participation rate, similar to the original standard_parameters function.

        Args:
            participation_rate: The ratio m0 / (D * L), where m0 is the
                metaorder intensity.
            model_type: Either 'discrete' or 'continuous'.
            xmin: Lower bound of price interval. Default computed from participation_rate.
            xmax: Upper bound of price interval. Default computed from participation_rate.
            Nt: Number of output frames. Default: 100.
            T: Total physical simulation time. Default: 100.0.
            **kwargs: Additional parameters to override.

        Returns:
            Configured SimulationConfig instance.
        """
        # Standard parameter values
        D = 1.0
        L = 1.0
        nu = 0.0
        m0 = participation_rate * D * L

        # Defaults based on participation rate
        if Nt is None:
            Nt = 100
        if T is None:
            T = 100.0

        impact_estimate = np.sqrt(2 * abs(m0) * T / L) if L > 0 else 10.0
        default_boundary = max(5 * impact_estimate, 50.0)

        if xmin is None:
            xmin = -default_boundary
        if xmax is None:
            xmax = default_boundary

        Nx = 100

        grid = GridConfig(xmin=xmin, xmax=xmax, Nx=Nx)

        return cls(
            model_type=model_type,
            grid=grid,
            D=D,
            L=L,
            nu=nu,
            T=T,
            Nt=Nt,
            metaorder=[m0],
            **kwargs,
        )
