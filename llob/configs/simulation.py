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
        - duration: Physical simulation time (in arbitrary units)
        - n_frames: Number of output frames/measurements
        - dt = duration / n_frames: Time between output frames
        - dt_step = dx² / (2D): Elementary diffusion timestep
        - steps_per_frame = int(dt / dt_step): Diffusion steps per output frame
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
    alpha: float = Field(
        ge=0,
        default=0.0,
        description="Spread-sensitivity coefficient for deposition rate (dimensionless). "
        "Effective rate = lambd * (1 + alpha * spread_ticks).",
    )
    duration: float = Field(default=100.0, gt=0, description="Total physical simulation time")
    n_frames: int = Field(default=100, gt=0, description="Number of output frames")
    metaorder: list[float] | None = Field(
        default=None,
        description="Metaorder intensity over time. If length 1, expanded to n_frames with frame_start/frame_end",
    )
    frame_start: int | None = Field(
        default=None,
        description="Frame index when metaorder starts. Defaults to 0",
    )
    frame_end: int | None = Field(
        default=None,
        description="Frame index when metaorder ends. Defaults to n_frames",
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
        # Validate frame_start and frame_end are within bounds
        frame_start = self.frame_start if self.frame_start is not None else 0
        frame_end = self.frame_end if self.frame_end is not None else self.n_frames

        if frame_start < 0:
            raise ValueError(f"frame_start ({frame_start}) must be >= 0")
        if frame_end > self.n_frames:
            raise ValueError(f"frame_end ({frame_end}) must be <= n_frames ({self.n_frames})")
        if frame_start >= frame_end:
            raise ValueError(f"frame_start ({frame_start}) must be < frame_end ({frame_end})")

        # Validate metaorder length if provided and not length 1
        if self.metaorder is not None and len(self.metaorder) > 1:
            if len(self.metaorder) != self.n_frames:
                raise ValueError(
                    f"metaorder length ({len(self.metaorder)}) must equal n_frames ({self.n_frames}) "
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
    def effective_frame_start(self) -> int:
        """Get effective frame_start (0 if not set)."""
        return self.frame_start if self.frame_start is not None else 0

    @property
    def effective_frame_end(self) -> int:
        """Get effective frame_end (n_frames if not set)."""
        return self.frame_end if self.frame_end is not None else self.n_frames

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
        return self.duration / self.n_frames

    @property
    def dt_step(self) -> float:
        """Elementary diffusion timestep for numerical stability."""
        return self.grid.dx**2 / (2 * self.D)

    @property
    def steps_per_frame(self) -> int:
        """Number of diffusion steps per output frame."""
        return int(self.dt / self.dt_step)

    @property
    def J(self) -> float:
        """Steady-state order flow (D * L_max)."""
        return self.D * self.L_max

    def get_full_metaorder(self) -> npt.NDArray[np.float64]:
        """Expand metaorder to full length n_frames array."""
        if self.metaorder is None:
            return np.zeros(self.n_frames, dtype=np.float64)

        metaorder_arr = np.asarray(self.metaorder, dtype=np.float64)
        if len(metaorder_arr) == 1:
            full = np.zeros(self.n_frames, dtype=np.float64)
            full[self.effective_frame_start : self.effective_frame_end] = metaorder_arr[0]
            return full
        return metaorder_arr

    @classmethod
    def from_standard_parameters(
        cls,
        participation_rate: float,
        model_type: Literal["discrete", "continuous"],
        xmin: float | None = None,
        xmax: float | None = None,
        n_frames: int | None = None,
        duration: float | None = None,
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
            n_frames: Number of output frames. Default: 100.
            duration: Total physical simulation time. Default: 100.0.
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
        if n_frames is None:
            n_frames = 100
        if duration is None:
            duration = 100.0

        impact_estimate = np.sqrt(2 * abs(m0) * duration / L) if L > 0 else 10.0
        default_boundary = max(5 * impact_estimate, 50.0)

        if xmin is None:
            xmin = -default_boundary
        if xmax is None:
            xmax = default_boundary

        n_grid = 100

        grid = GridConfig(xmin=xmin, xmax=xmax, n_grid=n_grid)

        return cls(
            model_type=model_type,
            grid=grid,
            D=D,
            L=L,
            nu=nu,
            duration=duration,
            n_frames=n_frames,
            metaorder=[m0],
            **kwargs,
        )
