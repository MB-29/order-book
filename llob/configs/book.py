"""Book configuration models for order book simulations."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from llob.configs.grid import GridConfig


class BookConfig(BaseModel):
    """Base configuration for order book models.

    Contains common parameters shared across all book types.
    """

    grid: GridConfig = Field(description="Spatial grid configuration")
    D: float = Field(gt=0, description="Diffusion constant")
    L: float = Field(gt=0, description="Latent liquidity / order density slope")

    model_config = {"frozen": True}

    @property
    def J(self) -> float:
        """Steady-state order flow (D * L)."""
        return self.D * self.L

    @property
    def dt(self) -> float:
        """Time step for numerical stability (dx^2 / 2D)."""
        return self.grid.dx**2 / (2 * self.D)


class LimitOrdersConfig(BaseModel):
    """Configuration for single-side limit order dynamics.

    Defines parameters for deposition, cancellation, and diffusion
    of limit orders on one side of the book.
    """

    grid: GridConfig = Field(description="Spatial grid configuration")
    side: Literal["bid", "ask"] = Field(description="Which side of the book")
    D: float = Field(gt=0, description="Diffusion constant")
    L: float | None = Field(
        default=None,
        description="Latent liquidity. Computed from lambd/sqrt(nu*D) if not provided",
    )
    nu: float = Field(ge=0, default=0.0, description="Cancellation rate parameter")
    lambd: float = Field(ge=0, default=0.0, description="Deposition intensity parameter")
    alpha: float = Field(
        ge=0,
        default=0.0,
        description="Spread-sensitivity coefficient for deposition rate (dimensionless). "
        "Effective rate = lambd * (1 + alpha * spread_ticks).",
    )
    initial_density: Literal["stationary", "linear", "empty"] = Field(
        default="stationary",
        description="Initial density profile for the order book",
    )
    boundary_conditions: Literal["flat", "linear"] = Field(
        default="flat",
        description="Boundary conditions at the grid edges",
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_and_compute_L(self) -> "LimitOrdersConfig":
        if self.L is None:
            if self.nu * self.D <= 0:
                raise ValueError(
                    "Cannot compute L without nu*D > 0. Provide L explicitly or set nu > 0."
                )
            # We can't modify frozen model, so this validation just checks
            # The actual computation happens in the class using this config
        return self

    def compute_L(self) -> float:
        """Compute L from lambd and nu if not provided."""
        if self.L is not None:
            return self.L
        return self.lambd / np.sqrt(self.nu * self.D)

    @property
    def dt(self) -> float:
        """Time step for numerical stability."""
        return self.grid.dx**2 / (2 * self.D)

    @property
    def boundary_flow(self) -> float:
        """Boundary flow based on boundary conditions."""
        L_val = self.L if self.L is not None else self.compute_L()
        return L_val if self.boundary_conditions == "linear" else 0.0


class DiscreteBookConfig(BaseModel):
    """Configuration for agent-based discrete order book model.

    Includes cancellation dynamics and deposition.
    """

    grid: GridConfig = Field(description="Spatial grid configuration")
    D: float = Field(gt=0, description="Diffusion constant")
    L: float | None = Field(
        default=None,
        description="Latent liquidity. Computed from lambd/sqrt(nu*D) if not provided",
    )
    nu: float = Field(gt=0, description="Cancellation rate parameter (must be > 0)")
    lambd: float = Field(gt=0, description="Deposition intensity parameter")
    alpha: float = Field(
        ge=0,
        default=0.0,
        description="Spread-sensitivity coefficient for deposition rate (dimensionless). "
        "Effective rate = lambd * (1 + alpha * spread_ticks).",
    )
    initial_density: Literal["stationary", "linear", "empty"] = Field(
        default="stationary",
        description="Initial density profile",
    )
    boundary_conditions: Literal["flat", "linear"] = Field(
        default="flat",
        description="Boundary conditions at grid edges",
    )

    model_config = {"frozen": True}

    def compute_L(self) -> float:
        """Compute L from lambd and nu if not provided."""
        if self.L is not None:
            return self.L
        return self.lambd / np.sqrt(self.nu * self.D)

    @property
    def dt(self) -> float:
        """Time step for numerical stability."""
        return self.grid.dx**2 / (2 * self.D)


class LinearDiscreteBookConfig(BaseModel):
    """Configuration for simplified linear regime discrete book.

    No cancellation dynamics (nu=0), purely diffusive.
    """

    grid: GridConfig = Field(description="Spatial grid configuration")
    D: float = Field(gt=0, description="Diffusion constant")
    L: float = Field(gt=0, description="Latent liquidity / order density slope")

    model_config = {"frozen": True}

    @property
    def dt(self) -> float:
        """Time step for numerical stability."""
        return self.grid.dx**2 / (2 * self.D)


class LinearContinuousBookConfig(BaseModel):
    """Configuration for PDE-based continuous order book model.

    Uses analytical/semi-analytical solutions for the linear regime.
    """

    grid: GridConfig = Field(description="Spatial grid configuration")
    D: float = Field(gt=0, description="Diffusion constant")
    L: float = Field(gt=0, description="Latent liquidity / order density slope")

    model_config = {"frozen": True}

    @property
    def dt(self) -> float:
        """Time step for numerical stability."""
        return self.grid.dx**2 / (2 * self.D)

    @property
    def J(self) -> float:
        """Steady-state order flow."""
        return self.D * self.L


class MultiDiscreteBookConfig(BaseModel):
    """Configuration for multi-actor discrete order book.

    Supports multiple participant types with different parameters.
    """

    grid: GridConfig = Field(description="Spatial grid configuration")
    D: float = Field(gt=0, description="Diffusion constant")
    L_list: list[float] = Field(
        min_length=1,
        description="Latent liquidity values for each actor type",
    )
    nu_list: list[float] = Field(
        min_length=1,
        description="Cancellation rates for each actor type",
    )
    lambd_list: list[float] = Field(
        min_length=1,
        description="Deposition intensities for each actor type",
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_list_lengths(self) -> "MultiDiscreteBookConfig":
        n_actors = len(self.L_list)
        if len(self.nu_list) != n_actors:
            raise ValueError(
                f"nu_list length ({len(self.nu_list)}) must match L_list length ({n_actors})"
            )
        if len(self.lambd_list) != n_actors:
            raise ValueError(
                f"lambd_list length ({len(self.lambd_list)}) must match L_list length ({n_actors})"
            )
        return self

    @property
    def n_actors(self) -> int:
        """Number of actor types."""
        return len(self.L_list)

    @property
    def dt(self) -> float:
        """Time step for numerical stability."""
        return self.grid.dx**2 / (2 * self.D)
