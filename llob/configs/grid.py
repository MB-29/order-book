"""Grid configuration for spatial discretization."""

from pydantic import BaseModel, Field, model_validator


class GridConfig(BaseModel):
    """Configuration for spatial grid discretization.

    Defines the price interval and resolution for order book simulations.
    """

    xmin: float = Field(description="Lower bound of the price interval")
    xmax: float = Field(description="Upper bound of the price interval")
    Nx: int = Field(gt=0, description="Number of spatial grid points")

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_bounds(self) -> "GridConfig":
        if self.xmin >= self.xmax:
            raise ValueError(f"xmin ({self.xmin}) must be less than xmax ({self.xmax})")
        return self

    @property
    def price_range(self) -> float:
        """Total width of the price interval."""
        return self.xmax - self.xmin

    @property
    def dx(self) -> float:
        """Spatial step size."""
        return self.price_range / self.Nx
