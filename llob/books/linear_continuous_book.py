
"""
Continuous order book using PDE discretization in the LLOB framework.
"""

import warnings
from typing import Any, Optional, Self

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .diffusion_schemes import theta_scheme_iteration


class LinearContinuousBook:
    """
    Continuous order book using PDE discretization.

    Models the order book density as a continuous field, solving the
    reaction-diffusion PDE using finite differences. This is the
    hydrodynamic limit of the discrete model.

    Attributes:
        density: Algebraic order density at each price level.
        X: Price grid array.
        dx: Price grid spacing.
        best_ask: Current best ask price.
        best_bid: Current best bid price.
        best_ask_index: Index of best ask in grid.
        best_bid_index: Index of best bid in grid.
    """

    def __init__(
        self,
        D: float,
        L: float,
        X: npt.NDArray[np.float64],
        dx: float,
        density: npt.NDArray[np.float64],
        resolution_volume: float,
    ) -> None:
        """
        Initialize a LinearContinuousBook with pre-computed values.

        Use `from_params` classmethod for convenient construction.

        Args:
            D: Diffusion constant.
            L: Order density slope (latent liquidity).
            X: Price grid array.
            dx: Price grid spacing.
            density: Initial algebraic order density.
            resolution_volume: Minimum volume for price detection.
        """
        self.D = D
        self.L = L
        self.X = X
        self.dx = dx
        self.density = density
        self.resolution_volume = resolution_volume

        # Derived values
        self.n_grid = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])
        self.price_range = (self.xmax - self.xmin) / 2
        self.boundary_distance = min(abs(self.xmin), abs(self.xmax))
        self.J = D * L

        # Initialize price tracking
        self.best_ask: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask_index: int = 0
        self.best_bid_index: int = 0
        self.best_ask_volume: float = 0.0
        self.best_bid_volume: float = 0.0
        self.update_prices()

        # Set volumes after initial price update
        # +1 / -1 ensure corresponding volume isn't partially consumed
        self.best_ask_volume = self.density[self.best_ask_index + 1]
        self.best_bid_volume = self.density[self.best_bid_index - 1]

        # Animation state
        self.density_ax: Optional[Axes] = None
        self.density_line: Optional[Line2D] = None
        self.best_ask_axis: Optional[Line2D] = None
        self.best_bid_axis: Optional[Line2D] = None

    @classmethod
    def from_params(
        cls,
        D: float,
        L: float,
        xmin: float,
        xmax: float,
        n_grid: int = 1000,
    ) -> Self:
        """
        Create a LinearContinuousBook from raw parameters.

        Args:
            D: Diffusion constant.
            L: Order density slope (latent liquidity).
            xmin: Price interval lower bound.
            xmax: Price interval upper bound.
            n_grid: Number of price grid points.

        Returns:
            Configured LinearContinuousBook instance.
        """
        # Compute grid
        X, dx = np.linspace(xmin, xmax, num=n_grid, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        # Initial density: linear profile
        density = -L * X

        # Resolution volume for price detection
        resolution_volume = L * dx**2
        if resolution_volume > L * dx * dx:
            warnings.warn(
                "Resolution volume may be too large and lead to an inaccurate price",
                stacklevel=2,
            )

        return cls(
            D=D,
            L=L,
            X=X,
            dx=dx,
            density=density,
            resolution_volume=resolution_volume,
        )

    def initial_density(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute initial linear density profile.

        Args:
            x: Price grid array.

        Returns:
            Algebraic density at each price (negative for asks, positive for bids).
        """
        return -self.L * x

    # ================== TIME EVOLUTION ==================

    def update_prices(self) -> None:
        """Update best ask, best bid based on current density."""
        bid_indices = np.where(self.density * self.dx > self.resolution_volume)[0]
        ask_indices = np.where(self.density * self.dx < -self.resolution_volume)[0]

        self.best_ask_index = ask_indices[0] if ask_indices.size > 0 else self.n_grid - 1
        self.best_bid_index = bid_indices[-1] if bid_indices.size > 0 else 0

        self.best_ask = float(self.X[self.best_ask_index - 1])
        self.best_bid = float(self.X[self.best_bid_index + 1])

    def execute_metaorder(self, volume: float) -> None:
        """
        Execute a metaorder by consuming density at the best price.

        If volume > 0: buy order, consumes ask liquidity.
        If volume < 0: sell order, consumes bid liquidity.

        Args:
            volume: Algebraic volume to execute.

        Raises:
            ValueError: If market lacks liquidity.
        """
        if volume == 0:
            return

        dq = volume
        if dq > 0:
            while dq > 0:
                # liquidity > 0 is the absolute available volume at best ask
                liquidity = -self.density[self.best_ask_index] * self.dx
                if dq < liquidity:
                    self.density[self.best_ask_index] += dq / self.dx
                    dq = 0
                    break

                self.density[self.best_ask_index] = 0
                self.best_ask_index += 1
                dq -= liquidity
                if self.best_ask_index > self.n_grid - 1:
                    raise ValueError("Market lacks ask liquidity")
        else:
            while dq < 0:
                liquidity = self.density[self.best_bid_index] * self.dx
                if -dq > liquidity:
                    dq += liquidity
                    self.density[self.best_bid_index] = 0
                    self.best_bid_index -= 1
                    if self.best_bid_index == 0:
                        raise ValueError("Market lacks bid liquidity")
                else:
                    self.density[self.best_bid_index] += dq / self.dx
                    dq = 0

        self.best_ask_volume = self.density[self.best_ask_index + 1]
        self.best_bid_volume = self.density[self.best_bid_index - 1]

    def timestep(self, tstep: float, volume: float) -> None:
        """
        Advance the book by one timestep.

        Args:
            tstep: Time step size.
            volume: Metaorder volume to execute.
        """
        self.execute_metaorder(volume)
        self.update_prices()
        if self.D != 0:
            self.density = theta_scheme_iteration(
                self.density, self.dx, tstep, self.D, self.L
            )
        self.update_prices()

    # ================== ANIMATION ==================

    def set_animation(self, fig: Figure, lims: dict[str, Any]) -> None:
        """
        Set up matplotlib animation components.

        Args:
            fig: Matplotlib figure to add subplot to.
            lims: Dict with optional 'xlim' key for x-axis limits.
        """
        xlims = lims.get("xlim", (self.xmin, self.xmax))
        y_max = self.L * xlims[1]

        self.density_ax = fig.add_subplot(2, 1, 1)
        self.density_ax.set_xlim(xlims)
        (self.density_line,) = self.density_ax.plot(
            [], [], label="Density", color="gray"
        )
        (self.best_ask_axis,) = self.density_ax.plot(
            [], [], color="blue", ls="dashed", lw=1, label="best ask"
        )
        (self.best_bid_axis,) = self.density_ax.plot(
            [], [], color="red", ls="dashed", lw=1, label="best bid"
        )
        self.density_ax.plot(
            [self.xmin, self.xmax], [0, 0], color="black", lw=0.5, ls="dashed"
        )
        self.density_ax.plot(
            [0, 0], [-y_max, y_max], color="black", lw=0.5, ls="dashed"
        )
        self.density_ax.set_title("Algebraic order density")
        self.density_ax.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
        self.density_ax.set_ylim(-y_max, y_max)

    def init_animation(self) -> list[Any]:
        """Initialize animation frame."""
        if self.density_line is None:
            return []
        self.density_line.set_data([], [])
        return [self.density_line, self.best_ask_axis, self.best_bid_axis]

    def update_animation(self, tstep: float, volume: float) -> list[Any]:
        """
        Update animation for one frame.

        Args:
            tstep: Time step size.
            volume: Metaorder volume for this step.

        Returns:
            List of artists that were modified.
        """
        if (
            self.density_line is None
            or self.best_ask_axis is None
            or self.best_bid_axis is None
        ):
            return []

        y_max = 1.5 * self.xmax * self.L

        self.timestep(tstep, volume)
        self.density_line.set_data(self.X, self.density)
        self.best_ask_axis.set_data([self.best_ask, self.best_ask], [-y_max, y_max])
        self.best_bid_axis.set_data([self.best_bid, self.best_bid], [-y_max, y_max])
        return [self.best_ask_axis, self.best_bid_axis, self.density_line]
