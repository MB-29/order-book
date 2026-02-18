
"""
Discrete order book in the LLOB framework.
"""

from typing import Any, Literal, Optional, Self

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .limit_orders import LimitOrders


class DiscreteBook:
    """
    Agent-based discrete order book in the LLOB framework.

    Models an order book with discrete order volumes at each price level.
    Supports stochastic dynamics including deposition, cancellation, and
    diffusion of orders.

    Attributes:
        bid_orders: LimitOrders instance for bid side.
        ask_orders: LimitOrders instance for ask side.
        X: Price grid array.
        dx: Price grid spacing.
        dt: Elementary timestep for dynamics.
        best_ask: Current best ask price.
        best_bid: Current best bid price.
        best_ask_index: Index of best ask in grid.
        best_bid_index: Index of best bid in grid.
    """

    def __init__(
        self,
        bid_orders: LimitOrders,
        ask_orders: LimitOrders,
        X: npt.NDArray[np.float64],
        dx: float,
        D: float,
    ) -> None:
        """
        Initialize a DiscreteBook with pre-constructed order sides.

        Use `from_params` classmethod for convenient construction.

        Args:
            bid_orders: LimitOrders instance for bid side.
            ask_orders: LimitOrders instance for ask side.
            X: Price grid array.
            dx: Price grid spacing.
            D: Diffusion constant.
        """
        self.bid_orders = bid_orders
        self.ask_orders = ask_orders
        self.X = X
        self.dx = dx
        self.D = D

        # Derived values
        self.Nx = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])
        self.dt = dx**2 / (2 * D) if D > 0 else float("inf")

        # Initialize price tracking
        self.best_ask: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask_index: int = 0
        self.best_bid_index: int = 0
        self.best_ask_volume: int = 0
        self.best_bid_volume: int = 0
        self.update_price()

        # Animation scaling
        self.y_max = (
            max(
                self.bid_orders.stationary_density(self.xmin),
                self.ask_orders.stationary_density(self.xmax),
            )
            * self.dx
        )

        # Animation state (set during set_animation)
        self.volume_ax: Optional[Axes] = None
        self.ask_bars: Optional[BarContainer] = None
        self.bid_bars: Optional[BarContainer] = None
        self.best_ask_axis: Optional[Line2D] = None
        self.best_bid_axis: Optional[Line2D] = None

    @classmethod
    def from_params(
        cls,
        D: float,
        xmin: float,
        xmax: float,
        Nx: int,
        lambd: float = 0.0,
        nu: float = 0.0,
        L: Optional[float] = None,
        initial_density: Literal["stationary", "linear", "empty"] = "stationary",
        boundary_conditions: Literal["flat", "linear"] = "flat",
    ) -> Self:
        """
        Create a DiscreteBook from raw parameters.

        Args:
            D: Diffusion constant.
            xmin: Price interval lower bound.
            xmax: Price interval upper bound.
            Nx: Number of price grid points.
            lambd: Deposition intensity parameter.
            nu: Cancellation rate parameter.
            L: Order density slope. If None, computed from lambd/(sqrt(nu*D)).
            initial_density: Initial density profile type.
            boundary_conditions: Boundary condition type.

        Returns:
            Configured DiscreteBook instance.
        """
        # Compute grid
        X, dx = np.linspace(xmin, xmax, num=Nx, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        # Create order sides
        bid_orders = LimitOrders.from_params(
            side="bid",
            lambd=lambd,
            nu=nu,
            D=D,
            xmin=xmin,
            xmax=xmax,
            Nx=Nx,
            L=L,
            initial_density=initial_density,
            boundary_conditions=boundary_conditions,
        )
        ask_orders = LimitOrders.from_params(
            side="ask",
            lambd=lambd,
            nu=nu,
            D=D,
            xmin=xmin,
            xmax=xmax,
            Nx=Nx,
            L=L,
            initial_density=initial_density,
            boundary_conditions=boundary_conditions,
        )

        return cls(
            bid_orders=bid_orders,
            ask_orders=ask_orders,
            X=X,
            dx=dx,
            D=D,
        )

    def get_ask_volumes(self) -> npt.NDArray[np.int64]:
        """Get order volumes for the ask side."""
        return self.ask_orders.volumes

    def get_bid_volumes(self) -> npt.NDArray[np.int64]:
        """Get order volumes for the bid side."""
        return self.bid_orders.volumes

    # ================== TIME EVOLUTION ==================

    def timestep(self, tstep: float, volume: float) -> None:
        """
        Advance the book by one timestep.

        Args:
            tstep: Time step size.
            volume: Metaorder volume to execute (positive=buy, negative=sell).
        """
        self.update_price()
        self.execute_metaorder(volume)
        self.stochastic_timestep()
        self.update_price()
        self.order_reaction()
        self.update_price()

    def stochastic_timestep(self) -> None:
        """Execute stochastic dynamics: cancellation, deposition, and jumps."""
        spread = self.best_ask_index - self.best_bid_index
        for orders in [self.ask_orders, self.bid_orders]:
            orders.cancellation()
            orders.deposition(spread)
            orders.jumps()

    def update_price(self) -> None:
        """Update best prices for both sides of the book."""
        for orders in [self.ask_orders, self.bid_orders]:
            orders.update_best_price()

        self.best_ask_index = self.ask_orders.best_price_index
        self.best_bid_index = self.bid_orders.best_price_index
        self.best_ask = float(self.X[self.best_ask_index - 1])
        self.best_bid = float(self.X[self.best_bid_index + 1])
        self.best_ask_volume = int(
            self.get_ask_volumes()[self.ask_orders.best_price_index]
        )
        self.best_bid_volume = int(
            self.get_bid_volumes()[self.bid_orders.best_price_index]
        )

    def order_reaction(self) -> None:
        """Execute matched orders where bid and ask cross."""
        if self.best_ask_index > self.best_bid_index:
            return

        reaction_volumes = np.minimum(self.get_ask_volumes(), self.get_bid_volumes())
        for side_orders in [self.ask_orders, self.bid_orders]:
            side_orders.execute_orders(reaction_volumes)

    def execute_metaorder(self, volume: float) -> None:
        """
        Execute a metaorder of given volume.

        Args:
            volume: Trade volume (positive=buy asks, negative=sell to bids).
        """
        side = "ask" if volume > 0 else "bid"
        orders = getattr(self, f"{side}_orders")
        orders.execute_best_orders(volume)

    def get_measure(self, quantity: str) -> Any:
        """
        Get a measurement from the book.

        Args:
            quantity: Name of the quantity to measure.

        Returns:
            The measured value.
        """
        if quantity in ["bid_volumes", "ask_volumes"]:
            return getattr(self, f"get_{quantity}")()
        return getattr(self, quantity)

    # ================== ANIMATION ==================

    def set_animation(self, fig: Figure, lims: Optional[dict[str, Any]] = None) -> None:
        """
        Set up matplotlib animation components.

        Args:
            fig: Matplotlib figure to add subplot to.
            lims: Optional dict with 'xlim' key for x-axis limits.
        """
        if lims is None:
            lims = {}

        self.volume_ax = fig.add_subplot(2, 1, 1)

        width = max((self.xmax - self.xmin) / self.Nx, 0.02)
        self.ask_bars = self.volume_ax.bar(
            self.X,
            self.get_ask_volumes(),
            align="edge",
            label="Ask",
            color="blue",
            width=width,
            animated=True,
        )
        self.bid_bars = self.volume_ax.bar(
            self.X,
            self.get_bid_volumes(),
            align="edge",
            label="Bid",
            color="red",
            width=-width,
            animated=True,
        )

        self.volume_ax.plot(
            [0, 0], [-self.y_max, self.y_max], color="black", lw=0.5, ls="dashed"
        )
        (self.best_ask_axis,) = self.volume_ax.plot(
            [], [], color="blue", ls="dashed", lw=1, label="best ask"
        )
        (self.best_bid_axis,) = self.volume_ax.plot(
            [], [], color="red", ls="dashed", lw=1, label="best bid"
        )

        xlims = lims.get("xlim", (self.xmin, self.xmax))
        self.volume_ax.set_xlim(xlims)
        self.volume_ax.set_ylim((0, self.y_max))
        self.volume_ax.set_title("Order volumes")
        self.volume_ax.legend(loc="upper center")

    def init_animation(self) -> list[Any]:
        """Initialize animation frame."""
        if self.ask_bars is None or self.bid_bars is None:
            return []

        for b in self.ask_bars:
            b.set_height(0)
        for b in self.bid_bars:
            b.set_height(0)

        return list(self.ask_bars) + list(self.bid_bars)

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
            self.ask_bars is None
            or self.bid_bars is None
            or self.best_ask_axis is None
            or self.best_bid_axis is None
        ):
            return []

        self.timestep(tstep, volume)

        for index in range(self.Nx):
            self.ask_bars[index].set_height(self.get_ask_volumes()[index])
            self.bid_bars[index].set_height(self.get_bid_volumes()[index])

        self.best_ask_axis.set_data([self.best_ask, self.best_ask], [0, self.y_max])
        self.best_bid_axis.set_data([self.best_bid, self.best_bid], [0, self.y_max])

        return (
            list(self.ask_bars)
            + list(self.bid_bars)
            + [self.best_ask_axis, self.best_bid_axis]
        )
