
"""
Multi-actor order book in the LLOB framework.
"""

from functools import reduce
from typing import Any, Optional, Self

import numpy as np
import numpy.typing as npt
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .discrete_book import DiscreteBook
from .linear_discrete_book import LinearDiscreteBook


class MultiDiscreteBook:
    """
    Multi-actor order book combining multiple discrete books.

    Models a market with multiple types of liquidity providers, each with
    their own parameters (L, nu, lambd). Aggregate liquidity is the sum
    of all actor books.

    Attributes:
        books: List of individual actor DiscreteBook instances.
        N_actors: Number of actors in the market.
        best_ask: Current aggregate best ask price.
        best_bid: Current aggregate best bid price.
        actor_trades: Fraction of volume executed by each actor.
    """

    def __init__(
        self,
        books: list[DiscreteBook],
        X: npt.NDArray[np.float64],
        dx: float,
        D: float,
    ) -> None:
        """
        Initialize a MultiDiscreteBook with pre-constructed actor books.

        Use `from_params` classmethod for convenient construction.

        Args:
            books: List of DiscreteBook instances, one per actor.
            X: Price grid array.
            dx: Price grid spacing.
            D: Diffusion constant.
        """
        self.books = books
        self.X = X
        self.dx = dx
        self.D = D

        # Derived values
        self.N_actors = len(books)
        self.Nx = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])
        self.dt = dx**2 / (2 * D) if D > 0 else float("inf")

        # Initialize price tracking
        self.best_ask: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask_index: int = 0
        self.best_bid_index: int = 0
        self.bid_volume: int = 0
        self.ask_volume: int = 0
        self.update_price()

        # Trade tracking
        self.actor_trades = np.full(self.N_actors, 1 / self.N_actors)

        # Animation scaling
        self.y_max = float(np.sum([book.y_max for book in self.books]))

        # Animation state
        self.volume_ax: Any = None
        self.ask_bars: list[BarContainer] = []
        self.bid_bars: list[BarContainer] = []
        self.best_ask_axis: Optional[Line2D] = None
        self.best_bid_axis: Optional[Line2D] = None

    @classmethod
    def from_params(
        cls,
        D: float,
        xmin: float,
        xmax: float,
        Nx: int,
        L_list: list[float],
        nu_list: list[float],
        lambd_list: list[float],
    ) -> Self:
        """
        Create a MultiDiscreteBook from raw parameters.

        Args:
            D: Diffusion constant (shared by all actors).
            xmin: Price interval lower bound.
            xmax: Price interval upper bound.
            Nx: Number of price grid points.
            L_list: List of latent liquidity values, one per actor.
            nu_list: List of cancellation rates, one per actor.
            lambd_list: List of deposition intensities, one per actor.

        Returns:
            Configured MultiDiscreteBook instance.
        """
        assert (
            len(L_list) == len(nu_list) == len(lambd_list)
        ), "L_list, nu_list, and lambd_list must have the same length"

        # Compute grid
        X, dx = np.linspace(xmin, xmax, num=Nx, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        # Create individual books
        books: list[DiscreteBook] = []
        for L, nu, lambd in zip(L_list, nu_list, lambd_list, strict=True):
            linear = nu == 0
            if linear:
                book = LinearDiscreteBook.from_params(
                    D=D,
                    xmin=xmin,
                    xmax=xmax,
                    Nx=Nx,
                    L=L,
                )
            else:
                book = DiscreteBook.from_params(
                    D=D,
                    xmin=xmin,
                    xmax=xmax,
                    Nx=Nx,
                    L=L,
                    nu=nu,
                    lambd=lambd,
                )
            books.append(book)

        return cls(books=books, X=X, dx=dx, D=D)

    def get_ask_volumes(self, index: Optional[int] = None) -> npt.NDArray[np.int64]:
        """
        Get aggregate ask volumes across all actors.

        Args:
            index: If provided, return volume at this specific index only.

        Returns:
            Aggregate ask volumes array, or single value if index provided.
        """
        volumes = reduce(np.add, [book.get_ask_volumes() for book in self.books])
        if index is not None:
            return volumes[index]
        return volumes

    def get_bid_volumes(self, index: Optional[int] = None) -> npt.NDArray[np.int64]:
        """
        Get aggregate bid volumes across all actors.

        Args:
            index: If provided, return volume at this specific index only.

        Returns:
            Aggregate bid volumes array, or single value if index provided.
        """
        volumes = reduce(np.add, [book.get_bid_volumes() for book in self.books])
        if index is not None:
            return volumes[index]
        return volumes

    def get_ask_proportions(self, index: int) -> npt.NDArray[np.float64]:
        """Get proportion of ask volume from each actor at given index."""
        return self.books[index].get_ask_volumes() / self.get_ask_volumes()

    def get_bid_proportions(self, index: int) -> npt.NDArray[np.float64]:
        """Get proportion of bid volume from each actor at given index."""
        return self.books[index].get_bid_volumes() / self.get_bid_volumes()

    # ================== TIME EVOLUTION ==================

    def timestep(self, tstep: float, volume: float) -> None:
        """
        Advance the book by one timestep.

        Args:
            tstep: Time step size.
            volume: Metaorder volume to execute.
        """
        self.update_price()
        self.execute_metaorder(volume)
        self.stochastic_timestep()
        self.update_price()
        self.order_reaction()
        self.update_price()

    def stochastic_timestep(self) -> None:
        """Execute stochastic dynamics for all actor books."""
        for book in self.books:
            book.stochastic_timestep()

    def update_price(self) -> None:
        """Update aggregate best prices from all actor books."""
        for book in self.books:
            book.update_price()

        self.best_ask_index = int(np.min([book.best_ask_index for book in self.books]))
        self.best_ask = float(self.X[self.best_ask_index])
        self.best_bid_index = int(np.max([book.best_bid_index for book in self.books]))
        self.best_bid = float(self.X[self.best_bid_index])
        self.bid_volume = int(self.get_bid_volumes()[self.best_bid_index])
        self.ask_volume = int(self.get_ask_volumes()[self.best_ask_index])

    def order_reaction(self) -> None:
        """
        Execute matched orders across all actor books.

        Orders are matched regardless of which actor's book they come from.
        """
        if self.best_ask_index > self.best_bid_index:
            return

        reaction_volumes = np.minimum(self.get_ask_volumes(), self.get_bid_volumes())

        for side in ["ask", "bid"]:
            side_volumes = getattr(self, f"get_{side}_volumes")()
            # Mask indicating where side's volumes are lower than other side's
            limiting_volume = np.array(side_volumes <= reaction_volumes)

            for book in self.books:
                actor_side_orders = getattr(book, f"{side}_orders")
                # Avoid division by zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    actor_proportion = actor_side_orders.volumes / side_volumes
                    actor_proportion = np.nan_to_num(actor_proportion, nan=0.0)

                # Execute all side's liquidity where side is lacking,
                # else execute proportional fraction
                executed_volumes = np.where(
                    limiting_volume,
                    reaction_volumes,
                    actor_proportion * reaction_volumes,
                )
                actor_side_orders.execute_orders(executed_volumes)

    def execute_metaorder(self, trade_volume: float) -> None:
        """
        Execute a metaorder on the aggregate book.

        Consumes volumes from actor books in order of price proximity.

        Args:
            trade_volume: Algebraic volume to execute.
        """
        if trade_volume == 0:
            return

        side = "bid" if trade_volume < 0 else "ask"
        sign = 1 if side == "bid" else -1

        executed_volume = 0.0
        self.actor_trades.fill(0)

        price_index = getattr(self, f"best_{side}_index")
        total_price_volume = getattr(self, f"{side}_volume")

        # Consume best price orders, price step after price step
        while executed_volume + total_price_volume < abs(trade_volume):
            for actor_index, book in enumerate(self.books):
                actor_price_index = getattr(book, f"best_{side}_index")
                if actor_price_index != price_index:
                    continue

                actor_volume = getattr(book, f"best_{side}_volume")
                book.execute_metaorder(-sign * actor_volume)
                self.actor_trades[actor_index] += actor_volume
                executed_volume += actor_volume

            price_index -= sign
            self.update_price()
            total_price_volume = getattr(self, f"{side}_volume")
            price_index = getattr(self, f"best_{side}_index")

        # Execute remaining volume proportionally
        remaining_volume = abs(trade_volume) - executed_volume
        for actor_index, book in enumerate(self.books):
            actor_price_index = getattr(book, f"best_{side}_index")
            if actor_price_index != price_index:
                continue

            actor_volume = getattr(book, f"best_{side}_volume")
            if total_price_volume > 0:
                actor_volume = actor_volume / total_price_volume * remaining_volume
            book.execute_metaorder(actor_volume)
            self.actor_trades[actor_index] += actor_volume

        if abs(trade_volume) > 0:
            self.actor_trades /= abs(trade_volume)

    def get_measures(self) -> dict[str, Any]:
        """Get current market measurements."""
        return {
            "bid": self.best_bid,
            "ask": self.best_ask,
            "actor_trades": np.copy(self.actor_trades),
        }

    def get_measure(self, quantity: str) -> Any:
        """Get a specific measurement from the book."""
        if quantity in ["bid_volumes", "ask_volumes"]:
            return getattr(self, f"get_{quantity}")()
        return getattr(self, quantity)

    # ================== ANIMATION ==================

    def set_animation(self, fig: Figure, lims: Optional[dict[str, Any]] = None) -> None:
        """Set up matplotlib animation components."""
        if lims is None:
            lims = {}

        self.volume_ax = fig.add_subplot(2, 1, 1)
        self.volume_ax.set_ylim((0, self.y_max))
        self.volume_ax.set_title("Order volumes")
        width = max((self.xmax - self.xmin) / self.Nx, 0.02)

        # Lines
        self.volume_ax.plot(
            [0, 0], [-self.y_max, self.y_max], color="black", lw=0.5, ls="dashed"
        )
        (self.best_ask_axis,) = self.volume_ax.plot(
            [], [], color="blue", ls="dashed", lw=1, label="best ask"
        )
        (self.best_bid_axis,) = self.volume_ax.plot(
            [], [], color="red", ls="dashed", lw=1, label="best bid"
        )

        # Bars for each actor
        self.ask_bars = []
        self.bid_bars = []
        for index, book in enumerate(self.books):
            brightness = 1 - (index / self.N_actors)

            actor_ask_bars = self.volume_ax.bar(
                self.X,
                book.get_ask_volumes(),
                align="edge",
                label=f"Ask {index}",
                width=width,
                color=(0, 0, brightness),
                animated=True,
            )
            self.ask_bars.append(actor_ask_bars)

            actor_bid_bars = self.volume_ax.bar(
                self.X,
                book.get_bid_volumes(),
                align="edge",
                label=f"Bid {index}",
                width=-width,
                color=(brightness, 0, 0),
                animated=True,
            )
            self.bid_bars.append(actor_bid_bars)

    def init_animation(self) -> list[Any]:
        """Initialize animation frame."""
        result: list[Any] = []

        for actor_index in range(min(1, self.N_actors)):
            for x_index in range(self.Nx):
                bar = self.ask_bars[actor_index][x_index]
                bar.set_y(0)
                bar.set_height(0)
                result.append(bar)

        for actor_index in range(min(1, self.N_actors)):
            for x_index in range(self.Nx):
                bar = self.bid_bars[actor_index][x_index]
                bar.set_y(0)
                bar.set_height(0)
                result.append(bar)

        return result

    def update_animation(self, tstep: float, volume: float) -> list[Any]:
        """Update animation for one frame."""
        self.timestep(tstep, volume)

        padding = 0.01 * self.y_max
        result: list[Any] = []

        # Update ask bars
        heights = np.zeros(self.Nx)
        for actor_index in range(self.N_actors):
            actor_ask_bars = self.ask_bars[actor_index]
            for x_index in range(self.Nx):
                bar = actor_ask_bars[x_index]
                bar.set_y(heights[x_index] + padding)
                bar.set_height(self.books[actor_index].get_ask_volumes()[x_index])
                heights[x_index] += padding + bar.get_height()
                result.append(bar)

        # Update bid bars
        heights = np.zeros(self.Nx)
        for actor_index in range(self.N_actors):
            actor_bid_bars = self.bid_bars[actor_index]
            for x_index in range(self.Nx):
                bar = actor_bid_bars[x_index]
                bar.set_y(heights[x_index] + padding)
                bar.set_height(self.books[actor_index].get_bid_volumes()[x_index])
                heights[x_index] += padding + bar.get_height()
                result.append(bar)

        if self.best_ask_axis is not None:
            self.best_ask_axis.set_data([self.best_ask, self.best_ask], [0, self.y_max])
            result.append(self.best_ask_axis)
        if self.best_bid_axis is not None:
            self.best_bid_axis.set_data([self.best_bid, self.best_bid], [0, self.y_max])
            result.append(self.best_bid_axis)

        return result
