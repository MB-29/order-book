"""
Multi-actor order book in the LLOB framework.
"""

from functools import reduce
from typing import Any, Optional, Self

import numpy as np
import numpy.typing as npt

from .discrete_book import DiscreteBook
from .linear_discrete_book import LinearDiscreteBook


class MultiDiscreteBook:
    """
    Multi-actor order book combining several DiscreteBook actors that share
    the same grid and diffusion constant. Aggregate liquidity is the sum
    over actor books.
    """

    def __init__(
        self,
        books: list[DiscreteBook],
        X: npt.NDArray[np.float64],
        dx: float,
        D: float,
    ) -> None:
        self.books = books
        self.X = X
        self.dx = dx
        self.D = D

        self.N_actors = len(books)
        self.n_grid = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])

        self.best_ask: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask_index: int = 0
        self.best_bid_index: int = 0
        self.bid_volume: int = 0
        self.ask_volume: int = 0
        self.update_price()

        self.actor_trades = np.full(self.N_actors, 1 / self.N_actors)

    @classmethod
    def from_params(
        cls,
        D: float,
        xmin: float,
        xmax: float,
        n_grid: int,
        L_list: list[float],
        nu_list: list[float],
        lambd_list: list[float],
    ) -> Self:
        assert len(L_list) == len(nu_list) == len(lambd_list), (
            "L_list, nu_list, and lambd_list must have the same length"
        )

        X, dx = np.linspace(xmin, xmax, num=n_grid, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        books: list[DiscreteBook] = []
        for L, nu, lambd in zip(L_list, nu_list, lambd_list, strict=True):
            if nu == 0:
                book = LinearDiscreteBook.from_params(
                    D=D, xmin=xmin, xmax=xmax, n_grid=n_grid, L=L,
                )
            else:
                book = DiscreteBook.from_params(
                    D=D, xmin=xmin, xmax=xmax, n_grid=n_grid,
                    L=L, nu=nu, lambd=lambd,
                )
            books.append(book)

        return cls(books=books, X=X, dx=dx, D=D)

    def get_ask_volumes(self, index: Optional[int] = None) -> npt.NDArray[np.int64]:
        volumes = reduce(np.add, [book.get_ask_volumes() for book in self.books])
        if index is not None:
            return volumes[index]
        return volumes

    def get_bid_volumes(self, index: Optional[int] = None) -> npt.NDArray[np.int64]:
        volumes = reduce(np.add, [book.get_bid_volumes() for book in self.books])
        if index is not None:
            return volumes[index]
        return volumes

    def get_ask_proportions(self, index: int) -> npt.NDArray[np.float64]:
        return self.books[index].get_ask_volumes() / self.get_ask_volumes()

    def get_bid_proportions(self, index: int) -> npt.NDArray[np.float64]:
        return self.books[index].get_bid_volumes() / self.get_bid_volumes()

    # ================== TIME EVOLUTION ==================

    def evolve(self, dt_frame: float, dq: float, dt_step: float) -> None:
        """Same contract as DiscreteBook.evolve, applied to all actor books."""
        self.execute_metaorder(dq)
        n_steps = max(1, int(round(dt_frame / dt_step)))
        for _ in range(n_steps):
            self.reaction_diffusion_step(dt_step)

    def reaction_diffusion_step(self, dt: float) -> None:
        self.stochastic_timestep(dt)
        self.update_price()
        self.order_reaction()
        self.update_price()

    def stochastic_timestep(self, dt: float) -> None:
        for book in self.books:
            book.stochastic_timestep(dt)

    def update_price(self) -> None:
        for book in self.books:
            book.update_price()
        self.best_ask_index = int(np.min([book.best_ask_index for book in self.books]))
        self.best_ask = float(self.X[self.best_ask_index])
        self.best_bid_index = int(np.max([book.best_bid_index for book in self.books]))
        self.best_bid = float(self.X[self.best_bid_index])
        self.bid_volume = int(self.get_bid_volumes()[self.best_bid_index])
        self.ask_volume = int(self.get_ask_volumes()[self.best_ask_index])

    def order_reaction(self) -> None:
        if self.best_ask_index > self.best_bid_index:
            return

        reaction_volumes = np.minimum(self.get_ask_volumes(), self.get_bid_volumes())

        for side in ("ask", "bid"):
            side_volumes = getattr(self, f"get_{side}_volumes")()
            limiting_volume = np.array(side_volumes <= reaction_volumes)

            for book in self.books:
                actor_side_orders = getattr(book, f"{side}_orders")
                with np.errstate(divide="ignore", invalid="ignore"):
                    actor_proportion = actor_side_orders.volumes / side_volumes
                    actor_proportion = np.nan_to_num(actor_proportion, nan=0.0)

                executed_volumes = np.where(
                    limiting_volume,
                    reaction_volumes,
                    actor_proportion * reaction_volumes,
                )
                actor_side_orders.execute_orders(executed_volumes)

    def execute_metaorder(self, trade_volume: float) -> None:
        if trade_volume == 0:
            return

        side = "bid" if trade_volume < 0 else "ask"
        sign = 1 if side == "bid" else -1

        executed_volume = 0.0
        self.actor_trades.fill(0)

        price_index = getattr(self, f"best_{side}_index")
        total_price_volume = getattr(self, f"{side}_volume")

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

    def get_measure(self, quantity: str) -> Any:
        if quantity in ("bid_volumes", "ask_volumes"):
            return getattr(self, f"get_{quantity}")()
        return getattr(self, quantity)
