"""
Discrete order book in the LLOB framework.
"""

import warnings
from typing import Any, Literal, Optional, Self

import numpy as np
import numpy.typing as npt

from .limit_orders import LimitOrders


class IntegerFloorWarning(UserWarning):
    """|dq|/n_steps < 1 — integer-volume floor will truncate sub-unit slices."""


class DiscreteBook:
    """
    Agent-based discrete order book in the LLOB framework.

    Two sides of LimitOrders with stochastic dynamics (deposition,
    cancellation, diffusion). The caller picks the inner diffusion timestep
    `dt_step` and passes it to evolve().
    """

    def __init__(
        self,
        bid_orders: LimitOrders,
        ask_orders: LimitOrders,
        X: npt.NDArray[np.float64],
        dx: float,
        D: float,
    ) -> None:
        self.bid_orders = bid_orders
        self.ask_orders = ask_orders
        self.X = X
        self.dx = dx
        self.D = D

        self.n_grid = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])

        self.best_ask: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask_index: int = 0
        self.best_bid_index: int = 0
        self.best_ask_volume: int = 0
        self.best_bid_volume: int = 0
        self.update_price()

    @classmethod
    def from_params(
        cls,
        D: float,
        xmin: float,
        xmax: float,
        n_grid: int,
        lambd: float = 0.0,
        nu: float = 0.0,
        L: Optional[float] = None,
        initial_density: Literal["stationary",
                                 "linear", "empty"] = "stationary",
        boundary_conditions: Literal["flat", "linear"] = "flat",
        alpha: float = 0.0,
    ) -> Self:
        X, dx = np.linspace(xmin, xmax, num=n_grid, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        sides = {}
        for side in ("bid", "ask"):
            sides[side] = LimitOrders.from_params(
                side=side,
                lambd=lambd,
                nu=nu,
                D=D,
                xmin=xmin,
                xmax=xmax,
                n_grid=n_grid,
                L=L,
                initial_density=initial_density,
                boundary_conditions=boundary_conditions,
                alpha=alpha,
            )

        return cls(bid_orders=sides["bid"], ask_orders=sides["ask"], X=X, dx=dx, D=D)

    def get_ask_volumes(self) -> npt.NDArray[np.int64]:
        return self.ask_orders.volumes

    def get_bid_volumes(self) -> npt.NDArray[np.int64]:
        return self.bid_orders.volumes

    # ================== TIME EVOLUTION ==================

    def evolve(self, dt_frame: float, dq: float, dt_step: float) -> None:
        """
        Advance the book by physical time ``dt_frame`` with metaorder volume ``dq``.

        The metaorder is split progressively over ``n_steps = round(dt_frame /
        dt_step)`` reaction-diffusion sub-steps, with ``dq/n_steps`` injected
        per sub-step. Validity requires ``|dq|/n_steps >> 1`` (integer-volume
        floor); below 1 the slice is silently truncated to 0.
        """
        n_steps = max(1, int(round(dt_frame / dt_step)))
        if dq != 0 and abs(dq) / n_steps < 1:
            warnings.warn(
                "|dq|/n_steps < 1: integer-volume floor truncates sub-unit "
                "metaorder slices; rescale L so per-step volumes are >> 1",
                IntegerFloorWarning, stacklevel=2,
            )
        for _ in range(n_steps):
            self.execute_metaorder(dq/n_steps)
            self.reaction_diffusion_step(dt_step)

    def reaction_diffusion_step(self, dt: float) -> None:
        """One reaction-diffusion step: stochastic dynamics + order matching."""
        self.stochastic_timestep(dt)
        self.update_price()
        self.order_reaction()
        self.update_price()

    def stochastic_timestep(self, dt: float) -> None:
        """Cancellation, deposition and jumps for both sides."""
        spread = self.best_ask_index - self.best_bid_index
        for orders in (self.ask_orders, self.bid_orders):
            orders.cancellation(dt)
            orders.deposition(dt, spread)
            orders.jumps()

    def update_price(self) -> None:
        for orders in (self.ask_orders, self.bid_orders):
            orders.update_best_price()
        self.best_ask_index = self.ask_orders.best_price_index
        self.best_bid_index = self.bid_orders.best_price_index
        self.best_ask = float(self.X[self.best_ask_index])
        self.best_bid = float(self.X[self.best_bid_index])
        self.best_ask_volume = int(self.get_ask_volumes()[
                                   self.ask_orders.best_price_index])
        self.best_bid_volume = int(self.get_bid_volumes()[
                                   self.bid_orders.best_price_index])

    def order_reaction(self) -> None:
        """Annihilate overlapping bid/ask volumes at every price level."""
        reaction_volumes = np.minimum(
            self.ask_orders.volumes, self.bid_orders.volumes)
        self.ask_orders.volumes -= reaction_volumes
        self.bid_orders.volumes -= reaction_volumes

    def execute_metaorder(self, volume: float) -> None:
        side = "ask" if volume > 0 else "bid"
        orders = getattr(self, f"{side}_orders")
        orders.execute_best_orders(volume)

    def get_measure(self, quantity: str) -> Any:
        if quantity in ("bid_volumes", "ask_volumes"):
            return getattr(self, f"get_{quantity}")()
        return getattr(self, quantity)
