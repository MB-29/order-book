"""
Linear discrete order book in the LLOB framework.

Simplified DiscreteBook for the linear regime (nu=0, lambd=0): only diffusion
occurs, no deposition or cancellation.
"""

from typing import Self

import numpy as np

from .discrete_book import DiscreteBook
from .limit_orders import LimitOrders


class LinearDiscreteBook(DiscreteBook):
    """Discrete order book in the linear regime (nu=0, lambd=0)."""

    @classmethod
    def from_params(
        cls,
        D: float,
        xmin: float,
        xmax: float,
        n_grid: int,
        L: float,
    ) -> Self:
        X, dx = np.linspace(xmin, xmax, num=n_grid, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        sides = {}
        for side in ("bid", "ask"):
            sides[side] = LimitOrders.from_params(
                side=side,
                lambd=0.0,
                nu=0.0,
                D=D,
                xmin=xmin,
                xmax=xmax,
                n_grid=n_grid,
                L=L,
                initial_density="linear",
                boundary_conditions="linear",
            )

        return cls(bid_orders=sides["bid"], ask_orders=sides["ask"], X=X, dx=dx, D=D)

    def stochastic_timestep(self, dt: float) -> None:
        """Linear regime: only jumps (no deposition, no cancellation)."""
        for orders in (self.ask_orders, self.bid_orders):
            orders.jumps()
            orders.update_best_price()
