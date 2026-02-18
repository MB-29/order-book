"""
Linear discrete order book in the LLOB framework.

This is a simplified discrete book for the linear regime (nu=0, lambd=0)
where only diffusion occurs - no deposition or cancellation.
"""

from typing import Self

from .discrete_book import DiscreteBook
from .limit_orders import LimitOrders


class LinearDiscreteBook(DiscreteBook):
    """
    Discrete order book in the linear regime.

    In the linear regime (nu=0, lambd=0), order dynamics consist only of
    diffusion (jumps). There is no deposition or cancellation of orders.
    Uses linear initial density and linear boundary conditions.

    This is computationally cheaper than the full nonlinear model.
    """

    @classmethod
    def from_params(
        cls,
        D: float,
        xmin: float,
        xmax: float,
        Nx: int,
        L: float,
    ) -> Self:
        """
        Create a LinearDiscreteBook from raw parameters.

        Args:
            D: Diffusion constant.
            xmin: Price interval lower bound.
            xmax: Price interval upper bound.
            Nx: Number of price grid points.
            L: Order density slope (latent liquidity).

        Returns:
            Configured LinearDiscreteBook instance.
        """
        # Use parent's from_params with linear settings
        import numpy as np

        X, dx = np.linspace(xmin, xmax, num=Nx, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        bid_orders = LimitOrders.from_params(
            side="bid",
            lambd=0.0,
            nu=0.0,
            D=D,
            xmin=xmin,
            xmax=xmax,
            Nx=Nx,
            L=L,
            initial_density="linear",
            boundary_conditions="linear",
        )
        ask_orders = LimitOrders.from_params(
            side="ask",
            lambd=0.0,
            nu=0.0,
            D=D,
            xmin=xmin,
            xmax=xmax,
            Nx=Nx,
            L=L,
            initial_density="linear",
            boundary_conditions="linear",
        )

        return cls(
            bid_orders=bid_orders,
            ask_orders=ask_orders,
            X=X,
            dx=dx,
            D=D,
        )

    def stochastic_timestep(self) -> None:
        """
        Execute stochastic dynamics for linear regime.

        In the linear regime, only jumps (diffusion) occur.
        No arrivals or cancellations.
        """
        for orders in [self.ask_orders, self.bid_orders]:
            orders.jumps()
            orders.update_best_price()
