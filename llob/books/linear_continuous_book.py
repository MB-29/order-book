"""
Continuous order book using PDE discretization in the LLOB framework.
"""

from typing import Self

import numpy as np
import numpy.typing as npt

from .diffusion_schemes import theta_scheme_iteration


class LinearContinuousBook:
    """
    Continuous order book.

    Models the algebraic order density as a continuous field, integrated with
    an implicit theta-scheme. Being implicit, the scheme is unconditionally
    stable so a per-frame dt is enough — there is no inner Courant loop.
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
        self.D = D
        self.L = L
        self.X = X
        self.dx = dx
        self.density = density
        self.resolution_volume = resolution_volume

        self.n_grid = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])
        self.J = D * L

        self.best_ask: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask_index: int = 0
        self.best_bid_index: int = 0
        self.best_ask_volume: float = 0.0
        self.best_bid_volume: float = 0.0
        self.update_prices()

        # +1 / -1 ensure the corresponding volume isn't partially consumed
        self.best_ask_volume = self.density[self.best_ask_index + 1]
        self.best_bid_volume = self.density[self.best_bid_index - 1]

    @classmethod
    def from_params(
        cls,
        D: float,
        L: float,
        xmin: float,
        xmax: float,
        n_grid: int = 1000,
    ) -> Self:
        X, dx = np.linspace(xmin, xmax, num=n_grid, retstep=True)
        X = np.asarray(X)
        dx = float(dx)
        density = -L * X
        resolution_volume = L * dx**2
        return cls(D=D, L=L, X=X, dx=dx, density=density, resolution_volume=resolution_volume)

    # ================== TIME EVOLUTION ==================

    def evolve(self, dt_frame: float, dq: float, dt_step: float | None = None) -> None:
        """
        Advance the book by physical time ``dt_frame`` with metaorder volume ``dq``.

        The theta scheme is implicit and unconditionally stable, so ``dt_step``
        is ignored — the whole frame is integrated in a single linear solve.
        """
        self.execute_metaorder(dq)
        self.update_prices()
        if self.D != 0:
            self.density = theta_scheme_iteration(self.density, self.dx, dt_frame, self.D, self.L)
        self.update_prices()

    def update_prices(self) -> None:
        bid_indices = np.where(self.density * self.dx > self.resolution_volume)[0]
        ask_indices = np.where(self.density * self.dx < -self.resolution_volume)[0]

        self.best_ask_index = ask_indices[0] if ask_indices.size > 0 else self.n_grid - 1
        self.best_bid_index = bid_indices[-1] if bid_indices.size > 0 else 0

        self.best_ask = float(self.X[self.best_ask_index - 1])
        self.best_bid = float(self.X[self.best_bid_index + 1])

    def execute_metaorder(self, volume: float) -> None:
        if volume == 0:
            return

        dq = volume
        if dq > 0:
            while dq > 0:
                liquidity = -self.density[self.best_ask_index] * self.dx
                if dq < liquidity:
                    self.density[self.best_ask_index] += dq / self.dx
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

    def get_measure(self, quantity: str) -> object:
        return getattr(self, quantity)
