"""
Limit orders implementation for one side of an order book.
"""

from typing import Literal, Optional, Self

import numpy as np
import numpy.typing as npt
from numba import float64, int64, njit

USE_NUMBA = False
DETERMINISTIC = False


class LimitOrders:
    """
    Order volumes for one side (bid or ask) with stochastic dynamics.

    Models limit-order dynamics on one side of the book: deposition (Poisson
    arrivals), cancellation (exponential lifetime), and jumps (Smoluchowski
    random walk). The timestep is supplied per call by the caller — this class
    holds no notion of dt.
    """

    def __init__(
        self,
        side: Literal["ask", "bid"],
        lambd: float,
        nu: float,
        D: float,
        L: float,
        X: npt.NDArray[np.float64],
        dx: float,
        volumes: npt.NDArray[np.int64],
        boundary_flow: float,
        alpha: float = 0.0,
    ) -> None:
        assert side in ("ask", "bid"), f"side must be 'ask' or 'bid', got {side}"

        self.side = side
        self.lambd = lambd
        self.nu = nu
        self.D = D
        self.L = L
        self.X = X
        self.dx = dx
        self.volumes = volumes
        self.boundary_flow = boundary_flow
        self.alpha = alpha

        self.n_grid = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])
        self.sign = -1 if side == "ask" else 1
        self.boundary_index = -1 if side == "ask" else 0

        self.best_price_index: int = 0
        self.best_price: float = 0.0
        self.best_price_volume: int = 0
        self.update_best_price()

    @classmethod
    def from_params(
        cls,
        side: Literal["ask", "bid"],
        lambd: float,
        nu: float,
        D: float,
        xmin: float,
        xmax: float,
        n_grid: int,
        L: Optional[float] = None,
        initial_density: Literal["stationary", "linear", "empty"] = "stationary",
        boundary_conditions: Literal["flat", "linear"] = "flat",
        alpha: float = 0.0,
    ) -> Self:
        X, dx = np.linspace(xmin, xmax, num=n_grid, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        if L is None:
            if nu * D <= 0:
                raise ValueError("Cannot compute L: nu * D must be positive")
            L = lambd / np.sqrt(nu * D)

        sign = -1 if side == "ask" else 1
        volumes = _compute_initial_volumes(X, dx, L, nu, D, lambd, sign, initial_density)
        boundary_flow = L if boundary_conditions == "linear" else 0.0

        return cls(
            side=side,
            lambd=lambd,
            nu=nu,
            D=D,
            L=L,
            X=X,
            dx=dx,
            volumes=volumes,
            boundary_flow=boundary_flow,
            alpha=alpha,
        )

    def stationary_density(self, x: float) -> float:
        if self.sign * x > 0:
            return 0.0
        if self.nu == 0:
            return self.L * x
        x_crit = np.sqrt(self.D / self.nu)
        return (self.lambd / self.nu) * (1 - np.exp(-abs(x) / x_crit))

    # ================== TIME EVOLUTION ==================

    def deposition(self, dt: float, spread: int) -> None:
        """
        Poisson arrivals on the book side at base rate ``lambd``, extended
        halfway into the spread at a boosted rate ``lambd * (1 + alpha * spread)``.
        """
        lam = self.lambd * dt * self.dx

        if self.side == "ask":
            base_size = self.n_grid - self.best_price_index % self.n_grid
        else:
            base_size = self.best_price_index + 1

        gap_size = spread // 2 if spread > 0 else 0
        boosted_lam = self.lambd * (1.0 + self.alpha * spread) * dt * self.dx

        size = base_size + gap_size
        padding_size = size - self.n_grid if self.side == "ask" else self.n_grid - size

        if USE_NUMBA:
            self.volumes = add_arrivals(self.volumes, lam, size, padding_size)
            return

        arrivals_base = np.random.poisson(lam=lam, size=base_size)
        arrivals_gap = (
            np.random.poisson(lam=boosted_lam, size=gap_size)
            if gap_size > 0
            else np.array([], dtype=int)
        )

        if self.side == "ask":
            arrivals = np.concatenate([arrivals_gap, arrivals_base])
        else:
            arrivals = np.concatenate([arrivals_base, arrivals_gap])

        padding = (self.n_grid - size, 0) if self.side == "ask" else (0, self.n_grid - size)
        arrivals = np.pad(arrivals, padding, mode="constant", constant_values=0)
        self.volumes += arrivals

    def cancellation(self, dt: float) -> None:
        """Cancel orders with exponential lifetime (scale 1/nu)."""
        scale = 1 / self.nu

        if DETERMINISTIC:
            self.volumes -= (self.nu * dt * self.volumes).astype(self.volumes.dtype)
            return

        if USE_NUMBA:
            self.volumes = substract_cancellations(self.volumes, scale, dt)
            return

        get_cancellation_vec = np.vectorize(
            lambda volume: self._get_cancellation(volume, dt, scale)
        )
        cancellations = get_cancellation_vec(self.volumes)
        self.volumes = self.volumes - cancellations

    def _get_cancellation(self, volume: int, dt: float, scale: float) -> int:
        if volume == 0:
            return 0
        life_times = np.random.exponential(scale=scale, size=volume)
        return int(np.sum(life_times < dt))

    def jumps(self) -> None:
        """One Smoluchowski random-walk step (Bernoulli left/right per order)."""
        if USE_NUMBA:
            self.volumes = add_flow(self.volumes, self.dx, self.boundary_index, self.boundary_flow)
            return

        jumps = np.zeros((self.n_grid, 2), dtype=int)
        for index, order_volume in enumerate(self.volumes):
            jumps_left = np.random.binomial(order_volume, 0.5)
            jumps[index, :] = [jumps_left, order_volume - jumps_left]

        boundary_volume = self.volumes[self.boundary_index] + self.boundary_flow * self.dx**2
        boundary_jumps = np.random.binomial(int(boundary_volume), 0.5)

        boundary_jumps_left = boundary_jumps if self.side == "ask" else 0
        boundary_jumps_right = boundary_jumps if self.side == "bid" else 0
        jumps_left = np.append(jumps[:, 0], boundary_jumps_left)
        jumps_right = np.insert(jumps[:, 1], 0, boundary_jumps_right)

        flow = jumps_right - jumps_left
        self.volumes = self.volumes - np.diff(flow)

    # ================== PRICE TRACKING ==================

    def update_best_price(self) -> int:
        end_index = 0 if self.side == "ask" else -1
        indices = np.nonzero(self.volumes)[0]
        if indices.size == 0:
            indices = np.array([self.boundary_index])
        self.best_price_index = int(indices[end_index])
        self.best_price = float(self.X[self.best_price_index])
        self.best_price_volume = int(self.volumes[self.best_price_index])
        return self.best_price_index

    def execute_best_orders(self, volume: float) -> None:
        if volume == 0:
            return
        if self.best_price_index in {0, self.n_grid}:
            raise ValueError(f"Market lacks {self.side} liquidity")

        index_increment = -self.sign
        trade_volume = abs(volume)

        while trade_volume > 0:
            liquidity = self.volumes[self.best_price_index]
            if trade_volume < liquidity:
                self.volumes[self.best_price_index] -= int(trade_volume)
                break

            self.volumes[self.best_price_index] = 0
            self.best_price_index += index_increment
            trade_volume -= liquidity

            if self.best_price_index > self.n_grid - 1 or self.best_price_index < 0:
                raise ValueError(f"Market lacks {self.side} liquidity")

    def execute_orders(self, volumes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        trade_volumes = np.minimum(volumes, self.volumes)
        self.volumes -= trade_volumes.astype(int)
        return trade_volumes

    def get_available_volume(self, price_index: int) -> int:
        price = self.X[price_index]
        if self.sign * (price - self.best_price) > 0:
            return 0
        lower = min(self.best_price_index, price_index)
        upper = max(self.best_price_index, price_index)
        return int(np.sum(self.volumes[lower : upper + 1]))


# ================== HELPER FUNCTIONS ==================


def _compute_initial_volumes(
    X: npt.NDArray[np.float64],
    dx: float,
    L: float,
    nu: float,
    D: float,
    lambd: float,
    sign: int,
    initial_density: str,
) -> npt.NDArray[np.int64]:
    def stationary_density(x: float) -> float:
        if sign * x > 0:
            return 0.0
        if nu == 0:
            return L * x
        x_crit = np.sqrt(D / nu)
        return (lambd / nu) * (1 - np.exp(-abs(x) / x_crit))

    def linear_density(x: float) -> float:
        return L * abs(x) if sign * x <= 0 else 0.0

    def empty_density(x: float) -> float:
        return 0.0

    density_funcs = {
        "stationary": stationary_density,
        "linear": linear_density,
        "empty": empty_density,
    }
    density_func = density_funcs[initial_density]
    volumes_func = np.vectorize(lambda x: int(dx * density_func(x)))
    return volumes_func(X).astype(np.int64)


# ================== NUMBA-OPTIMIZED FUNCTIONS ==================


@njit(int64[:](int64[:], float64, int64, float64))
def add_flow(
    volumes: npt.NDArray[np.int64],
    dx: float,
    boundary_index: int,
    boundary_flow: float,
) -> npt.NDArray[np.int64]:
    Nx = len(volumes)
    jumps = np.zeros((Nx, 2), dtype=int64)

    for index, order_volume in enumerate(volumes):
        jumps_left = np.random.binomial(order_volume, 0.5)
        jumps[index, :] = [jumps_left, order_volume - jumps_left]

    boundary_volume = int(volumes[boundary_index] + boundary_flow * (dx) ** 2)
    boundary_jumps = np.random.binomial(boundary_volume, 0.5, size=1)[0]

    boundary_jumps_left = boundary_jumps if boundary_index == -1 else 0
    boundary_jumps_right = boundary_jumps if boundary_index == 0 else 0
    jumps_left = np.append(jumps[:, 0], boundary_jumps_left)
    jumps_right = np.append(boundary_jumps_right, jumps[:, 1])
    flow = jumps_right - jumps_left

    volumes = volumes - np.diff(flow)
    return volumes


@njit(int64[:](int64[:], float64, int64, int64))
def add_arrivals(
    volumes: npt.NDArray[np.int64],
    lam: float,
    size: int,
    padding_size: int,
) -> npt.NDArray[np.int64]:
    arrivals = np.random.poisson(lam=lam, size=size)
    padding = np.zeros(abs(padding_size), dtype=int64)
    arrays = (padding, arrivals) if padding_size < 0 else (arrivals, padding)
    arrivals = np.concatenate(arrays)
    return np.add(volumes, arrivals)


@njit(int64[:](int64[:], float64, float64))
def substract_cancellations(
    volumes: npt.NDArray[np.int64],
    scale: float,
    dt: float,
) -> npt.NDArray[np.int64]:
    total_volume = np.sum(volumes)
    life_times = np.random.exponential(scale=scale, size=total_volume)
    cancellations = np.zeros(volumes.size, dtype=int64)
    volume_index = 0

    for index, order_volume in enumerate(volumes):
        cancellations[index] = np.sum(
            np.where(
                life_times[volume_index : volume_index + order_volume] < dt,
                1,
                0,
            )
        )
        volume_index += order_volume
    return volumes - cancellations
