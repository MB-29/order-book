
"""
Limit orders implementation for one side of an order book.
"""

import warnings
from typing import Literal, Optional, Self

import numpy as np
import numpy.typing as npt
from numba import float64, int64, njit

# Module-level flags for runtime behavior
USE_NUMBA = False
DETERMINISTIC = False


class LimitOrders:
    """
    Order volumes for one side (bid or ask) with stochastic dynamics.

    Models the dynamics of limit orders on one side of the book, including:
    - Deposition: new orders arrive via Poisson process
    - Cancellation: orders expire with exponential lifetime
    - Jumps: orders diffuse via random walk (Smoluchowski dynamics)

    Attributes:
        volumes: Order volumes at each price level.
        X: Price grid array.
        dx: Price grid spacing.
        dt: Elementary timestep for Smoluchowski dynamics.
        best_price: Current best price on this side.
        best_price_index: Index of best price in the grid.
        best_price_volume: Volume at the best price.
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
    ) -> None:
        """
        Initialize a LimitOrders instance with pre-computed values.

        Use `from_params` classmethod for convenient construction from
        raw parameters.

        Args:
            side: Order side, either 'ask' or 'bid'.
            lambd: Deposition intensity parameter.
            nu: Cancellation rate parameter.
            D: Diffusion constant.
            L: Order density slope (latent liquidity).
            X: Price grid array.
            dx: Price grid spacing.
            volumes: Initial order volumes at each price level.
            boundary_flow: Flow at the boundary for diffusion.
        """
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

        # Derived constants
        self.n_grid = len(X)
        self.xmin = float(X[0])
        self.xmax = float(X[-1])
        self.sign = -1 if side == "ask" else 1
        self.boundary_index = -1 if side == "ask" else 0
        self.dt = dx**2 / (2 * D) if D > 0 else float("inf")

        # Initialize price tracking
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
    ) -> Self:
        """
        Create a LimitOrders instance from raw parameters.

        Args:
            side: Order side, either 'ask' or 'bid'.
            lambd: Deposition intensity parameter.
            nu: Cancellation rate parameter.
            D: Diffusion constant.
            xmin: Price interval lower bound.
            xmax: Price interval upper bound.
            n_grid: Number of price grid points.
            L: Order density slope. If None, computed from lambd/(sqrt(nu*D)).
            initial_density: Initial density profile type.
            boundary_conditions: Boundary condition type.

        Returns:
            Configured LimitOrders instance.

        Raises:
            ValueError: If L is None and nu*D <= 0.
        """
        # Compute grid
        X, dx = np.linspace(xmin, xmax, num=n_grid, retstep=True)
        X = np.asarray(X)
        dx = float(dx)

        # Compute L if not provided
        if L is None:
            if nu * D <= 0:
                raise ValueError("Cannot compute L: nu * D must be positive")
            L = lambd / np.sqrt(nu * D)

        # Compute dt for stability check
        dt = dx**2 / (2 * D) if D > 0 else float("inf")
        if dt * nu >= 1:
            warnings.warn(
                "Elementary timestep is too large to guarantee "
                "multiplicative cancellation rate.",
                stacklevel=2,
            )

        # Compute initial volumes
        sign = -1 if side == "ask" else 1
        volumes = _compute_initial_volumes(
            X, dx, L, nu, D, lambd, sign, initial_density
        )

        # Compute boundary flow
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
        )

    def stationary_density(self, x: float) -> float:
        """
        Compute the stationary order density at price x.

        For linear book (nu=0): density = L * |x| on the correct side.
        For nonlinear book: density = (lambd/nu) * (1 - exp(-|x|/x_crit))

        Args:
            x: Price level.

        Returns:
            Order density at price x.
        """
        if self.sign * x > 0:
            return 0.0

        if self.nu == 0:
            return self.L * x

        x_crit = np.sqrt(self.D / self.nu)
        return (self.lambd / self.nu) * (1 - np.exp(-abs(x) / x_crit))

    # ================== TIME EVOLUTION ==================

    def deposition(self, spread: int) -> None:
        """
        Process order deposition stochastic step.

        Orders arrive via Poisson process with intensity lambd * dt * dx.

        Note:
            update_best_price() should be called before this method
            to ensure the deposition price range is correct.

        Args:
            spread: Current spread in grid units (best_ask_index - best_bid_index).
        """
        lam = self.lambd * self.dt * self.dx

        # Number of arrival points for a given side
        if self.side == "ask":
            size = self.n_grid - self.best_price_index % self.n_grid
        else:
            size = self.best_price_index + 1

        if spread > 0:
            size += spread // 2
        padding_size = size - self.n_grid if self.side == "ask" else self.n_grid - size

        if USE_NUMBA:
            self.volumes = add_arrivals(self.volumes, lam, size, padding_size)
            return

        arrivals = np.random.poisson(lam=lam, size=size)
        padding = (self.n_grid - size, 0) if self.side == "ask" else (0, self.n_grid - size)
        arrivals = np.pad(arrivals, padding, mode="constant", constant_values=0)
        self.volumes += arrivals

    def cancellation(self) -> None:
        """
        Process order cancellation stochastic step.

        Orders are cancelled according to an exponential lifetime distribution
        with scale 1/nu.
        """
        scale = 1 / self.nu

        if DETERMINISTIC:
            self.volumes -= int(self.nu * self.volumes)
            return

        if USE_NUMBA:
            self.volumes = substract_cancellations(self.volumes, scale, self.dt)
            return

        get_cancellation_vec = np.vectorize(
            lambda volume: self._get_cancellation(volume, scale)
        )
        cancellations = get_cancellation_vec(self.volumes)
        self.volumes = self.volumes - cancellations

    def _get_cancellation(self, volume: int, scale: float) -> int:
        """
        Compute cancellations for a given volume using exponential lifetime.

        Args:
            volume: Number of orders at this price level.
            scale: Scale parameter (1/nu) for exponential distribution.

        Returns:
            Number of orders that get cancelled.
        """
        if volume == 0:
            return 0
        life_times = np.random.exponential(scale=scale, size=volume)
        cancellations = np.where(life_times < self.dt, 1, 0)
        return int(np.sum(cancellations))

    def jumps(self) -> None:
        """
        Process order jumps (diffusion) stochastic step.

        Orders perform a random walk with equal probability of jumping
        left or right (Smoluchowski dynamics).
        """
        if USE_NUMBA:
            self.volumes = add_flow(
                self.volumes, self.dx, self.boundary_index, self.boundary_flow
            )
            return

        jumps = np.zeros((self.n_grid, 2), dtype=int)
        for index, order_volume in enumerate(self.volumes):
            jumps_left = np.random.binomial(order_volume, 0.5)
            jumps[index, :] = [jumps_left, order_volume - jumps_left]

        boundary_volume = (
            self.volumes[self.boundary_index] + self.boundary_flow * self.dx**2
        )
        boundary_jumps = np.random.binomial(int(boundary_volume), 0.5)

        boundary_jumps_left = boundary_jumps if self.side == "ask" else 0
        boundary_jumps_right = boundary_jumps if self.side == "bid" else 0
        jumps_left = np.append(jumps[:, 0], boundary_jumps_left)
        jumps_right = np.insert(jumps[:, 1], 0, boundary_jumps_right)

        flow = jumps_right - jumps_left
        self.volumes = self.volumes - np.diff(flow)

    # ================== PRICE TRACKING ==================

    def update_best_price(self) -> int:
        """
        Update the best price for this side of the book.

        For ask: best price is the leftmost nonzero volume.
        For bid: best price is the rightmost nonzero volume.

        Returns:
            Index of the best price in the grid.
        """
        end_index = 0 if self.side == "ask" else -1
        indices = np.nonzero(self.volumes)[0]
        if indices.size == 0:
            indices = np.array([self.boundary_index])
        self.best_price_index = int(indices[end_index])
        self.best_price = float(self.X[self.best_price_index])
        self.best_price_volume = int(self.volumes[self.best_price_index])
        return self.best_price_index

    def execute_best_orders(self, volume: float) -> None:
        """
        Execute orders at the best price, walking through price levels as needed.

        Args:
            volume: Order volume to execute (positive for ask, negative for bid).

        Raises:
            ValueError: If market lacks liquidity to fill the order.
        """
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

    def execute_orders(
        self, volumes: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Execute given order volumes up to available liquidity at each price.

        Args:
            volumes: Volumes to execute at each price level.

        Returns:
            Actually executed volumes (may be less than requested).
        """
        trade_volumes = np.minimum(volumes, self.volumes)
        self.volumes -= trade_volumes.astype(int)
        return trade_volumes

    def get_available_volume(self, price_index: int) -> int:
        """
        Compute total volume between a price index and the best price.

        Args:
            price_index: Target price index.

        Returns:
            Total volume between price_index and best_price (inclusive).
        """
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
    """Compute initial order volumes based on density profile."""

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
    """
    Numba-accelerated order jump flow computation.

    Args:
        volumes: Order volumes at each price level.
        dx: Price grid spacing.
        boundary_index: Boundary index (-1 for ask, 0 for bid).
        boundary_flow: Flow at the boundary.

    Returns:
        Updated volumes array.
    """
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
    """
    Numba-accelerated order arrival computation.

    Args:
        volumes: Order volumes at each price level.
        lam: Poisson arrival intensity.
        size: Size of the deposition price range.
        padding_size: Complementary size (sign indicates padding direction).

    Returns:
        Updated volumes array.
    """
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
    """
    Numba-accelerated order cancellation computation.

    Args:
        volumes: Order volumes at each price level.
        scale: Exponential lifetime scale (1/nu).
        dt: Time step size.

    Returns:
        Updated volumes array.
    """
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
