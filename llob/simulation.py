"""
LLOB simulation runner.
"""

import warnings
from collections.abc import Callable
from typing import Any, Literal, Optional, Self

import numpy as np
import numpy.typing as npt

from .books import (
    DiscreteBook,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
)

BookType = DiscreteBook | LinearDiscreteBook | LinearContinuousBook | MultiDiscreteBook


def courant_dt(dx: float, D: float) -> float:
    """CFL-stable inner timestep for the explicit diffusion scheme: dx²/(2D)."""
    return dx * dx / (2 * D) if D > 0 else float("inf")


class Simulation:
    """
    LLOB simulation runner.

    Iterates the order book over ``n_frames`` output frames, each of physical
    duration ``dt = duration / n_frames``. The caller picks the inner
    diffusion timestep ``dt_step``; a warning is emitted if it violates CFL.

    Attributes:
        book: Order book instance (any of the *Book classes).
        prices, asks, bids, spreads: Output arrays of length ``n_frames``.
        metaorder: Per-frame metaorder intensity (array of length ``n_frames``).
        time: Frame times (array of length ``n_frames``).
    """

    def __init__(
        self,
        book: BookType,
        duration: float,
        n_frames: int,
        dt_step: float,
        metaorder: Optional[npt.ArrayLike] = None,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        price_formula: Literal["middle", "best_ask", "best_bid", "vwap"] = "middle",
        measured_quantities: Optional[list[str]] = None,
        measurement_indices: Optional[list[int]] = None,
    ) -> None:
        self.book = book
        self.duration = duration
        self.n_frames = n_frames
        self.dt_step = dt_step
        self.dt = duration / n_frames
        self.time = np.linspace(0, duration, n_frames)

        self.frame_start = frame_start
        self.frame_end = n_frames if frame_end is None else frame_end
        self.metaorder = self._expand_metaorder(metaorder)

        # Warn on CFL violation
        courant = courant_dt(book.dx, book.D)
        if dt_step > courant:
            warnings.warn(
                f"dt_step={dt_step:.3e} exceeds the CFL limit dx²/(2D)={courant:.3e}; "
                "the explicit diffusion scheme may be unstable.",
                stacklevel=2,
            )

        self.price_formula = price_formula
        self.measured_quantities = measured_quantities or []
        self.measurement_indices = measurement_indices or []

        self.asks = np.zeros(n_frames)
        self.bids = np.zeros(n_frames)
        self.prices = np.zeros(n_frames)
        self.spreads = np.zeros(n_frames)
        self.measurements: dict[str, list[Any]] = {q: [] for q in self.measured_quantities}

        price_funcs: dict[str, Callable[[float, float], float]] = {
            "middle": lambda a, b: (a + b) / 2,
            "best_ask": lambda a, b: a,
            "best_bid": lambda a, b: b,
            "vwap": self._compute_vwap,
        }
        self.compute_price = price_funcs[price_formula]

    @classmethod
    def from_params(
        cls,
        model_type: Literal["discrete", "continuous"],
        duration: float,
        n_frames: int,
        xmin: float,
        xmax: float,
        n_grid: int,
        D: float,
        L: float | npt.ArrayLike,
        dt_step: float,
        nu: float = 0.0,
        alpha: float = 0.0,
        metaorder: Optional[npt.ArrayLike] = None,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        price_formula: Literal["middle", "best_ask", "best_bid", "vwap"] = "middle",
        measured_quantities: Optional[list[str]] = None,
        measurement_indices: Optional[list[int]] = None,
        seed: Optional[int] = None,
        **_: Any,
    ) -> Self:
        """Build a Simulation, constructing the book from raw parameters."""
        assert model_type in ("discrete", "continuous")
        if seed is not None:
            np.random.seed(seed)

        book = _build_book(model_type, D, xmin, xmax, n_grid, L, nu, alpha)
        return cls(
            book=book,
            duration=duration,
            n_frames=n_frames,
            dt_step=dt_step,
            metaorder=metaorder,
            frame_start=frame_start,
            frame_end=frame_end,
            price_formula=price_formula,
            measured_quantities=measured_quantities,
            measurement_indices=measurement_indices,
        )

    def _expand_metaorder(self, metaorder: Optional[npt.ArrayLike]) -> npt.NDArray[np.float64]:
        if metaorder is None:
            return np.zeros(self.n_frames, dtype=np.float64)
        arr = np.asarray(metaorder, dtype=np.float64)
        if arr.size == 1:
            full = np.zeros(self.n_frames, dtype=np.float64)
            full[self.frame_start : self.frame_end] = float(arr.reshape(-1)[0])
            return full
        assert arr.size == self.n_frames, (
            f"metaorder length {arr.size} != n_frames={self.n_frames}"
        )
        return arr

    def _compute_vwap(self, best_ask: float, best_bid: float) -> float:
        ask_vol = abs(self.book.best_ask_volume)
        bid_vol = abs(self.book.best_bid_volume)
        total_vol = ask_vol + bid_vol
        if total_vol == 0:
            return (best_ask + best_bid) / 2
        return (ask_vol * best_ask + bid_vol * best_bid) / total_vol

    # ================== RUN ==================

    def run(self) -> None:
        """Advance the book for ``n_frames`` frames and record state each frame."""
        for n in range(self.n_frames):
            dq = self.metaorder[n] * self.dt
            self.book.evolve(self.dt, dq, self.dt_step)

            self.asks[n] = self.book.best_ask
            self.bids[n] = self.book.best_bid
            self.prices[n] = self.compute_price(self.book.best_ask, self.book.best_bid)
            self.spreads[n] = self.book.best_ask - self.book.best_bid
            self._measure(n)

    def _measure(self, n: int) -> None:
        if n not in self.measurement_indices:
            return
        for quantity in self.measured_quantities:
            self.measurements[quantity].append(np.copy(self.book.get_measure(quantity)))

    def __str__(self) -> str:
        return (
            f"Simulation: duration={self.duration}, n_frames={self.n_frames}, "
            f"dt={self.dt:.3e}, dt_step={self.dt_step:.3e}, "
            f"book={type(self.book).__name__}, D={self.book.D}, dx={self.book.dx:.3e}"
        )


def _build_book(
    model_type: str,
    D: float,
    xmin: float,
    xmax: float,
    n_grid: int,
    L: float | npt.ArrayLike,
    nu: float,
    alpha: float,
) -> BookType:
    """Construct the appropriate book from raw parameters."""
    if not np.isscalar(L):
        L_arr = np.atleast_1d(L)
        nu_arr = np.zeros_like(L_arr) if np.isscalar(nu) else np.atleast_1d(nu)
        lambd_arr = L_arr * np.sqrt(nu_arr * D)
        return MultiDiscreteBook.from_params(
            D=D, xmin=xmin, xmax=xmax, n_grid=n_grid,
            L_list=list(L_arr), nu_list=list(nu_arr), lambd_list=list(lambd_arr),
        )

    L_scalar = float(L)
    if model_type == "continuous":
        return LinearContinuousBook.from_params(
            D=D, L=L_scalar, xmin=xmin, xmax=xmax, n_grid=n_grid,
        )

    if nu == 0:
        return LinearDiscreteBook.from_params(
            D=D, xmin=xmin, xmax=xmax, n_grid=n_grid, L=L_scalar,
        )

    lambd = L_scalar * np.sqrt(nu * D)
    return DiscreteBook.from_params(
        D=D, xmin=xmin, xmax=xmax, n_grid=n_grid,
        L=L_scalar, nu=nu, lambd=lambd, alpha=alpha,
    )


def standard_parameters(
    participation_rate: float,
    model_type: Literal["discrete", "continuous"],
    duration: float = 5000.0,
    n_frames: int = 100,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> dict[str, Any]:
    """
    Generate well-behaved simulation parameters for a given participation rate
    ``r = m0 / (D * L)``. The returned dict can be passed straight to
    ``Simulation.from_params(**params)``.
    """
    r = abs(participation_rate)
    D = 0.5
    Ipt = np.sqrt(2 * r * D * duration)

    if xmin is None:
        xmin = -1.1 * Ipt
    if xmax is None:
        xmax = 1.1 * Ipt

    if r <= 1:
        boundary_distance = np.sqrt(D * duration)
        xmax = max(np.sqrt(r) * xmax, boundary_distance)
        xmin = min(np.sqrt(r) * xmin, -boundary_distance)

    n_grid = max(int(xmax - xmin), 100)
    dx = (xmax - xmin) / n_grid
    L = 10 / (dx * dx)

    if r == float("inf"):
        D = 0.0
        m0 = (L * max(abs(xmin), abs(xmax))) / (5 * duration)
        dt_step = duration / (n_frames * 100)
    else:
        m0 = D * L * participation_rate
        dt_step = courant_dt(dx, D)

    return {
        "model_type": model_type,
        "duration": duration,
        "n_frames": n_frames,
        "n_grid": n_grid,
        "xmin": xmin,
        "xmax": xmax,
        "D": D,
        "L": L,
        "nu": 0.0,
        "dt_step": dt_step,
        "metaorder": [m0],
    }
