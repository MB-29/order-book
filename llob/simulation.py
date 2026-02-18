"""
LLOB simulation runner.
"""

import warnings
from collections.abc import Callable
from typing import Any, Literal, Optional, Self

import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .books import (
    DiscreteBook,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
)
from .configs import SimulationConfig

# Type alias for book types
BookType = DiscreteBook | LinearDiscreteBook | LinearContinuousBook | MultiDiscreteBook


class Simulation:
    """
    LLOB simulation runner.

    Orchestrates the simulation of a Latent Order Book, handling time evolution,
    metaorder execution, price tracking, and optional animation.

    Attributes:
        book: The underlying order book instance.
        prices: Array of prices at each time step.
        asks: Array of best ask prices at each time step.
        bids: Array of best bid prices at each time step.
        metaorder: Array of metaorder intensities over time.
        impact_th: Theoretical price impact.
        participation_rate: Normalized metaorder rate (m0 / (D * L)).
    """

    def __init__(
        self,
        book: BookType,
        model_type: Literal["discrete", "continuous"],
        T: int,
        Nt: int,
        xmin: float,
        xmax: float,
        Nx: int,
        D: float,
        L: float | npt.NDArray[np.float64],
        nu: float,
        metaorder: npt.NDArray[np.float64],
        n_start: int,
        n_end: int,
        price_formula: Literal["middle", "best_ask", "best_bid", "vwap"],
        measured_quantities: list[str],
        measurement_indices: list[int],
    ) -> None:
        """
        Initialize a Simulation with pre-constructed components.

        Use `from_params` classmethod for convenient construction.
        """
        self.book = book
        self.model_type = model_type
        self.T = T
        self.Nt = Nt
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.D = D
        self.L = L
        self.nu = nu
        self.metaorder = metaorder
        self.n_start = n_start
        self.n_end = n_end
        self.price_formula = price_formula
        self.measured_quantities = measured_quantities
        self.measurement_indices = measurement_indices

        # Derived values
        self.price_range = xmax - xmin
        self.dx = self.price_range / Nx
        self.boundary_distance = min(abs(xmin), xmax)
        self.time_interval, self.tstep = np.linspace(0, T, num=Nt, retstep=True)
        self.tstep = float(self.tstep)
        self.dt = self.dx**2 / (2 * D) if D > 0 else float("inf")

        self.is_multi_book = not np.isscalar(L)
        self.lambd = float(np.max(L)) * np.sqrt(nu * D) if np.isscalar(L) else 0.0
        self.J = D * float(np.max(L))

        # Metaorder stats
        self.m0 = (
            float(np.mean(metaorder[metaorder != 0])) if np.any(metaorder != 0) else 0.0
        )
        self.t_start = n_start * self.tstep
        self.t_end = n_end * self.tstep
        self.time_interval_shifted = self.time_interval - self.t_start

        # Output arrays
        self.asks = np.zeros(T)
        self.bids = np.zeros(T)
        self.prices = np.zeros(T)
        self.measurements: dict[str, list[Any]] = {q: [] for q in measured_quantities}

        # Price computation function
        price_funcs: dict[str, Callable[[float, float], float]] = {
            "middle": lambda a, b: (a + b) / 2,
            "best_ask": lambda a, b: a,
            "best_bid": lambda a, b: b,
            "vwap": self._compute_vwap,
        }
        self.compute_price = price_funcs[price_formula]

        # Compute theoretical values
        self._compute_theoretical_values()

        # Animation state
        self.price_ax: Optional[Axes] = None
        self.price_line: Optional[Line2D] = None
        self.best_ask_line: Optional[Line2D] = None
        self.best_bid_line: Optional[Line2D] = None
        self.animation: Optional[FuncAnimation] = None
        self.ymin = xmin
        self.ymax = xmax

    @classmethod
    def from_params(
        cls,
        model_type: Literal["discrete", "continuous"],
        metaorder: Optional[list[float] | npt.NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a Simulation from raw parameters.

        This is the primary constructor, maintaining backward compatibility
        with the original API.

        Args:
            model_type: Either 'discrete' or 'continuous'.
            metaorder: Metaorder intensity over time. If length 1, treated as constant.
            **kwargs: Additional parameters:
                - T: Total time steps (default: 1)
                - Nt: Number of output time points (default: 100)
                - xmin, xmax: Price interval bounds (required)
                - Nx: Number of spatial grid points (default: 100)
                - D: Diffusion constant (required)
                - L: Latent liquidity (required, can be array for multi-book)
                - nu: Cancellation rate (default: 0)
                - price_formula: 'middle', 'best_ask', 'best_bid', or 'vwap'
                - n_start, n_end: Metaorder start/end indices
                - measured_quantities: List of quantities to measure
                - measurement_indices: Time indices at which to measure

        Returns:
            Configured Simulation instance.
        """
        assert model_type in (
            "discrete",
            "continuous",
        ), f"model_type must be 'discrete' or 'continuous', got {model_type}"

        # Extract parameters with defaults
        T = kwargs.get("T", 1)
        Nt = kwargs.get("Nt", 100)
        xmin = kwargs["xmin"]
        xmax = kwargs["xmax"]
        Nx = kwargs.get("Nx", 100)
        D = kwargs["D"]
        L = kwargs["L"]
        nu = kwargs.get("nu", 0)
        price_formula = kwargs.get("price_formula", "middle")
        n_start = kwargs.get("n_start", 0)
        n_end = kwargs.get("n_end", Nt)
        measured_quantities = kwargs.get("measured_quantities", [])
        measurement_indices = kwargs.get("measurement_indices", [])

        # Process metaorder
        if metaorder is None:
            metaorder = [0]
        metaorder_arr = np.asarray(metaorder, dtype=np.float64)
        if len(metaorder_arr) == 1:
            full_metaorder = np.zeros(T, dtype=np.float64)
            full_metaorder[n_start:n_end] = metaorder_arr[0]
        else:
            assert (
                len(metaorder_arr) == T
            ), f"metaorder length {len(metaorder_arr)} != T={T}"
            full_metaorder = metaorder_arr

        # Create the appropriate book
        book = cls._create_book(model_type, D, xmin, xmax, Nx, L, nu)

        return cls(
            book=book,
            model_type=model_type,
            T=T,
            Nt=Nt,
            xmin=xmin,
            xmax=xmax,
            Nx=Nx,
            D=D,
            L=L,
            nu=nu,
            metaorder=full_metaorder,
            n_start=n_start,
            n_end=n_end,
            price_formula=price_formula,
            measured_quantities=measured_quantities,
            measurement_indices=measurement_indices,
        )

    @classmethod
    def from_config(cls, config: SimulationConfig) -> Self:
        """
        Create a Simulation from a SimulationConfig.

        This is the preferred constructor using typed configuration.

        Args:
            config: Simulation configuration object.

        Returns:
            Configured Simulation instance.
        """
        # Extract values from config
        model_type = config.model_type
        T = config.T
        Nt = config.Nt
        xmin = config.grid.xmin
        xmax = config.grid.xmax
        Nx = config.grid.Nx
        D = config.D
        L = config.L if not isinstance(config.L, list) else np.array(config.L)
        nu = config.nu if not isinstance(config.nu, list) else np.array(config.nu)
        n_start = config.effective_n_start
        n_end = config.effective_n_end
        price_formula = config.price_formula
        measured_quantities = config.measured_quantities
        measurement_indices = config.measurement_indices

        # Get full metaorder array
        full_metaorder = config.get_full_metaorder()

        # Create the appropriate book
        book = cls._create_book(model_type, D, xmin, xmax, Nx, L, nu)

        return cls(
            book=book,
            model_type=model_type,
            T=T,
            Nt=Nt,
            xmin=xmin,
            xmax=xmax,
            Nx=Nx,
            D=D,
            L=L,
            nu=nu,
            metaorder=full_metaorder,
            n_start=n_start,
            n_end=n_end,
            price_formula=price_formula,
            measured_quantities=measured_quantities,
            measurement_indices=measurement_indices,
        )

    @staticmethod
    def _create_book(
        model_type: str,
        D: float,
        xmin: float,
        xmax: float,
        Nx: int,
        L: float | npt.NDArray[np.float64],
        nu: float,
    ) -> BookType:
        """Create the appropriate order book instance."""
        is_multi = not np.isscalar(L)

        if is_multi:
            L_arr = np.atleast_1d(L)
            # For multi-book, nu should also be an array
            nu_arr = np.zeros_like(L_arr) if np.isscalar(nu) else np.atleast_1d(nu)
            lambd_arr = L_arr * np.sqrt(nu_arr * D)
            return MultiDiscreteBook.from_params(
                D=D,
                xmin=xmin,
                xmax=xmax,
                Nx=Nx,
                L_list=list(L_arr),
                nu_list=list(nu_arr),
                lambd_list=list(lambd_arr),
            )

        L_scalar = float(L)
        linear = nu == 0

        if model_type == "discrete":
            if linear:
                return LinearDiscreteBook.from_params(
                    D=D, xmin=xmin, xmax=xmax, Nx=Nx, L=L_scalar
                )
            else:
                lambd = L_scalar * np.sqrt(nu * D)
                return DiscreteBook.from_params(
                    D=D, xmin=xmin, xmax=xmax, Nx=Nx, L=L_scalar, nu=nu, lambd=lambd
                )
        else:  # continuous
            return LinearContinuousBook.from_params(
                D=D, L=L_scalar, xmin=xmin, xmax=xmax, Nx=Nx
            )

    def _compute_theoretical_values(self) -> None:
        """Compute theoretical predictions and validate parameters."""
        L_max = float(np.max(self.L))

        self.n_steps = int(self.tstep / self.dt) if self.dt < float("inf") else 0
        self.boundary_factor = (
            np.sqrt(self.D * self.T) / self.boundary_distance
            if self.boundary_distance > 0
            else float("inf")
        )
        self.infinity_density = L_max * self.xmax
        self.impact_th = (
            np.sqrt(2 * abs(self.m0) * self.T / L_max) if L_max > 0 else 0.0
        )
        self.density_shift_th = np.sqrt(abs(self.m0) * self.T * L_max)
        self.participation_rate = (
            self.m0 / (self.D * L_max) if self.D * L_max != 0 else float("inf")
        )
        self.r = abs(self.participation_rate)
        self.scheme_constant = self.D * self.tstep / (self.dx * self.dx)
        self.lower_impact = np.sqrt(abs(self.r) / (2 * np.pi)) * self.impact_th
        self.first_volume = L_max * self.dx * self.dx

        # Warnings
        if self.boundary_factor > 1:
                warnings.warn("Boundary effects may be significant", stacklevel=2)
        if self.model_type == "discrete":
            if self.r < 1 and self.n_steps < 100:
                warnings.warn(
                    f"Low number of diffusion steps ({self.n_steps} < 100), "
                    "try increasing spatial resolution.",
                    stacklevel=2,
                )
            if self.n_steps < 1 and self.r < float("inf"):
                raise ValueError(
                    "Order diffusion not possible: diffusion distance smaller than "
                    f"grid spacing. dt={self.dt}, tstep={self.tstep}"
                )

    def _compute_vwap(self, best_ask: float, best_bid: float) -> float:
        """Compute volume-weighted average price."""
        ask_vol = abs(self.book.best_ask_volume)
        bid_vol = abs(self.book.best_bid_volume)
        total_vol = ask_vol + bid_vol
        if total_vol == 0:
            return (best_ask + best_bid) / 2
        return (ask_vol * best_ask + bid_vol * best_bid) / total_vol

    def get_density(self) -> dict[str, npt.NDArray[np.int64]]:
        """Get current order density for both sides."""
        return {
            "bid": self.book.get_ask_volumes(),
            "ask": self.book.get_bid_volumes(),
        }

    def get_growth_th(self) -> npt.NDArray[np.float64]:
        """
        Get theoretical price impact profile.

        Returns:
            Array of theoretical prices from t_start to t_end.
        """
        L_max = float(np.max(self.L))
        if self.r < 1:
            A = self.m0 / (L_max * np.sqrt(self.D * np.pi))
        else:
            A = np.sign(self.m0) * np.sqrt(2) * np.sqrt(self.m0 / L_max)
        growth = A * np.sqrt(self.time_interval_shifted[self.n_start : self.n_end])
        return growth

    # ================== RUN ==================

    def run(
        self,
        fig: Optional[Figure] = None,
        animation: bool = False,
        save: bool = False,
    ) -> None:
        """
        Run the simulation.

        Args:
            fig: Figure for animation (required if animation=True).
            animation: Whether to display animation.
            save: Whether to save animation to file.
        """
        if animation:
            self._run_animation(fig, save)
            return

        for n in range(self.T):
            self.asks[n] = self.book.best_ask
            self.bids[n] = self.book.best_bid
            self.prices[n] = self.compute_price(self.book.best_ask, self.book.best_bid)
            self._measure(n)

            volume = self.metaorder[n] * self.dt
            self.book.timestep(self.dt, volume)

    def _measure(self, n: int) -> None:
        """Record measurements at specified indices."""
        if n not in self.measurement_indices:
            return
        for quantity in self.measured_quantities:
            value = self.book.get_measure(quantity)
            self.measurements[quantity].append(np.copy(value))

    # ================== ANIMATION ==================

    def _run_animation(self, fig: Figure | None, save: bool = False) -> None:
        """Run simulation with animation."""
        if fig is None:
            raise ValueError("Figure required for animation")

        self._set_animation(fig)
        self.animation = FuncAnimation(
            fig,
            self._update_animation,
            init_func=self._init_animation,
            repeat=False,
            frames=self.Nt,
            blit=True,
        )
        if save:
            self.animation.save("../animation.gif", writer="imagemagick", fps=60)

    def _set_animation(self, fig: Figure) -> None:
        """Set up animation components."""
        self.ymin, self.ymax = self.xmin, self.xmax
        self.book.set_animation(fig, {})

        self.price_ax = fig.add_subplot(2, 1, 2)
        self.price_ax.set_title("Price evolution")
        (self.best_ask_line,) = self.price_ax.plot(
            [], [], label="Best Ask", color="blue", ls="--"
        )
        (self.best_bid_line,) = self.price_ax.plot(
            [], [], label="Best Bid", color="red", ls="--"
        )
        (self.price_line,) = self.price_ax.plot(
            [], [], label=f"Price ({self.price_formula})", color="yellow"
        )
        self.price_ax.plot([0, self.T], [0, 0], ls="dashed", lw=0.5, color="black")
        self.price_ax.legend()
        self.price_ax.set_ylim((self.ymin, self.ymax))
        self.price_ax.set_xlim((0, self.T))

    def _init_animation(self) -> list[Any]:
        """Initialize animation frame."""
        if (
            self.price_line is None
            or self.best_bid_line is None
            or self.best_ask_line is None
        ):
            return []
        self.price_line.set_data([], [])
        self.best_bid_line.set_data([], [])
        self.best_ask_line.set_data([], [])
        return self.book.init_animation() + [
            self.price_line,
            self.best_ask_line,
            self.best_bid_line,
        ]

    def _update_animation(self, n: int) -> list[Any]:
        """Update animation for frame n."""
        if (
            self.price_line is None
            or self.best_bid_line is None
            or self.best_ask_line is None
        ):
            return []

        if n % 10 == 0:
            print(f"Step {n}")

        volume = self.metaorder[n] * self.dt
        self.asks[n] = self.book.best_ask
        self.bids[n] = self.book.best_bid
        self.prices[n] = self.compute_price(self.book.best_ask, self.book.best_bid)
        self._measure(n)

        self.price_line.set_data(self.time_interval[: n + 1], self.prices[: n + 1])
        self.best_ask_line.set_data(self.time_interval[: n + 1], self.asks[: n + 1])
        self.best_bid_line.set_data(self.time_interval[: n + 1], self.bids[: n + 1])

        return self.book.update_animation(self.tstep, volume) + [
            self.price_line,
            self.best_ask_line,
            self.best_bid_line,
        ]

    def __str__(self) -> str:
        """Return string representation of simulation parameters."""
        L_val = self.L if np.isscalar(self.L) else list(self.L)
        return f"""Order book simulation.
        Time parameters:
            T = {self.T}
            Nt = {self.Nt}
            tstep = {self.tstep:.1e}
            dt = {self.dt:.1e}
            n_steps = {self.n_steps}

        Space parameters:
            Price interval = [{self.xmin}, {self.xmax}]
            Nx = {self.Nx}
            dx = {self.dx:.1e}

        Model constants:
            D = {self.D:.1e}
            lambda = {self.lambd}
            nu = {self.nu}
            L = {L_val}
            J = {self.J}

        Metaorder:
            m0 = {self.m0:.1e}
            dq = {self.m0 * self.tstep:.1e}
            n_start, n_end = ({self.n_start}, {self.n_end})

        Theoretical values:
            Participation rate = {self.participation_rate:.1e}
            impact = {self.impact_th:.1e}
            lower impact = {self.lower_impact}
            alpha = {self.scheme_constant:.1e}
            boundary factor = {self.boundary_factor:.1e}
            first volume = {self.first_volume:.1f}
            lower resolution = {self.lower_impact / self.dx:.1e}
"""


def standard_parameters(
    participation_rate: float,
    model_type: Literal["discrete", "continuous"],
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    Nt: Optional[int] = None,
    T: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate standard simulation parameters for a given participation rate.

    Args:
        participation_rate: Normalized metaorder rate (m0 / (D * L)).
        model_type: Either 'discrete' or 'continuous'.
        xmin: Price interval lower bound (auto-computed if None).
        xmax: Price interval upper bound (auto-computed if None).
        Nt: Number of output time points (default: 100).
        T: Total time steps (default: Nt * 50).

    Returns:
        Dictionary of simulation parameters.

    Warning:
        Participation rates greater than ~2500 may cause errors.
    """
    if Nt is None:
        Nt = 100
    if T is None:
        T = Nt * 50

    r = abs(participation_rate)
    D = 0.5
    Ipt = np.sqrt(2 * r * D * T)

    if xmin is None:
        xmin = -1.1 * Ipt
    if xmax is None:
        xmax = 1.1 * Ipt

    if r <= 1:
        boundary_distance = np.sqrt(D * T)
        xmax = max(np.sqrt(r) * xmax, boundary_distance)
        xmin = min(np.sqrt(r) * xmin, -boundary_distance)

    Nx = int(xmax - xmin)
    X = max(abs(xmin), abs(xmax))
    dx = (xmax - xmin) / Nx
    L = 10 / (dx * dx)

    if r == float("inf"):
        D = 0.0
        m0 = (L * X) / (5 * T)
    else:
        D = 0.5
        m0 = D * L * participation_rate

    return {
        "model_type": model_type,
        "T": T,
        "Nt": Nt,
        "Nx": Nx,
        "xmin": xmin,
        "xmax": xmax,
        "D": D,
        "metaorder": [m0],
        "L": L,
        "nu": 0,
    }
