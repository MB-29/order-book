"""
Shared pytest fixtures for LLOB test suite.
"""

import numpy as np
import pytest

from llob import (
    DiscreteBook,
    LimitOrders,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
    Simulation,
)

# =============================================================================
# Grid Fixtures
# =============================================================================


@pytest.fixture
def small_grid() -> dict:
    """Small grid for fast tests."""
    return {
        "xmin": -10.0,
        "xmax": 10.0,
        "Nx": 20,
    }


@pytest.fixture
def standard_grid() -> dict:
    """Standard grid for typical tests."""
    return {
        "xmin": -50.0,
        "xmax": 50.0,
        "Nx": 100,
    }


# =============================================================================
# Parameter Fixtures
# =============================================================================


@pytest.fixture
def linear_params(small_grid: dict) -> dict:
    """Parameters for linear regime (nu=0)."""
    return {
        **small_grid,
        "D": 0.5,
        "L": 10.0,
        "nu": 0.0,
        "lambd": 0.0,
    }


@pytest.fixture
def nonlinear_params(small_grid: dict) -> dict:
    """Parameters for nonlinear regime (nu>0)."""
    D = 0.5
    nu = 0.1
    L = 10.0
    lambd = L * np.sqrt(nu * D)
    return {
        **small_grid,
        "D": D,
        "L": L,
        "nu": nu,
        "lambd": lambd,
    }


@pytest.fixture
def simulation_params(small_grid: dict) -> dict:
    """Ready-to-use Simulation parameters."""
    return {
        "model_type": "discrete",
        "T": 100,
        "Nt": 10,
        **small_grid,
        "D": 0.5,
        "L": 10.0,
        "nu": 0.0,
        "metaorder": [5.0],
    }


@pytest.fixture
def continuous_simulation_params(small_grid: dict) -> dict:
    """Simulation parameters for continuous model."""
    return {
        "model_type": "continuous",
        "T": 100,
        "Nt": 10,
        **small_grid,
        "D": 0.5,
        "L": 10.0,
        "nu": 0.0,
        "metaorder": [5.0],
    }


@pytest.fixture
def multi_book_params(small_grid: dict) -> dict:
    """Parameters for multi-actor book."""
    return {
        **small_grid,
        "D": 0.5,
        "L_list": [5.0, 5.0],
        "nu_list": [0.0, 0.0],
        "lambd_list": [0.0, 0.0],
    }


# =============================================================================
# LimitOrders Fixtures
# =============================================================================


@pytest.fixture
def limit_orders_ask(linear_params: dict) -> LimitOrders:
    """LimitOrders instance for ask side."""
    return LimitOrders.from_params(
        side="ask",
        lambd=linear_params["lambd"],
        nu=linear_params["nu"],
        D=linear_params["D"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        Nx=linear_params["Nx"],
        L=linear_params["L"],
        initial_density="linear",
        boundary_conditions="linear",
    )


@pytest.fixture
def limit_orders_bid(linear_params: dict) -> LimitOrders:
    """LimitOrders instance for bid side."""
    return LimitOrders.from_params(
        side="bid",
        lambd=linear_params["lambd"],
        nu=linear_params["nu"],
        D=linear_params["D"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        Nx=linear_params["Nx"],
        L=linear_params["L"],
        initial_density="linear",
        boundary_conditions="linear",
    )


@pytest.fixture
def limit_orders_nonlinear(nonlinear_params: dict) -> LimitOrders:
    """LimitOrders with nonlinear dynamics."""
    return LimitOrders.from_params(
        side="ask",
        lambd=nonlinear_params["lambd"],
        nu=nonlinear_params["nu"],
        D=nonlinear_params["D"],
        xmin=nonlinear_params["xmin"],
        xmax=nonlinear_params["xmax"],
        Nx=nonlinear_params["Nx"],
        L=nonlinear_params["L"],
        initial_density="stationary",
        boundary_conditions="flat",
    )


# =============================================================================
# Book Fixtures
# =============================================================================


@pytest.fixture
def discrete_book(linear_params: dict) -> DiscreteBook:
    """DiscreteBook instance for linear regime."""
    return DiscreteBook.from_params(
        D=linear_params["D"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        Nx=linear_params["Nx"],
        L=linear_params["L"],
        nu=linear_params["nu"],
        lambd=linear_params["lambd"],
        initial_density="linear",
        boundary_conditions="linear",
    )


@pytest.fixture
def nonlinear_discrete_book(nonlinear_params: dict) -> DiscreteBook:
    """DiscreteBook instance for nonlinear regime (nu > 0)."""
    return DiscreteBook.from_params(
        D=nonlinear_params["D"],
        xmin=nonlinear_params["xmin"],
        xmax=nonlinear_params["xmax"],
        Nx=nonlinear_params["Nx"],
        L=nonlinear_params["L"],
        nu=nonlinear_params["nu"],
        lambd=nonlinear_params["lambd"],
        initial_density="stationary",
        boundary_conditions="flat",
    )


@pytest.fixture
def linear_discrete_book(linear_params: dict) -> LinearDiscreteBook:
    """LinearDiscreteBook instance."""
    return LinearDiscreteBook.from_params(
        D=linear_params["D"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        Nx=linear_params["Nx"],
        L=linear_params["L"],
    )


@pytest.fixture
def continuous_book(linear_params: dict) -> LinearContinuousBook:
    """LinearContinuousBook instance."""
    return LinearContinuousBook.from_params(
        D=linear_params["D"],
        L=linear_params["L"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        Nx=linear_params["Nx"],
    )


@pytest.fixture
def multi_book(multi_book_params: dict) -> MultiDiscreteBook:
    """MultiDiscreteBook instance."""
    return MultiDiscreteBook.from_params(
        D=multi_book_params["D"],
        xmin=multi_book_params["xmin"],
        xmax=multi_book_params["xmax"],
        Nx=multi_book_params["Nx"],
        L_list=multi_book_params["L_list"],
        nu_list=multi_book_params["nu_list"],
        lambd_list=multi_book_params["lambd_list"],
    )


# =============================================================================
# Simulation Fixtures
# =============================================================================


@pytest.fixture
def simulation(simulation_params: dict) -> Simulation:
    """Simulation instance for discrete model."""
    return Simulation.from_params(**simulation_params)


@pytest.fixture
def continuous_simulation(continuous_simulation_params: dict) -> Simulation:
    """Simulation instance for continuous model."""
    return Simulation.from_params(**continuous_simulation_params)


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def seed_random():
    """Seed random number generators for reproducibility."""
    np.random.seed(42)
    yield
    # Reset to non-deterministic after test
    np.random.seed(None)
