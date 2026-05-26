"""
Shared pytest fixtures for the LLOB test suite.
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
    courant_dt,
)

# =============================================================================
# Grid fixtures
# =============================================================================


@pytest.fixture
def small_grid() -> dict:
    return {"xmin": -10.0, "xmax": 10.0, "n_grid": 20}


@pytest.fixture
def standard_grid() -> dict:
    return {"xmin": -50.0, "xmax": 50.0, "n_grid": 100}


# =============================================================================
# Parameter fixtures
# =============================================================================


def _dt_step(params: dict) -> float:
    dx = (params["xmax"] - params["xmin"]) / params["n_grid"]
    return courant_dt(dx, params["D"])


@pytest.fixture
def linear_params(small_grid: dict) -> dict:
    return {**small_grid, "D": 0.5, "L": 10.0, "nu": 0.0, "lambd": 0.0}


@pytest.fixture
def nonlinear_params(small_grid: dict) -> dict:
    D, nu, L = 0.5, 0.1, 10.0
    return {
        **small_grid,
        "D": D, "L": L, "nu": nu,
        "lambd": L * np.sqrt(nu * D),
    }


@pytest.fixture
def simulation_params(small_grid: dict) -> dict:
    base = {
        "model_type": "discrete",
        "duration": 100.0,
        "n_frames": 10,
        **small_grid,
        "D": 0.5,
        "L": 10.0,
        "nu": 0.0,
        "metaorder": [5.0],
    }
    base["dt_step"] = _dt_step(base)
    return base


@pytest.fixture
def continuous_simulation_params(small_grid: dict) -> dict:
    base = {
        "model_type": "continuous",
        "duration": 100.0,
        "n_frames": 10,
        **small_grid,
        "D": 0.5,
        "L": 10.0,
        "nu": 0.0,
        "metaorder": [5.0],
    }
    base["dt_step"] = _dt_step(base)
    return base


@pytest.fixture
def multi_book_params(small_grid: dict) -> dict:
    return {
        **small_grid,
        "D": 0.5,
        "L_list": [5.0, 5.0],
        "nu_list": [0.0, 0.0],
        "lambd_list": [0.0, 0.0],
    }


# =============================================================================
# LimitOrders fixtures
# =============================================================================


def _make_limit_orders(params: dict, side: str, **overrides) -> LimitOrders:
    kwargs = dict(
        side=side,
        lambd=params["lambd"],
        nu=params["nu"],
        D=params["D"],
        xmin=params["xmin"],
        xmax=params["xmax"],
        n_grid=params["n_grid"],
        L=params["L"],
        initial_density="linear",
        boundary_conditions="linear",
    )
    kwargs.update(overrides)
    return LimitOrders.from_params(**kwargs)


@pytest.fixture
def limit_orders_ask(linear_params: dict) -> LimitOrders:
    return _make_limit_orders(linear_params, "ask")


@pytest.fixture
def limit_orders_bid(linear_params: dict) -> LimitOrders:
    return _make_limit_orders(linear_params, "bid")


@pytest.fixture
def limit_orders_nonlinear(nonlinear_params: dict) -> LimitOrders:
    return _make_limit_orders(
        nonlinear_params, "ask",
        initial_density="stationary",
        boundary_conditions="flat",
    )


# =============================================================================
# Book fixtures
# =============================================================================


@pytest.fixture
def discrete_book(linear_params: dict) -> DiscreteBook:
    return DiscreteBook.from_params(
        D=linear_params["D"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        n_grid=linear_params["n_grid"],
        L=linear_params["L"],
        nu=linear_params["nu"],
        lambd=linear_params["lambd"],
        initial_density="linear",
        boundary_conditions="linear",
    )


@pytest.fixture
def nonlinear_discrete_book(nonlinear_params: dict) -> DiscreteBook:
    return DiscreteBook.from_params(
        D=nonlinear_params["D"],
        xmin=nonlinear_params["xmin"],
        xmax=nonlinear_params["xmax"],
        n_grid=nonlinear_params["n_grid"],
        L=nonlinear_params["L"],
        nu=nonlinear_params["nu"],
        lambd=nonlinear_params["lambd"],
        initial_density="stationary",
        boundary_conditions="flat",
    )


@pytest.fixture
def linear_discrete_book(linear_params: dict) -> LinearDiscreteBook:
    return LinearDiscreteBook.from_params(
        D=linear_params["D"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        n_grid=linear_params["n_grid"],
        L=linear_params["L"],
    )


@pytest.fixture
def continuous_book(linear_params: dict) -> LinearContinuousBook:
    return LinearContinuousBook.from_params(
        D=linear_params["D"],
        L=linear_params["L"],
        xmin=linear_params["xmin"],
        xmax=linear_params["xmax"],
        n_grid=linear_params["n_grid"],
    )


@pytest.fixture
def multi_book(multi_book_params: dict) -> MultiDiscreteBook:
    return MultiDiscreteBook.from_params(
        D=multi_book_params["D"],
        xmin=multi_book_params["xmin"],
        xmax=multi_book_params["xmax"],
        n_grid=multi_book_params["n_grid"],
        L_list=multi_book_params["L_list"],
        nu_list=multi_book_params["nu_list"],
        lambd_list=multi_book_params["lambd_list"],
    )


# =============================================================================
# Simulation fixtures
# =============================================================================


@pytest.fixture
def simulation(simulation_params: dict) -> Simulation:
    return Simulation.from_params(**simulation_params)


@pytest.fixture
def continuous_simulation(continuous_simulation_params: dict) -> Simulation:
    return Simulation.from_params(**continuous_simulation_params)


# =============================================================================
# Utility fixtures
# =============================================================================


@pytest.fixture
def seed_random():
    np.random.seed(42)
    yield
    np.random.seed(None)
