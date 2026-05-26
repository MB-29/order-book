"""
Tests for the Simulation class.
"""

import numpy as np
import pytest

from llob import (
    DiscreteBook,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
    Simulation,
    courant_dt,
    standard_parameters,
)


def _params(**overrides) -> dict:
    base = {
        "model_type": "discrete",
        "duration": 100.0,
        "n_frames": 10,
        "xmin": -10.0,
        "xmax": 10.0,
        "n_grid": 20,
        "D": 0.5,
        "L": 10.0,
        "nu": 0.0,
        "metaorder": [1.0],
    }
    base.update(overrides)
    base["dt_step"] = courant_dt(
        (base["xmax"] - base["xmin"]) / base["n_grid"], base["D"]
    )
    return base


class TestSimulationConstruction:
    def test_from_params_discrete_model(self, simulation_params: dict):
        sim = Simulation.from_params(**simulation_params)
        assert isinstance(sim.book, (DiscreteBook, LinearDiscreteBook))

    def test_from_params_continuous_model(self, continuous_simulation_params: dict):
        sim = Simulation.from_params(**continuous_simulation_params)
        assert isinstance(sim.book, LinearContinuousBook)

    def test_from_params_multi_book(self):
        sim = Simulation.from_params(**_params(L=np.array([5.0, 5.0])))
        assert isinstance(sim.book, MultiDiscreteBook)

    def test_from_params_validates_model_type(self, simulation_params: dict):
        params = dict(simulation_params)
        params["model_type"] = "invalid"
        with pytest.raises(AssertionError):
            Simulation.from_params(**params)

    def test_from_params_expands_single_metaorder(self, simulation_params: dict):
        sim = Simulation.from_params(**simulation_params)
        assert len(sim.metaorder) == sim.n_frames

    def test_from_params_accepts_full_metaorder(self):
        Nt = 10
        metaorder = np.random.randn(Nt)
        sim = Simulation.from_params(**_params(n_frames=Nt, metaorder=metaorder))
        np.testing.assert_array_equal(sim.metaorder, metaorder)


class TestSimulationCFLWarning:
    def test_warns_on_cfl_violation(self):
        params = _params()
        params["dt_step"] = 100.0  # way above dx²/(2D)
        with pytest.warns(UserWarning, match="CFL"):
            Simulation.from_params(**params)


class TestSimulationPriceFormulas:
    def test_price_formula_middle(self, simulation_params: dict):
        params = dict(simulation_params, price_formula="middle")
        sim = Simulation.from_params(**params)
        assert sim.compute_price(10.0, 8.0) == 9.0

    def test_price_formula_best_ask(self, simulation_params: dict):
        params = dict(simulation_params, price_formula="best_ask")
        sim = Simulation.from_params(**params)
        assert sim.compute_price(10.0, 8.0) == 10.0

    def test_price_formula_best_bid(self, simulation_params: dict):
        params = dict(simulation_params, price_formula="best_bid")
        sim = Simulation.from_params(**params)
        assert sim.compute_price(10.0, 8.0) == 8.0

    def test_price_formula_vwap(self, simulation_params: dict):
        params = dict(simulation_params, price_formula="vwap")
        sim = Simulation.from_params(**params)
        price = sim.compute_price(10.0, 8.0)
        assert 8.0 <= price <= 10.0


class TestSimulationRunning:
    def test_run_populates_price_arrays(self, simulation: Simulation):
        simulation.run()
        assert len(simulation.prices) == simulation.n_frames
        assert len(simulation.asks) == simulation.n_frames
        assert len(simulation.bids) == simulation.n_frames

    def test_run_with_metaorder_moves_price(self):
        params = standard_parameters(participation_rate=10.0, model_type="discrete",
                                     duration=100.0, n_frames=20)
        sim = Simulation.from_params(**params)
        initial_price = sim.prices[0]
        sim.run()
        assert sim.prices[-1] > initial_price

    def test_run_records_measurements(self):
        params = _params(
            n_frames=100,
            measured_quantities=["best_ask", "best_bid"],
            measurement_indices=[10, 50, 90],
        )
        sim = Simulation.from_params(**params)
        sim.run()
        assert len(sim.measurements["best_ask"]) == 3
        assert len(sim.measurements["best_bid"]) == 3


class TestStandardParameters:
    def test_returns_runnable_dict(self):
        params = standard_parameters(participation_rate=1.0, model_type="discrete",
                                     duration=50.0, n_frames=10)
        for key in ("duration", "n_frames", "D", "L", "xmin", "xmax", "dt_step"):
            assert key in params

    def test_various_rates(self):
        for rate in [0.1, 1.0, 10.0, 100.0]:
            params = standard_parameters(participation_rate=rate, model_type="discrete")
            assert params["L"] > 0

    def test_creates_runnable_sim(self):
        params = standard_parameters(participation_rate=1.0, model_type="discrete",
                                     duration=50.0, n_frames=10)
        sim = Simulation.from_params(**params)
        sim.run()
        assert len(sim.prices) == params["n_frames"]


class TestSimulationBookTypes:
    def test_linear_regime_uses_linear_discrete_book(self):
        sim = Simulation.from_params(**_params(nu=0.0))
        assert isinstance(sim.book, LinearDiscreteBook)

    def test_nonlinear_regime_uses_discrete_book(self):
        sim = Simulation.from_params(**_params(nu=0.1))
        assert isinstance(sim.book, DiscreteBook)
        assert not isinstance(sim.book, LinearDiscreteBook)

    def test_continuous_model_uses_continuous_book(self):
        sim = Simulation.from_params(**_params(model_type="continuous"))
        assert isinstance(sim.book, LinearContinuousBook)


class TestSimulationStr:
    def test_str_contains_key_params(self, simulation: Simulation):
        s = str(simulation)
        assert "duration=" in s
        assert "n_frames=" in s
        assert "dt_step=" in s
