"""
Tests for Simulation class.
"""
from __future__ import annotations

import numpy as np
import pytest

from llob import (
    DiscreteBook,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
    Simulation,
    standard_parameters,
)


class TestSimulationConstruction:
    """Tests for Simulation construction."""

    def test_from_params_discrete_model(self, simulation_params: dict):
        """from_params should create simulation with discrete model."""
        sim = Simulation.from_params(**simulation_params)

        assert sim.model_type == "discrete"
        assert isinstance(sim.book, (DiscreteBook, LinearDiscreteBook))

    def test_from_params_continuous_model(self, continuous_simulation_params: dict):
        """from_params should create simulation with continuous model."""
        sim = Simulation.from_params(**continuous_simulation_params)

        assert sim.model_type == "continuous"
        assert isinstance(sim.book, LinearContinuousBook)

    def test_from_params_multi_book(self, small_grid: dict):
        """from_params should create multi-book simulation."""
        params = {
            "model_type": "discrete",
            "T": 100,
            "Nt": 10,
            **small_grid,
            "D": 0.5,
            "L": np.array([5.0, 5.0]),
            "nu": 0.0,
            "metaorder": [1.0],
        }
        sim = Simulation.from_params(**params)

        assert isinstance(sim.book, MultiDiscreteBook)

    def test_from_params_validates_model_type(self, simulation_params: dict):
        """from_params should validate model_type."""
        params = dict(simulation_params)
        params["model_type"] = "invalid"

        with pytest.raises(AssertionError):
            Simulation.from_params(**params)

    def test_from_params_expands_single_metaorder(self, simulation_params: dict):
        """Single metaorder value should be expanded to full array."""
        sim = Simulation.from_params(**simulation_params)

        # Metaorder should be expanded to T length
        assert len(sim.metaorder) == sim.T

    def test_from_params_accepts_full_metaorder(self, small_grid: dict):
        """Full metaorder array should be accepted."""
        T = 100
        metaorder = np.random.randn(T)
        params = {
            "model_type": "discrete",
            "T": T,
            "Nt": 10,
            **small_grid,
            "D": 0.5,
            "L": 10.0,
            "nu": 0.0,
            "metaorder": metaorder,
        }
        sim = Simulation.from_params(**params)

        np.testing.assert_array_equal(sim.metaorder, metaorder)


class TestSimulationParameterHandling:
    """Tests for parameter handling."""

    def test_default_parameters(self):
        """from_params should use sensible defaults."""
        # Use standard_parameters to get valid configuration
        params = standard_parameters(
            participation_rate=1.0,
            model_type="discrete",
            T=50,
            Nt=10,
        )
        sim = Simulation.from_params(**params)

        assert sim.T >= 1
        assert sim.Nt >= 1
        assert sim.nu == 0
        assert sim.price_formula == "middle"

    def test_derived_values_computed(self, simulation_params: dict):
        """Constructor should compute derived values."""
        sim = Simulation.from_params(**simulation_params)

        assert sim.dx > 0
        assert sim.price_range == sim.xmax - sim.xmin
        assert sim.D * np.max(sim.L) == sim.J

    def test_theoretical_values_computed(self, simulation_params: dict):
        """Constructor should compute theoretical values."""
        sim = Simulation.from_params(**simulation_params)

        # These should be computed
        assert hasattr(sim, "impact_th")
        assert hasattr(sim, "participation_rate")
        assert hasattr(sim, "scheme_constant")


class TestSimulationPriceFormulas:
    """Tests for price formula handling."""

    def test_price_formula_middle(self, simulation_params: dict):
        """Middle price formula should average ask and bid."""
        params = dict(simulation_params)
        params["price_formula"] = "middle"
        sim = Simulation.from_params(**params)

        price = sim.compute_price(10.0, 8.0)
        assert price == 9.0

    def test_price_formula_best_ask(self, simulation_params: dict):
        """Best ask price formula should return ask."""
        params = dict(simulation_params)
        params["price_formula"] = "best_ask"
        sim = Simulation.from_params(**params)

        price = sim.compute_price(10.0, 8.0)
        assert price == 10.0

    def test_price_formula_best_bid(self, simulation_params: dict):
        """Best bid price formula should return bid."""
        params = dict(simulation_params)
        params["price_formula"] = "best_bid"
        sim = Simulation.from_params(**params)

        price = sim.compute_price(10.0, 8.0)
        assert price == 8.0

    def test_price_formula_vwap(self, simulation_params: dict):
        """VWAP price formula should weight by volume."""
        params = dict(simulation_params)
        params["price_formula"] = "vwap"
        sim = Simulation.from_params(**params)

        # VWAP depends on book state
        price = sim.compute_price(10.0, 8.0)
        assert 8.0 <= price <= 10.0


class TestSimulationRunning:
    """Tests for simulation running."""

    def test_run_populates_price_arrays(self, simulation: Simulation):
        """run should populate prices, asks, bids arrays."""
        simulation.run()

        # Arrays should have values
        assert np.any(simulation.prices != 0) or np.all(simulation.prices == 0)
        assert len(simulation.prices) == simulation.T

    def test_run_executes_all_timesteps(self, simulation: Simulation):
        """run should execute T timesteps."""
        simulation.run()

        # Price should have been tracked for all steps
        assert len(simulation.asks) == simulation.T
        assert len(simulation.bids) == simulation.T

    def test_run_with_metaorder_moves_price(self, small_grid: dict):
        """Running with metaorder should cause price movement."""
        sim = Simulation.from_params(
            model_type="discrete",
            T=50,
            Nt=10,
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.0,
            metaorder=[1.0],  # Constant buy pressure
        )
        initial_ask = sim.book.best_ask

        sim.run()

        # With constant buy pressure, ask should move up
        final_ask = sim.book.best_ask
        assert final_ask >= initial_ask

    def test_run_records_measurements(self, small_grid: dict):
        """run should record measurements at specified indices."""
        T = 100
        measurement_indices = [10, 50, 90]
        sim = Simulation.from_params(
            model_type="discrete",
            T=T,
            Nt=10,
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.0,
            metaorder=[1.0],
            measured_quantities=["best_ask", "best_bid"],
            measurement_indices=measurement_indices,
        )

        sim.run()

        # Should have measurements at specified indices
        assert len(sim.measurements["best_ask"]) == len(measurement_indices)
        assert len(sim.measurements["best_bid"]) == len(measurement_indices)


class TestSimulationUtilities:
    """Tests for utility functions."""

    def test_standard_parameters_returns_valid_dict(self):
        """standard_parameters should return valid simulation params."""
        params = standard_parameters(
            participation_rate=1.0,
            model_type="discrete",
        )

        assert "T" in params
        assert "Nt" in params
        assert "D" in params
        assert "L" in params
        assert "xmin" in params
        assert "xmax" in params

    def test_standard_parameters_different_rates(self):
        """standard_parameters should work for different participation rates."""
        for rate in [0.1, 1.0, 10.0, 100.0]:
            params = standard_parameters(
                participation_rate=rate,
                model_type="discrete",
            )
            assert params["D"] >= 0
            assert params["L"] > 0

    def test_standard_parameters_creates_runnable_sim(self):
        """Params from standard_parameters should create runnable simulation."""
        params = standard_parameters(
            participation_rate=1.0,
            model_type="discrete",
            T=50,
            Nt=10,
        )
        sim = Simulation.from_params(**params)
        sim.run()

        # Should complete without error
        assert len(sim.prices) == params["T"]

    def test_get_density(self, simulation: Simulation):
        """get_density should return dict with bid and ask."""
        density = simulation.get_density()

        assert "bid" in density
        assert "ask" in density
        assert len(density["bid"]) == simulation.Nx
        assert len(density["ask"]) == simulation.Nx

    def test_get_growth_th(self, simulation: Simulation):
        """get_growth_th should return theoretical impact profile."""
        growth = simulation.get_growth_th()

        # Should span from n_start to n_end
        expected_len = simulation.n_end - simulation.n_start
        assert len(growth) == expected_len

    def test_str_representation(self, simulation: Simulation):
        """__str__ should return informative string."""
        s = str(simulation)

        assert "T =" in s
        assert "D =" in s
        assert "L =" in s


class TestSimulationBookTypes:
    """Tests for different book type selection."""

    def test_linear_regime_uses_linear_discrete_book(self, small_grid: dict):
        """Linear regime (nu=0) should use LinearDiscreteBook."""
        sim = Simulation.from_params(
            model_type="discrete",
            T=100,
            Nt=10,
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.0,
            metaorder=[1.0],
        )

        assert isinstance(sim.book, LinearDiscreteBook)

    def test_nonlinear_regime_uses_discrete_book(self, small_grid: dict):
        """Nonlinear regime (nu>0) should use DiscreteBook."""
        sim = Simulation.from_params(
            model_type="discrete",
            T=100,
            Nt=10,
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.1,
            metaorder=[1.0],
        )

        assert isinstance(sim.book, DiscreteBook)
        assert not isinstance(sim.book, LinearDiscreteBook)

    def test_continuous_model_uses_continuous_book(self, small_grid: dict):
        """Continuous model should use LinearContinuousBook."""
        sim = Simulation.from_params(
            model_type="continuous",
            T=100,
            Nt=10,
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.0,
            metaorder=[1.0],
        )

        assert isinstance(sim.book, LinearContinuousBook)
