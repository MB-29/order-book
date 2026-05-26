"""
Tests for the alpha (spread-dependent deposition) parameter.
"""

import numpy as np

from llob import DiscreteBook, LimitOrders, Simulation, courant_dt

DT = 0.01


def _make_orders(params: dict, alpha: float | None = None) -> LimitOrders:
    kwargs = dict(
        side="ask",
        lambd=params["lambd"],
        nu=params["nu"],
        D=params["D"],
        xmin=params["xmin"],
        xmax=params["xmax"],
        n_grid=params["n_grid"],
        L=params["L"],
        initial_density="stationary",
        boundary_conditions="flat",
    )
    if alpha is not None:
        kwargs["alpha"] = alpha
    return LimitOrders.from_params(**kwargs)


def _sim_params(**overrides) -> dict:
    base = {
        "model_type": "discrete",
        "duration": 50.0,
        "n_frames": 10,
        "xmin": -20.0,
        "xmax": 20.0,
        "n_grid": 40,
        "D": 0.5,
        "L": 10.0,
        "nu": 0.1,
        "metaorder": [1.0],
        "alpha": 0.1,
    }
    base["dt_step"] = courant_dt((base["xmax"] - base["xmin"]) / base["n_grid"], base["D"])
    base.update(overrides)
    return base


class TestLimitOrdersAlpha:
    def test_alpha_zero_backwards_compatible(self, nonlinear_params: dict, seed_random):
        np.random.seed(42)
        orders1 = _make_orders(nonlinear_params, alpha=0.0)
        orders1.deposition(DT, spread=5)
        vol_after1 = orders1.volumes.copy()

        np.random.seed(42)
        orders2 = _make_orders(nonlinear_params)  # alpha default
        orders2.deposition(DT, spread=5)
        np.testing.assert_array_equal(vol_after1, orders2.volumes)

    def test_alpha_positive_zero_spread(self, nonlinear_params: dict, seed_random):
        np.random.seed(42)
        orders_no_alpha = _make_orders(nonlinear_params, alpha=0.0)
        orders_no_alpha.deposition(DT, spread=0)

        np.random.seed(42)
        orders_with_alpha = _make_orders(nonlinear_params, alpha=1.0)
        orders_with_alpha.deposition(DT, spread=0)

        np.testing.assert_array_equal(orders_no_alpha.volumes, orders_with_alpha.volumes)

    def test_alpha_positive_increases_deposition(self, nonlinear_params: dict, seed_random):
        np.random.seed(42)
        orders_low = _make_orders(nonlinear_params, alpha=0.1)
        initial = orders_low.volumes.sum()
        for _ in range(100):
            orders_low.deposition(DT, spread=10)
        vol_low = orders_low.volumes.sum() - initial

        np.random.seed(42)
        orders_high = _make_orders(nonlinear_params, alpha=1.0)
        initial = orders_high.volumes.sum()
        for _ in range(100):
            orders_high.deposition(DT, spread=10)
        vol_high = orders_high.volumes.sum() - initial

        assert vol_high > vol_low

    def test_alpha_stored_correctly(self, nonlinear_params: dict):
        orders = _make_orders(nonlinear_params, alpha=0.42)
        assert orders.alpha == 0.42


class TestDiscreteBookAlpha:
    def test_from_params_passes_alpha_to_orders(self, nonlinear_params: dict):
        book = DiscreteBook.from_params(
            D=nonlinear_params["D"],
            xmin=nonlinear_params["xmin"],
            xmax=nonlinear_params["xmax"],
            n_grid=nonlinear_params["n_grid"],
            L=nonlinear_params["L"],
            nu=nonlinear_params["nu"],
            lambd=nonlinear_params["lambd"],
            alpha=0.25,
        )
        assert book.ask_orders.alpha == 0.25
        assert book.bid_orders.alpha == 0.25

    def test_from_params_alpha_default_zero(self, nonlinear_params: dict):
        book = DiscreteBook.from_params(
            D=nonlinear_params["D"],
            xmin=nonlinear_params["xmin"],
            xmax=nonlinear_params["xmax"],
            n_grid=nonlinear_params["n_grid"],
            L=nonlinear_params["L"],
            nu=nonlinear_params["nu"],
            lambd=nonlinear_params["lambd"],
        )
        assert book.ask_orders.alpha == 0.0
        assert book.bid_orders.alpha == 0.0


class TestSimulationAlpha:
    def test_full_simulation_with_alpha(self, seed_random):
        sim = Simulation.from_params(**_sim_params(alpha=0.1))
        sim.run()
        assert len(sim.prices) == sim.n_frames
        assert np.all(np.isfinite(sim.prices))
        assert np.all(np.isfinite(sim.spreads))

    def test_simulation_spreads_tracking(self, seed_random):
        sim = Simulation.from_params(**_sim_params(alpha=0.0))
        sim.run()
        assert len(sim.spreads) == sim.n_frames
        assert np.all(sim.spreads >= 0)

    def test_simulation_seed_reproducibility(self):
        params = _sim_params(alpha=0.1)
        sim1 = Simulation.from_params(**params, seed=42)
        sim1.run()
        sim2 = Simulation.from_params(**params, seed=42)
        sim2.run()
        np.testing.assert_array_equal(sim1.prices, sim2.prices)
        np.testing.assert_array_equal(sim1.spreads, sim2.spreads)


class TestSpreadStabilization:
    def test_alpha_runs_without_error(self, seed_random):
        common = _sim_params(
            duration=200.0, n_frames=50,
            xmin=-50.0, xmax=50.0, n_grid=100,
            metaorder=[0.0],
        )

        np.random.seed(123)
        sim_no_alpha = Simulation.from_params(**{**common, "alpha": 0.0})
        sim_no_alpha.run()

        np.random.seed(123)
        sim_with_alpha = Simulation.from_params(**{**common, "alpha": 0.5})
        sim_with_alpha.run()

        assert np.all(np.isfinite(sim_no_alpha.spreads))
        assert np.all(np.isfinite(sim_with_alpha.spreads))
