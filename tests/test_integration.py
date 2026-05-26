"""
Integration tests for end-to-end LLOB workflows.
"""

import numpy as np

from llob import MonteCarlo, Simulation, courant_dt, standard_parameters


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


class TestSimulationIntegration:
    def test_discrete_simulation_runs_to_completion(self):
        sim = Simulation.from_params(**_params())
        sim.run()
        assert len(sim.prices) == sim.n_frames
        assert len(sim.asks) == sim.n_frames
        assert len(sim.bids) == sim.n_frames

    def test_continuous_simulation_runs_to_completion(self):
        sim = Simulation.from_params(**_params(model_type="continuous"))
        sim.run()
        assert len(sim.prices) == sim.n_frames

    def test_nonlinear_discrete_simulation(self, seed_random):
        sim = Simulation.from_params(**_params(duration=50.0, nu=0.1))
        sim.run()
        assert len(sim.prices) == sim.n_frames

    def test_multi_book_simulation(self, seed_random):
        sim = Simulation.from_params(**_params(duration=50.0, L=np.array([5.0, 5.0])))
        sim.run()
        assert len(sim.prices) == sim.n_frames


class TestMonteCarloIntegration:
    def test_monte_carlo_ensemble_runs(self, simulation_params: dict, seed_random):
        mc = MonteCarlo.from_params(
            N_samples=3,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.run()
        assert mc.price_samples.shape[1] == 3
        assert len(mc.price_mean) == mc.n_frames

    def test_monte_carlo_gather_results(self, simulation_params: dict, seed_random):
        mc = MonteCarlo.from_params(
            N_samples=2,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.run()
        results = mc.gather_results()
        assert "price_mean" in results and "price_variance" in results


class TestPhysicalProperties:
    def test_buy_orders_increase_price(self):
        params = standard_parameters(participation_rate=10.0, model_type="discrete",
                                     duration=200.0, n_frames=20)
        sim = Simulation.from_params(**params)
        sim.run()
        assert sim.prices[-1] > sim.prices[0]

    def test_sell_orders_decrease_price(self):
        params = standard_parameters(participation_rate=-10.0, model_type="discrete",
                                     duration=200.0, n_frames=20)
        sim = Simulation.from_params(**params)
        sim.run()
        assert sim.prices[-1] < sim.prices[0]

    def test_no_metaorder_stable_price(self):
        sim = Simulation.from_params(**_params(metaorder=[0.0]))
        sim.run()
        assert np.abs(np.mean(sim.prices)) < 1.0

    def test_simulation_completes_with_metaorder(self):
        params = standard_parameters(participation_rate=10.0, model_type="discrete",
                                     duration=200.0, n_frames=20)
        sim = Simulation.from_params(**params)
        sim.run()
        assert len(sim.prices) == sim.n_frames
        assert np.all(np.isfinite(sim.prices))

    def test_discrete_vs_continuous_qualitative_agreement(self):
        params = standard_parameters(participation_rate=10.0, model_type="discrete",
                                     duration=200.0, n_frames=20)
        common = {k: v for k, v in params.items() if k != "model_type"}

        sim_d = Simulation.from_params(model_type="discrete", **common)
        sim_c = Simulation.from_params(model_type="continuous", **common)
        sim_d.run()
        sim_c.run()

        assert sim_d.prices[-1] > sim_d.prices[0]
        assert sim_c.prices[-1] > sim_c.prices[0]

        d_impact = abs(sim_d.prices[-1] - sim_d.prices[0])
        c_impact = abs(sim_c.prices[-1] - sim_c.prices[0])
        if d_impact > 0 and c_impact > 0:
            ratio = max(d_impact, c_impact) / min(d_impact, c_impact)
            assert ratio < 20.0


class TestParameterSensitivity:
    def test_higher_L_reduces_impact(self):
        params = standard_parameters(participation_rate=10.0, model_type="discrete",
                                     duration=200.0, n_frames=20)
        sim_normal = Simulation.from_params(**params)
        sim_normal.run()
        impact_normal = abs(sim_normal.prices[-1] - sim_normal.prices[0])

        params_high_L = dict(params, L=params["L"] * 2)
        sim_high = Simulation.from_params(**params_high_L)
        sim_high.run()
        impact_high = abs(sim_high.prices[-1] - sim_high.prices[0])

        assert impact_high <= impact_normal + 0.1
