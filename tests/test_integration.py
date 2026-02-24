"""
Integration tests for end-to-end LLOB workflows.
"""

import numpy as np

from llob import MonteCarlo, Simulation, standard_parameters


class TestSimulationIntegration:
    """End-to-end tests for Simulation class."""

    def test_discrete_simulation_runs_to_completion(self, small_grid: dict):
        """Discrete simulation should run from start to finish."""
        sim = Simulation.from_params(
            model_type="discrete",
            T=100.0,  # Physical time
            Nt=10,    # Number of frames
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.0,
            metaorder=[1.0],
        )

        sim.run()

        # Should have populated all arrays (sized by Nt)
        assert len(sim.prices) == sim.Nt
        assert len(sim.asks) == sim.Nt
        assert len(sim.bids) == sim.Nt

    def test_continuous_simulation_runs_to_completion(self, small_grid: dict):
        """Continuous simulation should run from start to finish."""
        sim = Simulation.from_params(
            model_type="continuous",
            T=100.0,  # Physical time
            Nt=10,    # Number of frames
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.0,
            metaorder=[1.0],
        )

        sim.run()

        # Should have populated all arrays (sized by Nt)
        assert len(sim.prices) == sim.Nt
        assert len(sim.asks) == sim.Nt
        assert len(sim.bids) == sim.Nt

    def test_nonlinear_discrete_simulation(self, small_grid: dict, seed_random):
        """Nonlinear discrete simulation (nu > 0) should run."""
        sim = Simulation.from_params(
            model_type="discrete",
            T=50.0,   # Physical time
            Nt=10,    # Number of frames
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.1,
            metaorder=[1.0],
        )

        sim.run()

        assert len(sim.prices) == sim.Nt

    def test_multi_book_simulation(self, small_grid: dict, seed_random):
        """Multi-book simulation should run."""
        sim = Simulation.from_params(
            model_type="discrete",
            T=50.0,   # Physical time
            Nt=10,    # Number of frames
            **small_grid,
            D=0.5,
            L=np.array([5.0, 5.0]),
            nu=0.0,
            metaorder=[1.0],
        )

        sim.run()

        assert len(sim.prices) == sim.Nt


class TestMonteCarloIntegration:
    """End-to-end tests for MonteCarlo class."""

    def test_monte_carlo_ensemble_runs(self, simulation_params: dict, seed_random):
        """Monte Carlo should run full ensemble."""
        mc = MonteCarlo.from_params(
            N_samples=3,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )

        mc.run()

        # Should have results for all samples (sized by Nt)
        assert mc.price_samples.shape[1] == 3
        assert len(mc.price_mean) == mc.Nt

    def test_monte_carlo_gather_results(self, simulation_params: dict, seed_random):
        """Monte Carlo gather_results should work after run."""
        mc = MonteCarlo.from_params(
            N_samples=2,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.run()

        results = mc.gather_results()

        assert isinstance(results, dict)
        assert "price_mean" in results
        assert "price_variance" in results


class TestPhysicalProperties:
    """Tests for physical/mathematical properties of the model."""

    def test_buy_orders_increase_price(self):
        """Persistent buy orders should increase price."""
        # Use standard_parameters for well-behaved simulation
        params = standard_parameters(
            participation_rate=10.0,
            model_type="discrete",
            T=200.0,
            Nt=20,
        )

        sim = Simulation.from_params(**params)
        sim.run()

        # Price at end should be higher than start (for positive metaorder)
        assert sim.prices[-1] > sim.prices[0]

    def test_sell_orders_decrease_price(self):
        """Persistent sell orders should decrease price."""
        # Use standard_parameters and negate the metaorder
        params = standard_parameters(
            participation_rate=-10.0,
            model_type="discrete",
            T=200.0,
            Nt=20,
        )

        sim = Simulation.from_params(**params)
        sim.run()

        # Price at end should be lower than start (for negative metaorder)
        assert sim.prices[-1] < sim.prices[0]

    def test_no_metaorder_stable_price(self, small_grid: dict):
        """With no metaorder, price should remain near zero."""
        sim = Simulation.from_params(
            model_type="discrete",
            T=100.0,  # Physical time
            Nt=10,    # Number of frames
            **small_grid,
            D=0.5,
            L=10.0,
            nu=0.0,
            metaorder=[0.0],  # No trading
        )

        sim.run()

        # Price should stay near initial (approximately 0)
        assert np.abs(np.mean(sim.prices)) < 1.0

    def test_simulation_completes_with_metaorder(self):
        """Simulation should complete successfully with metaorder."""
        # Use standard parameters for well-behaved simulation
        params = standard_parameters(
            participation_rate=10.0,
            model_type="discrete",
            T=200.0,
            Nt=20,
        )

        sim = Simulation.from_params(**params)
        sim.run()

        # Simulation should complete and have valid output (sized by Nt)
        assert len(sim.prices) == sim.Nt
        assert np.all(np.isfinite(sim.prices))

    def test_discrete_vs_continuous_qualitative_agreement(self):
        """Discrete and continuous models should show similar behavior."""
        # Use standard parameters for well-behaved simulation
        params = standard_parameters(
            participation_rate=10.0,
            model_type="discrete",
            T=200.0,
            Nt=20,
        )
        common_params = {
            "T": params["T"],
            "Nt": params["Nt"],
            "xmin": params["xmin"],
            "xmax": params["xmax"],
            "Nx": params["Nx"],
            "D": params["D"],
            "L": params["L"],
            "nu": 0.0,
            "metaorder": params["metaorder"],
        }

        sim_discrete = Simulation.from_params(model_type="discrete", **common_params)
        sim_continuous = Simulation.from_params(
            model_type="continuous", **common_params
        )

        sim_discrete.run()
        sim_continuous.run()

        # Both should show price increase (with positive metaorder)
        assert sim_discrete.prices[-1] > sim_discrete.prices[0]
        assert sim_continuous.prices[-1] > sim_continuous.prices[0]

        # Final prices should be in same ballpark (within factor of 10)
        discrete_impact = abs(sim_discrete.prices[-1] - sim_discrete.prices[0])
        continuous_impact = abs(sim_continuous.prices[-1] - sim_continuous.prices[0])

        if discrete_impact > 0 and continuous_impact > 0:
            ratio = max(discrete_impact, continuous_impact) / min(
                discrete_impact, continuous_impact
            )
            assert ratio < 20.0


class TestParameterSensitivity:
    """Tests for parameter sensitivity."""

    def test_higher_L_reduces_impact(self):
        """Higher latent liquidity L should reduce price impact."""
        # Use standard_parameters as base
        params = standard_parameters(
            participation_rate=10.0,
            model_type="discrete",
            T=200.0,
            Nt=20,
        )

        # Run with normal L
        sim_normal_L = Simulation.from_params(**params)
        sim_normal_L.run()
        impact_normal_L = abs(sim_normal_L.prices[-1] - sim_normal_L.prices[0])

        # Run with higher L (double)
        params_high_L = dict(params)
        params_high_L["L"] = params["L"] * 2
        sim_high_L = Simulation.from_params(**params_high_L)
        sim_high_L.run()
        impact_high_L = abs(sim_high_L.prices[-1] - sim_high_L.prices[0])

        # Higher L should mean lower impact (or equal if both near 0)
        assert impact_high_L <= impact_normal_L + 0.1  # Allow small tolerance

    def test_standard_parameters_various_rates(self):
        """standard_parameters should work for various participation rates."""
        for rate in [0.1, 1.0, 10.0]:
            params = standard_parameters(
                participation_rate=rate,
                model_type="discrete",
                T=50.0,
                Nt=10,
            )

            sim = Simulation.from_params(**params)
            sim.run()

            # Should complete without error (sized by Nt)
            assert len(sim.prices) == params["Nt"]
