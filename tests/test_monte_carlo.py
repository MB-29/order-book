"""
Tests for MonteCarlo class.
"""

import numpy as np

from llob import MonteCarlo


class TestMonteCarloConstruction:
    """Tests for MonteCarlo construction."""

    def test_from_params_creates_instance(self, simulation_params: dict):
        """from_params should create a valid MonteCarlo instance."""
        mc = MonteCarlo.from_params(
            N_samples=5,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )

        assert mc.N_samples == 5
        assert mc.hurst == 0.75
        assert mc.m0 == 1.0
        assert mc.m1 == 0.1

    def test_from_params_extracts_time_params(self, simulation_params: dict):
        """from_params should extract T and Nt from simulation_args."""
        mc = MonteCarlo.from_params(
            N_samples=5,
            noise_args={"hurst": 0.75},
            simulation_args=simulation_params,
        )

        assert simulation_params["duration"] == mc.duration
        assert mc.n_frames == simulation_params["n_frames"]

    def test_from_params_defaults(self, simulation_params: dict):
        """from_params should use default m0=0, m1=0 when not specified."""
        mc = MonteCarlo.from_params(
            N_samples=5,
            noise_args={"hurst": 0.75},
            simulation_args=simulation_params,
        )

        assert mc.m0 == 0.0
        assert mc.m1 == 0.0

    def test_output_array_shapes(self, simulation_params: dict):
        """Output arrays should have correct shapes."""
        N_samples = 5
        Nt = simulation_params["n_frames"]  # Arrays sized by Nt (frames), not T (time)

        mc = MonteCarlo.from_params(
            N_samples=N_samples,
            noise_args={"hurst": 0.75},
            simulation_args=simulation_params,
        )

        assert mc.price_samples.shape == (Nt, N_samples)
        assert mc.ask_samples.shape == (Nt, N_samples)
        assert mc.bid_samples.shape == (Nt, N_samples)
        assert mc.noise.shape == (Nt, N_samples)


class TestMonteCarloNoiseGeneration:
    """Tests for noise generation."""

    def test_generate_noise_shape(self, simulation_params: dict, seed_random):
        """generate_noise should create arrays of correct shape."""
        N_samples = 5
        Nt = simulation_params["n_frames"]  # Arrays sized by Nt (frames), not T (time)

        mc = MonteCarlo.from_params(
            N_samples=N_samples,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.5},
            simulation_args=simulation_params,
        )
        mc.generate_noise()

        assert mc.noisy_metaorders.shape == (Nt, N_samples)

    def test_generate_noise_mean_approx_m0(self, simulation_params: dict, seed_random):
        """Generated noise should have mean approximately m0."""
        mc = MonteCarlo.from_params(
            N_samples=20,
            noise_args={"hurst": 0.75, "m0": 5.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.generate_noise()

        mean = np.mean(mc.noisy_metaorders)
        assert np.isclose(mean, 5.0, atol=1.0)  # Within 1.0 of m0

    def test_generate_noise_zero_m1_deterministic(self, simulation_params: dict, seed_random):
        """With m1=0, noise should be deterministic (all m0)."""
        mc = MonteCarlo.from_params(
            N_samples=5,
            noise_args={"hurst": 0.75, "m0": 3.0, "m1": 0.0},
            simulation_args=simulation_params,
        )
        mc.generate_noise()

        # All values should be exactly m0
        np.testing.assert_array_almost_equal(
            mc.noisy_metaorders, np.full_like(mc.noisy_metaorders, 3.0)
        )


class TestMonteCarloRunning:
    """Tests for running Monte Carlo simulations."""

    def test_run_populates_samples(self, simulation_params: dict, seed_random):
        """run should populate sample arrays."""
        mc = MonteCarlo.from_params(
            N_samples=3,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.run()

        # Samples should have been populated
        assert not np.all(mc.price_samples == 0)

    def test_run_computes_statistics(self, simulation_params: dict, seed_random):
        """run should compute mean and variance statistics."""
        mc = MonteCarlo.from_params(
            N_samples=3,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.run()

        # Statistics should be computed (sized by Nt)
        assert len(mc.price_mean) == mc.n_frames
        assert len(mc.price_variance) == mc.n_frames

    def test_run_sets_simulation_reference(self, simulation_params: dict, seed_random):
        """run should set simulation reference."""
        mc = MonteCarlo.from_params(
            N_samples=2,
            noise_args={"hurst": 0.75},
            simulation_args=simulation_params,
        )
        mc.run()

        assert mc.simulation is not None


class TestMonteCarloStatistics:
    """Tests for statistics computation."""

    def test_compute_statistics_mean(self, simulation_params: dict, seed_random):
        """compute_statistics should calculate mean correctly."""
        mc = MonteCarlo.from_params(
            N_samples=3,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.run()

        # Mean should match samples
        expected_mean = mc.price_samples.mean(axis=1)
        np.testing.assert_array_almost_equal(mc.price_mean, expected_mean)

    def test_compute_statistics_variance(self, simulation_params: dict, seed_random):
        """compute_statistics should calculate variance correctly."""
        mc = MonteCarlo.from_params(
            N_samples=3,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.5},
            simulation_args=simulation_params,
        )
        mc.run()

        # Variance should match samples
        expected_var = mc.price_samples.var(axis=1)
        np.testing.assert_array_almost_equal(mc.price_variance, expected_var)

    def test_gather_results_contains_all_keys(self, simulation_params: dict, seed_random):
        """gather_results should return dict with all expected keys."""
        mc = MonteCarlo.from_params(
            N_samples=2,
            noise_args={"hurst": 0.75, "m0": 1.0, "m1": 0.1},
            simulation_args=simulation_params,
        )
        mc.run()
        results = mc.gather_results()

        # Check required keys
        assert "price_mean" in results
        assert "price_variance" in results
        assert "ask_mean" in results
        assert "ask_variance" in results
        assert "bid_mean" in results
        assert "bid_variance" in results
        assert "m0" in results
        assert "m1" in results
        assert "N_samples" in results
        assert "hurst" in results
        assert "params" in results

    def test_gather_results_values_correct(self, simulation_params: dict, seed_random):
        """gather_results values should match instance attributes."""
        mc = MonteCarlo.from_params(
            N_samples=2,
            noise_args={"hurst": 0.75, "m0": 2.0, "m1": 0.3},
            simulation_args=simulation_params,
        )
        mc.run()
        results = mc.gather_results()

        np.testing.assert_array_equal(results["price_mean"], mc.price_mean)
        np.testing.assert_array_equal(results["price_variance"], mc.price_variance)
        assert results["m0"] == mc.m0
        assert results["m1"] == mc.m1
        assert results["N_samples"] == mc.N_samples
        assert results["hurst"] == mc.hurst


class TestMonteCarloMeasurements:
    """Tests for measurement handling."""

    def test_measured_quantities_recorded(self, small_grid: dict, seed_random):
        """Measured quantities should be recorded for each sample."""
        params = {
            "model_type": "discrete",
            "duration": 50.0,  # Physical time
            "n_frames": 10,  # Number of frames
            **small_grid,
            "D": 0.5,
            "L": 10.0,
            "nu": 0.0,
            "metaorder": [1.0],
            "measured_quantities": ["best_ask", "best_bid"],
            "measurement_indices": [2, 8],  # Within Nt range
        }
        mc = MonteCarlo.from_params(
            N_samples=2,
            noise_args={"hurst": 0.75},
            simulation_args=params,
        )
        mc.run()

        # Should have recorded measurements
        assert "best_ask" in mc.measured_samples
        assert "best_bid" in mc.measured_samples
        assert len(mc.measured_samples["best_ask"]) == 2  # N_samples
