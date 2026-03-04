"""Tests for pydantic configuration models."""

import numpy as np
import pytest
from pydantic import ValidationError

from llob import (
    GridConfig,
    MonteCarloConfig,
    NoiseConfig,
    Simulation,
    SimulationConfig,
)
from llob.configs import (
    DiscreteBookConfig,
    LimitOrdersConfig,
    LinearContinuousBookConfig,
    LinearDiscreteBookConfig,
    MultiDiscreteBookConfig,
)


class TestGridConfig:
    """Tests for GridConfig."""

    def test_basic_creation(self):
        """Test basic grid config creation."""
        grid = GridConfig(xmin=-10.0, xmax=10.0, n_grid=20)
        assert grid.xmin == -10.0
        assert grid.xmax == 10.0
        assert grid.n_grid == 20

    def test_computed_properties(self):
        """Test computed grid properties."""
        grid = GridConfig(xmin=-10.0, xmax=10.0, n_grid=20)
        assert grid.price_range == 20.0
        assert grid.dx == 1.0

    def test_validation_xmin_xmax(self):
        """Test that xmin must be less than xmax."""
        with pytest.raises(ValueError, match="xmin.*must be less than xmax"):
            GridConfig(xmin=10.0, xmax=-10.0, n_grid=20)

    def test_validation_nx_positive(self):
        """Test that Nx must be positive."""
        with pytest.raises(ValueError):
            GridConfig(xmin=-10.0, xmax=10.0, n_grid=0)

    def test_immutability(self):
        """Test that config is frozen."""
        grid = GridConfig(xmin=-10.0, xmax=10.0, n_grid=20)
        with pytest.raises(ValidationError):
            grid.xmin = -20.0


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_basic_creation(self):
        """Test basic simulation config creation."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = SimulationConfig(
            model_type="discrete",
            grid=grid,
            D=1.0,
            L=1.0,
            duration=100.0,  # Physical time
            n_frames=100,  # Number of frames
        )
        assert config.model_type == "discrete"
        assert config.D == 1.0
        assert config.L == 1.0

    def test_metaorder_expansion(self):
        """Test that single-value metaorder is expanded to Nt length."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = SimulationConfig(
            model_type="discrete",
            grid=grid,
            D=1.0,
            L=1.0,
            duration=100.0,  # Physical time
            n_frames=100,  # Number of frames
            metaorder=[0.5],
            frame_start=10,
            frame_end=90,
        )
        full = config.get_full_metaorder()
        assert len(full) == 100  # Length is Nt
        assert full[0] == 0.0
        assert full[10] == 0.5
        assert full[89] == 0.5
        assert full[90] == 0.0

    def test_multi_book_config(self):
        """Test multi-actor configuration."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = SimulationConfig(
            model_type="discrete",
            grid=grid,
            D=1.0,
            L=[1.0, 2.0],
            nu=[0.1, 0.2],
            duration=100.0,  # Physical time
            n_frames=100,  # Number of frames
        )
        assert config.is_multi_book
        assert config.L_max == 2.0

    def test_from_standard_parameters(self):
        """Test convenience constructor."""
        config = SimulationConfig.from_standard_parameters(
            participation_rate=1.0,
            model_type="discrete",
        )
        assert config.model_type == "discrete"
        assert config.D == 1.0
        assert config.L == 1.0


class TestNoiseConfig:
    """Tests for NoiseConfig."""

    def test_basic_creation(self):
        """Test basic noise config creation."""
        noise = NoiseConfig(m0=0.5, m1=0.1, hurst=0.7)
        assert noise.m0 == 0.5
        assert noise.m1 == 0.1
        assert noise.hurst == 0.7

    def test_deterministic_check(self):
        """Test deterministic detection."""
        noise_det = NoiseConfig(m0=0.5, m1=0.0, hurst=0.5)
        noise_stoch = NoiseConfig(m0=0.5, m1=0.1, hurst=0.5)
        assert noise_det.is_deterministic
        assert not noise_stoch.is_deterministic

    def test_hurst_bounds(self):
        """Test Hurst exponent bounds."""
        with pytest.raises(ValueError):
            NoiseConfig(m0=0.5, m1=0.1, hurst=-0.1)
        with pytest.raises(ValueError):
            NoiseConfig(m0=0.5, m1=0.1, hurst=1.5)


class TestMonteCarloConfig:
    """Tests for MonteCarloConfig."""

    def test_basic_creation(self):
        """Test basic Monte Carlo config creation."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        sim_config = SimulationConfig(
            model_type="discrete",
            grid=grid,
            D=1.0,
            L=1.0,
            duration=100.0,  # Physical time
            n_frames=100,  # Number of frames
        )
        noise_config = NoiseConfig(m0=0.5, m1=0.1, hurst=0.7)
        mc_config = MonteCarloConfig(
            N_samples=10,
            noise=noise_config,
            simulation=sim_config,
        )
        assert mc_config.N_samples == 10
        assert mc_config.duration == 100.0
        assert mc_config.n_frames == 100

    def test_to_simulation_args(self):
        """Test conversion to legacy simulation args."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        sim_config = SimulationConfig(
            model_type="discrete",
            grid=grid,
            D=1.0,
            L=1.0,
            duration=100.0,  # Physical time
            n_frames=100,  # Number of frames
        )
        noise_config = NoiseConfig(m0=0.5, m1=0.1, hurst=0.7)
        mc_config = MonteCarloConfig(
            N_samples=10,
            noise=noise_config,
            simulation=sim_config,
        )
        args = mc_config.to_simulation_args()
        assert args["model_type"] == "discrete"
        assert args["duration"] == 100.0
        assert args["D"] == 1.0


class TestBookConfigs:
    """Tests for book configuration models."""

    def test_limit_orders_config(self):
        """Test LimitOrdersConfig."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = LimitOrdersConfig(
            grid=grid,
            side="ask",
            D=1.0,
            L=1.0,
            nu=0.1,
            lambd=0.5,
            boundary_conditions="linear",
        )
        assert config.side == "ask"
        assert config.boundary_flow == 1.0  # linear boundary

    def test_discrete_book_config(self):
        """Test DiscreteBookConfig."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = DiscreteBookConfig(
            grid=grid,
            D=1.0,
            nu=0.1,
            lambd=0.5,
        )
        L = config.compute_L()
        assert pytest.approx(0.5 / np.sqrt(0.1 * 1.0)) == L

    def test_linear_discrete_book_config(self):
        """Test LinearDiscreteBookConfig."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = LinearDiscreteBookConfig(
            grid=grid,
            D=1.0,
            L=1.0,
        )
        assert config.dt == pytest.approx(grid.dx**2 / (2 * 1.0))

    def test_linear_continuous_book_config(self):
        """Test LinearContinuousBookConfig."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = LinearContinuousBookConfig(
            grid=grid,
            D=1.0,
            L=1.0,
        )
        assert config.J == 1.0

    def test_multi_discrete_book_config(self):
        """Test MultiDiscreteBookConfig."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = MultiDiscreteBookConfig(
            grid=grid,
            D=1.0,
            L_list=[1.0, 2.0],
            nu_list=[0.1, 0.2],
            lambd_list=[0.5, 0.6],
        )
        assert config.n_actors == 2

    def test_multi_book_list_length_validation(self):
        """Test that list lengths must match."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        with pytest.raises(ValueError, match="nu_list length"):
            MultiDiscreteBookConfig(
                grid=grid,
                D=1.0,
                L_list=[1.0, 2.0],
                nu_list=[0.1],  # Wrong length
                lambd_list=[0.5, 0.6],
            )


class TestSimulationFromConfig:
    """Tests for Simulation.from_config."""

    def test_from_config_discrete(self):
        """Test creating simulation from config."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = SimulationConfig(
            model_type="discrete",
            grid=grid,
            D=1.0,
            L=1.0,
            duration=100.0,  # Physical time
            n_frames=100,  # Number of frames
            metaorder=[0.5],
        )
        sim = Simulation.from_config(config)
        assert sim.model_type == "discrete"
        assert sim.D == 1.0
        assert sim.L == 1.0
        assert sim.duration == 100.0

    def test_from_config_continuous(self):
        """Test creating continuous simulation from config."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = SimulationConfig(
            model_type="continuous",
            grid=grid,
            D=1.0,
            L=1.0,
            duration=100.0,  # Physical time
            n_frames=100,  # Number of frames
        )
        sim = Simulation.from_config(config)
        assert sim.model_type == "continuous"

    def test_from_config_runs(self):
        """Test that simulation from config actually runs."""
        grid = GridConfig(xmin=-50.0, xmax=50.0, n_grid=100)
        config = SimulationConfig(
            model_type="discrete",
            grid=grid,
            D=1.0,
            L=1.0,
            duration=50.0,  # Physical time
            n_frames=50,  # Number of frames
            metaorder=[0.5],
        )
        sim = Simulation.from_config(config)
        sim.run()
        # Verify simulation produced valid output
        assert np.all(np.isfinite(sim.prices))
        assert len(sim.prices) == sim.n_frames
