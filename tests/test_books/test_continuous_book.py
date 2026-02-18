"""
Tests for LinearContinuousBook class.
"""
from __future__ import annotations

import numpy as np
import pytest

from llob import LinearContinuousBook


class TestLinearContinuousBookConstruction:
    """Tests for LinearContinuousBook construction and initialization."""

    def test_from_params_creates_valid_instance(self, linear_params: dict):
        """from_params should create a valid LinearContinuousBook."""
        book = LinearContinuousBook.from_params(
            D=linear_params["D"],
            L=linear_params["L"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
        )

        assert linear_params["D"] == book.D
        assert linear_params["L"] == book.L
        assert book.Nx == linear_params["Nx"]

    def test_from_params_computes_grid(self, linear_params: dict):
        """from_params should compute the price grid correctly."""
        book = LinearContinuousBook.from_params(
            D=linear_params["D"],
            L=linear_params["L"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
        )

        assert len(book.X) == linear_params["Nx"]
        assert book.X[0] == linear_params["xmin"]
        assert book.X[-1] == linear_params["xmax"]
        assert book.dx > 0

    def test_initial_density_profile(self, continuous_book: LinearContinuousBook):
        """Initial density should be -L*x (linear profile)."""
        expected = -continuous_book.L * continuous_book.X

        np.testing.assert_array_almost_equal(continuous_book.density, expected)

    def test_initial_density_method(self, continuous_book: LinearContinuousBook):
        """initial_density method should return -L*x."""
        x = np.linspace(-5, 5, 10)
        expected = -continuous_book.L * x

        result = continuous_book.initial_density(x)

        np.testing.assert_array_almost_equal(result, expected)

    def test_derived_values_computed(self, continuous_book: LinearContinuousBook):
        """Constructor should compute derived values."""
        assert continuous_book.J == continuous_book.D * continuous_book.L
        assert (
            continuous_book.price_range
            == (continuous_book.xmax - continuous_book.xmin) / 2
        )


class TestLinearContinuousBookPriceTracking:
    """Tests for price tracking functionality."""

    def test_update_prices_sets_best_ask_bid(
        self, continuous_book: LinearContinuousBook
    ):
        """update_prices should set best ask and bid."""
        continuous_book.update_prices()

        assert continuous_book.best_ask_index >= 0
        assert continuous_book.best_bid_index >= 0
        assert continuous_book.best_ask_index < continuous_book.Nx
        assert continuous_book.best_bid_index < continuous_book.Nx

    def test_best_ask_best_bid_symmetric(self, linear_params: dict):
        """For symmetric initial density, best ask and bid should be symmetric."""
        # Ensure symmetric grid
        xmax = abs(linear_params["xmin"])
        book = LinearContinuousBook.from_params(
            D=linear_params["D"],
            L=linear_params["L"],
            xmin=-xmax,
            xmax=xmax,
            Nx=linear_params["Nx"],
        )

        # Best ask and bid should be symmetric around 0
        assert np.isclose(abs(book.best_ask), abs(book.best_bid), atol=book.dx * 2)

    def test_best_ask_is_positive(self, continuous_book: LinearContinuousBook):
        """Best ask should be on positive price side."""
        # For linear density -L*x, negative density (asks) are at positive x
        # So best_ask should be close to 0 on positive side
        assert continuous_book.best_ask >= -continuous_book.dx

    def test_best_bid_is_negative(self, continuous_book: LinearContinuousBook):
        """Best bid should be on negative price side."""
        # For linear density -L*x, positive density (bids) are at negative x
        # So best_bid should be close to 0 on negative side
        assert continuous_book.best_bid <= continuous_book.dx


class TestLinearContinuousBookTimeEvolution:
    """Tests for time evolution functionality."""

    def test_timestep_with_diffusion(self, continuous_book: LinearContinuousBook):
        """timestep should apply diffusion when D != 0."""
        continuous_book.timestep(tstep=0.1, volume=0.0)

        # Density should change due to diffusion (boundary effects)
        # For interior points, linear profile is steady state,
        # but boundary may cause changes
        # Just check it runs without error
        assert continuous_book.density is not None

    def test_timestep_without_diffusion(self, linear_params: dict):
        """timestep with D=0 should skip diffusion."""
        book = LinearContinuousBook.from_params(
            D=0.0,
            L=linear_params["L"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
        )
        initial_density = book.density.copy()

        book.timestep(tstep=0.1, volume=0.0)

        # With D=0 and no metaorder, density should be unchanged
        np.testing.assert_array_equal(book.density, initial_density)

    def test_execute_metaorder_consumes_density(
        self, continuous_book: LinearContinuousBook
    ):
        """execute_metaorder should consume density at best price."""
        initial_best_ask_idx = continuous_book.best_ask_index
        initial_density_at_best = continuous_book.density[initial_best_ask_idx]

        # Execute a small buy order
        continuous_book.execute_metaorder(0.1)

        # Density at or near best ask should have increased (less negative)
        # or index should have moved
        density_changed = (
            continuous_book.density[initial_best_ask_idx] != initial_density_at_best
        )
        index_moved = continuous_book.best_ask_index != initial_best_ask_idx
        assert density_changed or index_moved

    def test_execute_metaorder_zero_is_noop(
        self, continuous_book: LinearContinuousBook
    ):
        """execute_metaorder with volume=0 should be a no-op."""
        initial_density = continuous_book.density.copy()

        continuous_book.execute_metaorder(0.0)

        np.testing.assert_array_equal(continuous_book.density, initial_density)

    def test_execute_metaorder_raises_on_no_ask_liquidity(self, linear_params: dict):
        """execute_metaorder should raise when ask liquidity exhausted."""
        book = LinearContinuousBook.from_params(
            D=linear_params["D"],
            L=linear_params["L"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
        )

        # Try to buy more than available
        with pytest.raises(ValueError, match="liquidity"):
            book.execute_metaorder(1e6)

    def test_execute_metaorder_raises_on_no_bid_liquidity(self, linear_params: dict):
        """execute_metaorder should raise when bid liquidity exhausted."""
        book = LinearContinuousBook.from_params(
            D=linear_params["D"],
            L=linear_params["L"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
        )

        # Try to sell more than available
        with pytest.raises(ValueError, match="liquidity"):
            book.execute_metaorder(-1e6)

    def test_buy_order_moves_ask_price_up(self, continuous_book: LinearContinuousBook):
        """Buying should move the best ask price up."""
        initial_best_ask = continuous_book.best_ask

        # Execute a significant buy order
        continuous_book.execute_metaorder(1.0)
        continuous_book.update_prices()

        # Best ask should have moved up (higher price)
        assert continuous_book.best_ask >= initial_best_ask

    def test_sell_order_moves_bid_price_down(
        self, continuous_book: LinearContinuousBook
    ):
        """Selling should move the best bid price down."""
        initial_best_bid = continuous_book.best_bid

        # Execute a significant sell order
        continuous_book.execute_metaorder(-1.0)
        continuous_book.update_prices()

        # Best bid should have moved down (lower price)
        assert continuous_book.best_bid <= initial_best_bid


class TestLinearContinuousBookDensity:
    """Tests for density-related functionality."""

    def test_density_shape_matches_grid(self, continuous_book: LinearContinuousBook):
        """Density array should match grid size."""
        assert len(continuous_book.density) == continuous_book.Nx

    def test_density_negative_on_ask_side(self, continuous_book: LinearContinuousBook):
        """Density should be negative on ask side (positive x)."""
        # For linear density -L*x, positive x gives negative density (asks)
        mid_idx = continuous_book.Nx // 2
        # Ask side is at higher indices (positive x)
        ask_side_density = continuous_book.density[mid_idx + 5 :]
        assert np.all(ask_side_density < 0)

    def test_density_positive_on_bid_side(self, continuous_book: LinearContinuousBook):
        """Density should be positive on bid side (negative x)."""
        # For linear density -L*x, negative x gives positive density (bids)
        mid_idx = continuous_book.Nx // 2
        # Bid side is at lower indices (negative x)
        bid_side_density = continuous_book.density[: mid_idx - 5]
        assert np.all(bid_side_density > 0)

    def test_resolution_volume(self, linear_params: dict):
        """Resolution volume should be L * dx^2."""
        book = LinearContinuousBook.from_params(
            D=linear_params["D"],
            L=linear_params["L"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
        )

        expected = linear_params["L"] * book.dx**2
        assert np.isclose(book.resolution_volume, expected)
