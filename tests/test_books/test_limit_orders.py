"""
Tests for LimitOrders class.
"""

import numpy as np
import pytest

from llob import LimitOrders


class TestLimitOrdersConstruction:
    """Tests for LimitOrders construction and initialization."""

    def test_from_params_creates_valid_instance(self, linear_params: dict):
        """from_params should create a valid LimitOrders instance."""
        orders = LimitOrders.from_params(
            side="ask",
            lambd=linear_params["lambd"],
            nu=linear_params["nu"],
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
            L=linear_params["L"],
        )

        assert orders.side == "ask"
        assert linear_params["D"] == orders.D
        assert linear_params["L"] == orders.L
        assert orders.Nx == linear_params["Nx"]
        assert len(orders.volumes) == linear_params["Nx"]

    def test_from_params_computes_L_when_none(self):
        """from_params should compute L from lambd, nu, D when L is None."""
        D = 0.5
        nu = 0.1
        lambd = 1.0
        expected_L = lambd / np.sqrt(nu * D)

        orders = LimitOrders.from_params(
            side="ask",
            lambd=lambd,
            nu=nu,
            D=D,
            xmin=-10,
            xmax=10,
            Nx=20,
            L=None,
        )

        assert np.isclose(orders.L, expected_L)

    def test_from_params_validates_side(self, linear_params: dict):
        """from_params should only accept 'ask' or 'bid' as side."""
        with pytest.raises(AssertionError):
            LimitOrders.from_params(
                side="invalid",
                lambd=linear_params["lambd"],
                nu=linear_params["nu"],
                D=linear_params["D"],
                xmin=linear_params["xmin"],
                xmax=linear_params["xmax"],
                Nx=linear_params["Nx"],
                L=linear_params["L"],
            )

    def test_initial_density_linear(self, linear_params: dict):
        """Linear initial density should give volumes proportional to |x|."""
        orders = LimitOrders.from_params(
            side="ask",
            lambd=linear_params["lambd"],
            nu=linear_params["nu"],
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
            L=linear_params["L"],
            initial_density="linear",
        )

        # For ask side with linear density, volumes should be nonzero for x > 0
        # (asks are on positive price side)
        mid_idx = linear_params["Nx"] // 2
        # Ask orders should be on the positive price side (x > 0)
        assert np.sum(orders.volumes[mid_idx:]) > 0  # Some volume on positive side

    def test_initial_density_empty(self, linear_params: dict):
        """Empty initial density should give all zero volumes."""
        orders = LimitOrders.from_params(
            side="ask",
            lambd=linear_params["lambd"],
            nu=linear_params["nu"],
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
            L=linear_params["L"],
            initial_density="empty",
        )

        assert np.all(orders.volumes == 0)


class TestLimitOrdersPriceTracking:
    """Tests for price tracking functionality."""

    def test_update_best_price_ask_side(self, limit_orders_ask: LimitOrders):
        """update_best_price should find leftmost nonzero volume for ask."""
        limit_orders_ask.update_best_price()

        # Best ask should be the leftmost nonzero index
        nonzero_indices = np.nonzero(limit_orders_ask.volumes)[0]
        if nonzero_indices.size > 0:
            assert limit_orders_ask.best_price_index == nonzero_indices[0]

    def test_update_best_price_bid_side(self, limit_orders_bid: LimitOrders):
        """update_best_price should find rightmost nonzero volume for bid."""
        limit_orders_bid.update_best_price()

        # Best bid should be the rightmost nonzero index
        nonzero_indices = np.nonzero(limit_orders_bid.volumes)[0]
        if nonzero_indices.size > 0:
            assert limit_orders_bid.best_price_index == nonzero_indices[-1]

    def test_best_price_empty_book(self, linear_params: dict):
        """Empty book should use boundary index as best price."""
        orders = LimitOrders.from_params(
            side="ask",
            lambd=linear_params["lambd"],
            nu=linear_params["nu"],
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
            L=linear_params["L"],
            initial_density="empty",
        )

        orders.update_best_price()
        # For ask, boundary index is -1 (stored as -1, not Nx-1)
        assert orders.best_price_index == orders.boundary_index

    def test_stationary_density_linear_regime(self, limit_orders_ask: LimitOrders):
        """In linear regime (nu=0), stationary_density should be L*x on ask side."""
        # For ask side (sign=-1), density is nonzero only for x > 0
        # (asks are at higher prices than midprice which is at 0)
        # For x < 0 (wrong side for ask), density should be 0
        x_wrong_side = -5.0  # Negative x is wrong side for ask
        density_wrong = limit_orders_ask.stationary_density(x_wrong_side)
        assert density_wrong == 0.0

        # For x > 0, density = L * x
        x_correct_side = 5.0  # Positive x is correct side for ask
        density_correct = limit_orders_ask.stationary_density(x_correct_side)
        expected = limit_orders_ask.L * x_correct_side
        assert np.isclose(density_correct, expected)

    def test_stationary_density_nonlinear_regime(
        self, limit_orders_nonlinear: LimitOrders
    ):
        """In nonlinear regime, stationary_density should follow exponential formula."""
        # For ask side (sign=-1), density is nonzero only for x > 0
        x = 5.0  # On the correct side for ask (positive prices)
        density = limit_orders_nonlinear.stationary_density(x)

        # For nonlinear regime: (lambd/nu) * (1 - exp(-|x|/x_crit))
        nu = limit_orders_nonlinear.nu
        lambd = limit_orders_nonlinear.lambd
        D = limit_orders_nonlinear.D
        x_crit = np.sqrt(D / nu)
        expected = (lambd / nu) * (1 - np.exp(-abs(x) / x_crit))

        assert np.isclose(density, expected)

        # Wrong side should be 0
        x_wrong = -5.0  # Negative x is wrong side for ask
        assert limit_orders_nonlinear.stationary_density(x_wrong) == 0.0


class TestLimitOrdersExecution:
    """Tests for order execution functionality."""

    def test_execute_best_orders_partial_fill(self, limit_orders_ask: LimitOrders):
        """execute_best_orders should partially fill if less than best price volume."""
        initial_volume = limit_orders_ask.best_price_volume
        if initial_volume > 1:
            execute_volume = 1.0
            limit_orders_ask.execute_best_orders(execute_volume)

            # Volume at best price should decrease
            assert (
                limit_orders_ask.volumes[limit_orders_ask.best_price_index]
                < initial_volume
            )

    def test_execute_best_orders_walks_price_levels(
        self, limit_orders_ask: LimitOrders
    ):
        """execute_best_orders should walk through price levels if volume exceeds best."""
        initial_best_idx = limit_orders_ask.best_price_index
        total_volume = np.sum(limit_orders_ask.volumes)

        if total_volume > 0:
            # Execute more than the best price volume
            execute_volume = float(limit_orders_ask.best_price_volume + 1)
            limit_orders_ask.execute_best_orders(execute_volume)
            limit_orders_ask.update_best_price()

            # Best price index should have moved
            assert limit_orders_ask.best_price_index != initial_best_idx

    def test_execute_best_orders_raises_on_no_liquidity(self, linear_params: dict):
        """execute_best_orders should raise ValueError when liquidity exhausted."""
        orders = LimitOrders.from_params(
            side="ask",
            lambd=linear_params["lambd"],
            nu=linear_params["nu"],
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            Nx=linear_params["Nx"],
            L=linear_params["L"],
            initial_density="empty",
        )
        # Set one small volume so we have something to deplete
        orders.volumes[5] = 1

        with pytest.raises(ValueError, match="liquidity"):
            orders.execute_best_orders(1000.0)

    def test_execute_orders_respects_available_volume(
        self, limit_orders_ask: LimitOrders
    ):
        """execute_orders should not execute more than available at each level."""
        requested = np.full(limit_orders_ask.Nx, 1000.0)
        original_volumes = limit_orders_ask.volumes.copy()

        executed = limit_orders_ask.execute_orders(requested)

        # Executed should be min(requested, available)
        assert np.all(executed <= original_volumes)
        assert np.all(limit_orders_ask.volumes >= 0)

    def test_get_available_volume(self, limit_orders_ask: LimitOrders):
        """get_available_volume should sum volumes between index and best price."""
        best_idx = limit_orders_ask.best_price_index
        target_idx = best_idx + 5  # A few levels away

        if target_idx < limit_orders_ask.Nx:
            available = limit_orders_ask.get_available_volume(target_idx)
            expected = np.sum(
                limit_orders_ask.volumes[
                    min(best_idx, target_idx) : max(best_idx, target_idx) + 1
                ]
            )
            assert available == expected


class TestLimitOrdersStochasticDynamics:
    """Tests for stochastic dynamics (deposition, cancellation, jumps)."""

    def test_deposition_increases_volume(
        self, limit_orders_nonlinear: LimitOrders, seed_random
    ):
        """Deposition should generally increase total volume."""
        initial_total = np.sum(limit_orders_nonlinear.volumes)

        # Run many depositions to see effect
        for _ in range(100):
            limit_orders_nonlinear.deposition(spread=1)

        final_total = np.sum(limit_orders_nonlinear.volumes)

        # With positive lambd, total should increase (probabilistically)
        assert final_total >= initial_total

    def test_cancellation_decreases_volume(
        self, limit_orders_nonlinear: LimitOrders, seed_random
    ):
        """Cancellation should generally decrease total volume."""
        # Ensure we have volume to cancel
        limit_orders_nonlinear.volumes[:] = 100
        initial_total = np.sum(limit_orders_nonlinear.volumes)

        limit_orders_nonlinear.cancellation()

        final_total = np.sum(limit_orders_nonlinear.volumes)

        # With positive nu, total should decrease
        assert final_total <= initial_total

    def test_jumps_approximately_conserves_volume(
        self, limit_orders_ask: LimitOrders, seed_random
    ):
        """Jumps should approximately conserve total volume (ignoring boundary)."""
        # Set uniform volumes away from boundary
        limit_orders_ask.volumes[5:-5] = 10
        limit_orders_ask.boundary_flow = 0  # No boundary flux
        initial_total = np.sum(limit_orders_ask.volumes)

        limit_orders_ask.jumps()

        final_total = np.sum(limit_orders_ask.volumes)

        # Should be approximately conserved (some boundary effects possible)
        assert abs(final_total - initial_total) < initial_total * 0.5  # Within 50%
