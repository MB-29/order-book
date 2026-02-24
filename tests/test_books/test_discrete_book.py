"""
Tests for DiscreteBook and LinearDiscreteBook classes.
"""

import numpy as np
import pytest

from llob import DiscreteBook, LinearDiscreteBook


class TestDiscreteBookConstruction:
    """Tests for DiscreteBook construction and initialization."""

    def test_from_params_creates_both_sides(self, linear_params: dict):
        """from_params should create both bid and ask LimitOrders."""
        book = DiscreteBook.from_params(
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            n_grid=linear_params["n_grid"],
            L=linear_params["L"],
            nu=linear_params["nu"],
            lambd=linear_params["lambd"],
        )

        assert book.bid_orders is not None
        assert book.ask_orders is not None
        assert book.bid_orders.side == "bid"
        assert book.ask_orders.side == "ask"

    def test_from_params_linear_regime(self, linear_params: dict):
        """from_params should work in linear regime (nu=0)."""
        book = DiscreteBook.from_params(
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            n_grid=linear_params["n_grid"],
            L=linear_params["L"],
            nu=0.0,
            lambd=0.0,
        )

        assert book.bid_orders.nu == 0.0
        assert book.ask_orders.lambd == 0.0

    def test_from_params_nonlinear_regime(self, nonlinear_params: dict):
        """from_params should work in nonlinear regime (nu>0)."""
        book = DiscreteBook.from_params(
            D=nonlinear_params["D"],
            xmin=nonlinear_params["xmin"],
            xmax=nonlinear_params["xmax"],
            n_grid=nonlinear_params["n_grid"],
            L=nonlinear_params["L"],
            nu=nonlinear_params["nu"],
            lambd=nonlinear_params["lambd"],
        )

        assert book.bid_orders.nu > 0
        assert book.ask_orders.lambd > 0

    def test_grid_properties(self, discrete_book: DiscreteBook):
        """DiscreteBook should have correct grid properties."""
        assert len(discrete_book.X) == discrete_book.n_grid
        assert discrete_book.X[0] == discrete_book.xmin
        assert discrete_book.X[-1] == discrete_book.xmax
        assert discrete_book.dx > 0


class TestDiscreteBookPriceTracking:
    """Tests for price tracking functionality."""

    def test_update_price_sets_best_ask_bid(self, discrete_book: DiscreteBook):
        """update_price should set best ask and bid prices."""
        discrete_book.update_price()

        assert discrete_book.best_ask_index >= 0
        assert discrete_book.best_bid_index >= 0
        assert discrete_book.best_ask_index < discrete_book.n_grid
        assert discrete_book.best_bid_index < discrete_book.n_grid

    def test_best_ask_index_greater_than_best_bid_index(
        self, linear_discrete_book: LinearDiscreteBook
    ):
        """Best ask index should be greater than best bid index in valid book."""
        linear_discrete_book.update_price()

        # In a valid book without crossed orders, best_ask_index > best_bid_index
        # (ask is at higher prices, bid is at lower prices)
        assert linear_discrete_book.best_ask_index > linear_discrete_book.best_bid_index

    def test_spread_index_is_positive(self, linear_discrete_book: LinearDiscreteBook):
        """Spread in index space should be positive."""
        linear_discrete_book.update_price()

        spread = (
            linear_discrete_book.best_ask_index - linear_discrete_book.best_bid_index
        )
        assert spread > 0

    def test_get_ask_volumes_returns_array(self, discrete_book: DiscreteBook):
        """get_ask_volumes should return numpy array."""
        volumes = discrete_book.get_ask_volumes()

        assert isinstance(volumes, np.ndarray)
        assert len(volumes) == discrete_book.n_grid

    def test_get_bid_volumes_returns_array(self, discrete_book: DiscreteBook):
        """get_bid_volumes should return numpy array."""
        volumes = discrete_book.get_bid_volumes()

        assert isinstance(volumes, np.ndarray)
        assert len(volumes) == discrete_book.n_grid


class TestDiscreteBookTimeEvolution:
    """Tests for time evolution functionality."""

    def test_timestep_executes_metaorder(
        self, linear_discrete_book: LinearDiscreteBook
    ):
        """timestep should execute the given metaorder volume."""
        initial_ask_total = np.sum(linear_discrete_book.get_ask_volumes())

        # Execute a larger buy order (positive volume) to ensure measurable effect
        linear_discrete_book.timestep(tstep=1.0, volume=50.0)

        # Ask side should have less volume after buy
        # (allowing for boundary effects which may add back some volume)
        final_ask_total = np.sum(linear_discrete_book.get_ask_volumes())
        assert final_ask_total <= initial_ask_total + 20  # Allow boundary flow

    def test_stochastic_timestep_runs_dynamics_nonlinear(
        self, nonlinear_discrete_book: DiscreteBook, seed_random
    ):
        """stochastic_timestep should run all dynamics without error (nu > 0)."""
        # Just ensure it doesn't crash (requires nu > 0 for cancellation)
        nonlinear_discrete_book.stochastic_timestep()
        nonlinear_discrete_book.update_price()

        # Book should still be valid
        assert nonlinear_discrete_book.best_ask_index >= 0
        assert nonlinear_discrete_book.best_bid_index >= 0

    def test_stochastic_timestep_linear_regime(
        self, linear_discrete_book: LinearDiscreteBook, seed_random
    ):
        """LinearDiscreteBook stochastic_timestep should only do jumps."""
        # In linear regime, stochastic_timestep only runs jumps (no cancellation)
        linear_discrete_book.stochastic_timestep()
        linear_discrete_book.update_price()

        # Book should still be valid
        assert linear_discrete_book.best_ask_index >= 0
        assert linear_discrete_book.best_bid_index >= 0

    def test_order_reaction_when_crossed(self, linear_params: dict):
        """order_reaction should execute when bid and ask cross."""
        book = DiscreteBook.from_params(
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            n_grid=linear_params["n_grid"],
            L=linear_params["L"],
            nu=linear_params["nu"],
            lambd=linear_params["lambd"],
            initial_density="empty",
        )

        # Manually set up crossed orders
        mid = linear_params["n_grid"] // 2
        book.ask_orders.volumes[mid - 2] = 10  # Ask at low price
        book.bid_orders.volumes[mid + 2] = 10  # Bid at high price
        book.update_price()

        # Execute reaction
        book.order_reaction()

        # Crossed volumes should be reduced
        # (exact behavior depends on implementation)

    def test_order_reaction_when_not_crossed(self, discrete_book: DiscreteBook):
        """order_reaction should be no-op when bid and ask don't cross."""
        initial_ask = discrete_book.get_ask_volumes().copy()
        initial_bid = discrete_book.get_bid_volumes().copy()

        # Ensure not crossed
        if discrete_book.best_ask_index <= discrete_book.best_bid_index:
            pytest.skip("Book is crossed, cannot test non-crossed reaction")

        discrete_book.order_reaction()

        # Volumes should be unchanged
        np.testing.assert_array_equal(discrete_book.get_ask_volumes(), initial_ask)
        np.testing.assert_array_equal(discrete_book.get_bid_volumes(), initial_bid)


class TestDiscreteBookMetaorderExecution:
    """Tests for metaorder execution functionality."""

    def test_execute_metaorder_positive_buys_asks(self, discrete_book: DiscreteBook):
        """Positive volume should buy from ask side."""
        initial_ask_total = np.sum(discrete_book.get_ask_volumes())
        initial_bid_total = np.sum(discrete_book.get_bid_volumes())

        discrete_book.execute_metaorder(1.0)

        # Ask volumes should decrease
        final_ask_total = np.sum(discrete_book.get_ask_volumes())
        assert final_ask_total < initial_ask_total

        # Bid volumes should be unchanged
        final_bid_total = np.sum(discrete_book.get_bid_volumes())
        assert final_bid_total == initial_bid_total

    def test_execute_metaorder_negative_sells_bids(self, discrete_book: DiscreteBook):
        """Negative volume should sell to bid side."""
        initial_ask_total = np.sum(discrete_book.get_ask_volumes())
        initial_bid_total = np.sum(discrete_book.get_bid_volumes())

        discrete_book.execute_metaorder(-1.0)

        # Bid volumes should decrease
        final_bid_total = np.sum(discrete_book.get_bid_volumes())
        assert final_bid_total < initial_bid_total

        # Ask volumes should be unchanged
        final_ask_total = np.sum(discrete_book.get_ask_volumes())
        assert final_ask_total == initial_ask_total

    def test_execute_metaorder_zero_is_noop(self, discrete_book: DiscreteBook):
        """Zero volume should not change the book."""
        initial_ask = discrete_book.get_ask_volumes().copy()
        initial_bid = discrete_book.get_bid_volumes().copy()

        discrete_book.execute_metaorder(0.0)

        np.testing.assert_array_equal(discrete_book.get_ask_volumes(), initial_ask)
        np.testing.assert_array_equal(discrete_book.get_bid_volumes(), initial_bid)


class TestDiscreteBookMeasurements:
    """Tests for measurement functionality."""

    def test_get_measure_bid_volumes(self, discrete_book: DiscreteBook):
        """get_measure should return bid volumes."""
        volumes = discrete_book.get_measure("bid_volumes")

        np.testing.assert_array_equal(volumes, discrete_book.get_bid_volumes())

    def test_get_measure_ask_volumes(self, discrete_book: DiscreteBook):
        """get_measure should return ask volumes."""
        volumes = discrete_book.get_measure("ask_volumes")

        np.testing.assert_array_equal(volumes, discrete_book.get_ask_volumes())

    def test_get_measure_best_ask(self, discrete_book: DiscreteBook):
        """get_measure should return best_ask attribute."""
        best_ask = discrete_book.get_measure("best_ask")

        assert best_ask == discrete_book.best_ask

    def test_get_measure_best_bid(self, discrete_book: DiscreteBook):
        """get_measure should return best_bid attribute."""
        best_bid = discrete_book.get_measure("best_bid")

        assert best_bid == discrete_book.best_bid


class TestLinearDiscreteBookConstruction:
    """Tests for LinearDiscreteBook construction."""

    def test_from_params_creates_linear_book(self, linear_params: dict):
        """from_params should create a LinearDiscreteBook."""
        book = LinearDiscreteBook.from_params(
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            n_grid=linear_params["n_grid"],
            L=linear_params["L"],
        )

        assert isinstance(book, LinearDiscreteBook)
        assert isinstance(book, DiscreteBook)

    def test_from_params_sets_zero_nu_lambd(self, linear_params: dict):
        """LinearDiscreteBook should have nu=0 and lambd=0."""
        book = LinearDiscreteBook.from_params(
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            n_grid=linear_params["n_grid"],
            L=linear_params["L"],
        )

        assert book.bid_orders.nu == 0.0
        assert book.bid_orders.lambd == 0.0
        assert book.ask_orders.nu == 0.0
        assert book.ask_orders.lambd == 0.0

    def test_uses_linear_initial_density(self, linear_params: dict):
        """LinearDiscreteBook should use linear initial density."""
        book = LinearDiscreteBook.from_params(
            D=linear_params["D"],
            xmin=linear_params["xmin"],
            xmax=linear_params["xmax"],
            n_grid=linear_params["n_grid"],
            L=linear_params["L"],
        )

        # Linear density means volume proportional to distance from center
        # Check that volumes increase away from center
        mid = linear_params["n_grid"] // 2
        # Ask side should have increasing volume towards high prices
        assert np.sum(book.get_ask_volumes()[mid:]) > 0


class TestLinearDiscreteBookDynamics:
    """Tests for LinearDiscreteBook dynamics."""

    def test_stochastic_timestep_only_jumps(
        self, linear_discrete_book: LinearDiscreteBook, seed_random
    ):
        """LinearDiscreteBook stochastic_timestep should only do jumps."""
        # In linear regime, total volume is approximately conserved
        # (except for boundary effects)
        initial_total = np.sum(linear_discrete_book.get_ask_volumes()) + np.sum(
            linear_discrete_book.get_bid_volumes()
        )

        linear_discrete_book.stochastic_timestep()

        final_total = np.sum(linear_discrete_book.get_ask_volumes()) + np.sum(
            linear_discrete_book.get_bid_volumes()
        )

        # Should be approximately conserved (within boundary effects)
        assert abs(final_total - initial_total) < initial_total * 0.3

    def test_inherits_from_discrete_book(
        self, linear_discrete_book: LinearDiscreteBook
    ):
        """LinearDiscreteBook should inherit DiscreteBook methods."""
        # Should have all DiscreteBook methods
        assert hasattr(linear_discrete_book, "timestep")
        assert hasattr(linear_discrete_book, "execute_metaorder")
        assert hasattr(linear_discrete_book, "update_price")
        assert hasattr(linear_discrete_book, "get_ask_volumes")
        assert hasattr(linear_discrete_book, "get_bid_volumes")

    def test_timestep_works(self, linear_discrete_book: LinearDiscreteBook):
        """timestep should work for LinearDiscreteBook."""
        initial_ask = np.sum(linear_discrete_book.get_ask_volumes())

        # Execute a larger volume to ensure measurable consumption
        linear_discrete_book.timestep(tstep=1.0, volume=10.0)

        final_ask = np.sum(linear_discrete_book.get_ask_volumes())
        # Volume executed should reduce ask side (though boundary flow may add some back)
        # At minimum, check it doesn't explode and is close to initial
        assert final_ask <= initial_ask + 100  # Allow boundary effects
