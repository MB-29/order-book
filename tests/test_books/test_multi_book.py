"""
Tests for MultiDiscreteBook class.
"""

import numpy as np
import pytest

from llob import DiscreteBook, LinearDiscreteBook, MultiDiscreteBook


class TestMultiDiscreteBookConstruction:
    """Tests for MultiDiscreteBook construction and initialization."""

    def test_from_params_creates_multiple_books(self, multi_book_params: dict):
        """from_params should create the correct number of actor books."""
        book = MultiDiscreteBook.from_params(
            D=multi_book_params["D"],
            xmin=multi_book_params["xmin"],
            xmax=multi_book_params["xmax"],
            n_grid=multi_book_params["n_grid"],
            L_list=multi_book_params["L_list"],
            nu_list=multi_book_params["nu_list"],
            lambd_list=multi_book_params["lambd_list"],
        )

        assert book.N_actors == len(multi_book_params["L_list"])
        assert len(book.books) == book.N_actors

    def test_from_params_validates_list_lengths(self, small_grid: dict):
        """from_params should validate that all lists have same length."""
        with pytest.raises(AssertionError):
            MultiDiscreteBook.from_params(
                D=0.5,
                xmin=small_grid["xmin"],
                xmax=small_grid["xmax"],
                n_grid=small_grid["n_grid"],
                L_list=[5.0, 5.0],
                nu_list=[0.0],  # Wrong length
                lambd_list=[0.0, 0.0],
            )

    def test_from_params_creates_linear_books_when_nu_zero(
        self, multi_book_params: dict
    ):
        """from_params should create LinearDiscreteBook when nu=0."""
        book = MultiDiscreteBook.from_params(
            D=multi_book_params["D"],
            xmin=multi_book_params["xmin"],
            xmax=multi_book_params["xmax"],
            n_grid=multi_book_params["n_grid"],
            L_list=multi_book_params["L_list"],
            nu_list=[0.0, 0.0],
            lambd_list=[0.0, 0.0],
        )

        for actor_book in book.books:
            assert isinstance(actor_book, LinearDiscreteBook)

    def test_from_params_creates_discrete_books_when_nu_nonzero(self, small_grid: dict):
        """from_params should create DiscreteBook when nu > 0."""
        book = MultiDiscreteBook.from_params(
            D=0.5,
            xmin=small_grid["xmin"],
            xmax=small_grid["xmax"],
            n_grid=small_grid["n_grid"],
            L_list=[5.0],
            nu_list=[0.1],
            lambd_list=[0.5],
        )

        # Should be DiscreteBook but not LinearDiscreteBook
        for actor_book in book.books:
            assert isinstance(actor_book, DiscreteBook)
            assert not isinstance(actor_book, LinearDiscreteBook)

    def test_mixed_linear_nonlinear_actors(self, small_grid: dict):
        """from_params should handle mixed linear and nonlinear actors."""
        book = MultiDiscreteBook.from_params(
            D=0.5,
            xmin=small_grid["xmin"],
            xmax=small_grid["xmax"],
            n_grid=small_grid["n_grid"],
            L_list=[5.0, 5.0],
            nu_list=[0.0, 0.1],  # First linear, second nonlinear
            lambd_list=[0.0, 0.5],
        )

        assert isinstance(book.books[0], LinearDiscreteBook)
        assert not isinstance(book.books[1], LinearDiscreteBook)

    def test_grid_properties(self, multi_book: MultiDiscreteBook):
        """MultiDiscreteBook should have correct grid properties."""
        assert len(multi_book.X) == multi_book.n_grid
        assert multi_book.X[0] == multi_book.xmin
        assert multi_book.X[-1] == multi_book.xmax
        assert multi_book.dx > 0


class TestMultiDiscreteBookAggregation:
    """Tests for volume aggregation functionality."""

    def test_get_ask_volumes_aggregates(self, multi_book: MultiDiscreteBook):
        """get_ask_volumes should sum volumes from all actors."""
        aggregate = multi_book.get_ask_volumes()

        # Should equal sum of individual book volumes
        expected = np.sum([book.get_ask_volumes() for book in multi_book.books], axis=0)
        np.testing.assert_array_equal(aggregate, expected)

    def test_get_bid_volumes_aggregates(self, multi_book: MultiDiscreteBook):
        """get_bid_volumes should sum volumes from all actors."""
        aggregate = multi_book.get_bid_volumes()

        # Should equal sum of individual book volumes
        expected = np.sum([book.get_bid_volumes() for book in multi_book.books], axis=0)
        np.testing.assert_array_equal(aggregate, expected)

    def test_get_ask_volumes_with_index(self, multi_book: MultiDiscreteBook):
        """get_ask_volumes with index should return single value."""
        index = 5
        result = multi_book.get_ask_volumes(index=index)
        expected = multi_book.get_ask_volumes()[index]

        assert result == expected

    def test_get_bid_volumes_with_index(self, multi_book: MultiDiscreteBook):
        """get_bid_volumes with index should return single value."""
        index = 5
        result = multi_book.get_bid_volumes(index=index)
        expected = multi_book.get_bid_volumes()[index]

        assert result == expected

    def test_best_ask_is_minimum_across_actors(self, multi_book: MultiDiscreteBook):
        """Best ask index should be minimum across all actor books."""
        multi_book.update_price()

        actor_best_asks = [book.best_ask_index for book in multi_book.books]
        assert multi_book.best_ask_index == min(actor_best_asks)

    def test_best_bid_is_maximum_across_actors(self, multi_book: MultiDiscreteBook):
        """Best bid index should be maximum across all actor books."""
        multi_book.update_price()

        actor_best_bids = [book.best_bid_index for book in multi_book.books]
        assert multi_book.best_bid_index == max(actor_best_bids)


class TestMultiDiscreteBookPriceTracking:
    """Tests for price tracking functionality."""

    def test_update_price_sets_best_ask_bid(self, multi_book: MultiDiscreteBook):
        """update_price should set best ask and bid prices."""
        multi_book.update_price()

        assert multi_book.best_ask_index >= 0
        assert multi_book.best_bid_index >= 0
        assert multi_book.best_ask_index < multi_book.n_grid
        assert multi_book.best_bid_index < multi_book.n_grid

    def test_update_price_updates_all_actor_books(self, multi_book: MultiDiscreteBook):
        """update_price should update prices in all actor books."""
        multi_book.update_price()

        for book in multi_book.books:
            assert book.best_ask_index >= 0
            assert book.best_bid_index >= 0


class TestMultiDiscreteBookTimeEvolution:
    """Tests for time evolution functionality."""

    def test_timestep_updates_all_actors(
        self, multi_book: MultiDiscreteBook, seed_random
    ):
        """timestep should update all actor books."""
        # Record initial state
        initial_volumes = [
            (book.get_ask_volumes().copy(), book.get_bid_volumes().copy())
            for book in multi_book.books
        ]

        multi_book.timestep(tstep=1.0, volume=1.0)

        # At least one book should have changed
        any_changed = False
        for i, book in enumerate(multi_book.books):
            ask_changed = not np.array_equal(
                book.get_ask_volumes(), initial_volumes[i][0]
            )
            bid_changed = not np.array_equal(
                book.get_bid_volumes(), initial_volumes[i][1]
            )
            if ask_changed or bid_changed:
                any_changed = True
                break

        assert any_changed

    def test_stochastic_timestep_runs_all_books(
        self, multi_book: MultiDiscreteBook, seed_random
    ):
        """stochastic_timestep should run dynamics for all actor books."""
        # Just ensure no errors
        multi_book.stochastic_timestep()
        multi_book.update_price()

        assert multi_book.best_ask_index >= 0

    def test_order_reaction_when_not_crossed(self, multi_book: MultiDiscreteBook):
        """order_reaction should be no-op when books not crossed."""
        # Ensure not crossed
        multi_book.update_price()
        if multi_book.best_ask_index <= multi_book.best_bid_index:
            pytest.skip("Book is crossed")

        initial_asks = [book.get_ask_volumes().copy() for book in multi_book.books]
        initial_bids = [book.get_bid_volumes().copy() for book in multi_book.books]

        multi_book.order_reaction()

        for i, book in enumerate(multi_book.books):
            np.testing.assert_array_equal(book.get_ask_volumes(), initial_asks[i])
            np.testing.assert_array_equal(book.get_bid_volumes(), initial_bids[i])


class TestMultiDiscreteBookMetaorderExecution:
    """Tests for metaorder execution functionality."""

    def test_execute_metaorder_tracks_actor_trades(self, multi_book: MultiDiscreteBook):
        """execute_metaorder should track which actors provided liquidity."""
        multi_book.execute_metaorder(1.0)

        # actor_trades should sum to 1 (normalized)
        assert np.isclose(np.sum(multi_book.actor_trades), 1.0)

    def test_execute_metaorder_zero_is_noop(self, multi_book: MultiDiscreteBook):
        """execute_metaorder with zero volume should be a no-op."""
        initial_asks = [book.get_ask_volumes().copy() for book in multi_book.books]
        initial_bids = [book.get_bid_volumes().copy() for book in multi_book.books]

        multi_book.execute_metaorder(0.0)

        for i, book in enumerate(multi_book.books):
            np.testing.assert_array_equal(book.get_ask_volumes(), initial_asks[i])
            np.testing.assert_array_equal(book.get_bid_volumes(), initial_bids[i])

    def test_execute_metaorder_positive_buys_asks(self, multi_book: MultiDiscreteBook):
        """Positive volume should buy from ask side."""
        initial_total_ask = np.sum(multi_book.get_ask_volumes())

        # Execute a larger volume to ensure some is consumed
        multi_book.execute_metaorder(10.0)

        final_total_ask = np.sum(multi_book.get_ask_volumes())
        assert final_total_ask <= initial_total_ask

    def test_execute_metaorder_negative_sells_bids(self, multi_book: MultiDiscreteBook):
        """Negative volume should sell to bid side."""
        initial_total_bid = np.sum(multi_book.get_bid_volumes())

        # Execute a larger volume to ensure some is consumed
        multi_book.execute_metaorder(-10.0)

        final_total_bid = np.sum(multi_book.get_bid_volumes())
        assert final_total_bid <= initial_total_bid


class TestMultiDiscreteBookMeasurements:
    """Tests for measurement functionality."""

    def test_get_measures_returns_dict(self, multi_book: MultiDiscreteBook):
        """get_measures should return a dictionary of measurements."""
        measures = multi_book.get_measures()

        assert isinstance(measures, dict)
        assert "bid" in measures
        assert "ask" in measures
        assert "actor_trades" in measures

    def test_get_measure_bid_volumes(self, multi_book: MultiDiscreteBook):
        """get_measure should return bid volumes."""
        volumes = multi_book.get_measure("bid_volumes")

        np.testing.assert_array_equal(volumes, multi_book.get_bid_volumes())

    def test_get_measure_ask_volumes(self, multi_book: MultiDiscreteBook):
        """get_measure should return ask volumes."""
        volumes = multi_book.get_measure("ask_volumes")

        np.testing.assert_array_equal(volumes, multi_book.get_ask_volumes())

    def test_get_measure_best_ask(self, multi_book: MultiDiscreteBook):
        """get_measure should return best_ask attribute."""
        best_ask = multi_book.get_measure("best_ask")

        assert best_ask == multi_book.best_ask
