import numpy as np
from functools import reduce

from limit_orders import LimitOrders
from discrete_book import DiscreteBook
from linear_discrete_book import LinearDiscreteBook


class MultiDiscreteBook(DiscreteBook):
    """A multi-actor order book.
    """

    def __init__(self, **multi_book_args):
        """
        :param multi_book_args: Packs book arguments where L and nu are arrays
        """
        L_list = multi_book_args['L']
        lambd_list = multi_book_args['lambd']
        nu_list = multi_book_args['nu']
        self.N_actors = len(L_list)
        self.books = []

        # Get spatial values
        self.xmin = multi_book_args['xmin']
        self.xmax = multi_book_args['xmax']
        self.n_diff = multi_book_args['n_diff']
        self.Nx = multi_book_args['Nx']
        self.X, self.dx = np.linspace(
            self.xmin, self.xmax, num=self.Nx, retstep=True)
        self.L = multi_book_args['L']

        for n in range(self.N_actors):
            L = L_list[n]
            lambd = lambd_list[n]
            nu = nu_list[n]
            linear = nu == 0
            model = LinearDiscreteBook if linear else DiscreteBook
            multi_book_args['L'] = L
            multi_book_args['lambd'] = lambd
            multi_book_args['nu'] = nu
            self.books.append(model(**multi_book_args))

        self.dq = 0

        self.best_ask = np.mean(
            [actor_book.best_ask for actor_book in self.books])
        self.best_bid = np.mean(
            [actor_book.best_bid for actor_book in self.books])

        self.y_max = np.sum([actor_book.y_max for actor_book in self.books])

    def get_ask_volumes(self):
        return reduce(np.add, [actor_book.get_ask_volumes() for actor_book in self.books])

    def get_bid_volumes(self):
        return reduce(np.add, [actor_book.get_bid_volumes() for actor_book in self.books])

    def timestep(self):
        self.execute_metaorder(self.dq)
        for actor_book in self.books:
            actor_book.evolve()
        self.best_ask = np.mean(
            [actor_book.best_ask for actor_book in self.books])
        self.best_bid = np.mean(
            [actor_book.best_bid for actor_book in self.books])

    def execute_metaorder(self, trade_volume):
        """Execute a meta-order on the total order book, by consuming the volumes
        of the various books in the order of proximity.
        :param trade_volume: algebraic volume to execute
        """
        executed_volume = 0
        traded = True
        while traded:
            traded = False
            # Each book is considered at each loop, and executed if possible
            for actor_book in self.books:
                volume = actor_book.best_ask_volume if trade_volume > 0 else - \
                    actor_book.best_bid_volume
                if abs(volume) + executed_volume > abs(trade_volume):
                    continue
                actor_book.execute_metaorder(volume)
                executed_volume += abs(volume)
                traded = True

    # ================== ANIMATION ==================

    #  Additional that aim at displaying multiple books

    def set_animation(self, fig=None, lims=None):
        """Create subplot axes, lines and texts
        """
        # Ax
        self.volume_ax = fig.add_subplot(1, 2, 1)
        self.volume_ax.set_ylim((0, self.y_max))
        width = max(1/self.Nx, 0.02)

        # Bars
        self.ask_bars = []
        self.bid_bars = []
        for index, actor_book in enumerate(self.books):
            brightness = 1 - (index/self.N_actors)
            # Ask
            actor_ask_bars = self.volume_ax.bar(
                self.X, actor_book.get_ask_volumes(),
                align='edge',
                label=f'Ask {index}',
                width=width,
                color=(0, 0, brightness),
                animated=True)
            self.ask_bars.append(actor_ask_bars)

            # Bid
            actor_bid_bars = self.volume_ax.bar(
                self.X, actor_book.get_bid_volumes(),
                align='edge',
                label=f'Ask {index}',
                width=-width,
                color=(brightness, 0, 0),
                animated=True)
            self.bid_bars.append(actor_bid_bars)

    def init_animation(self):
        """Init function called by FuncAnimation
        """
        result = []
        # Ask
        for actor_index in range(1):
            actor_ask_bars = self.ask_bars[actor_index]
            for x_index in range(self.Nx):
                bar = actor_ask_bars[x_index]
                bar.set_y(0)
                bar.set_height(0)
                result.append(bar)
        # Bid
        for actor_index in range(1):
            actor_bid_bars = self.bid_bars[actor_index]
            for x_index in range(self.Nx):
                bar = actor_bid_bars[x_index]
                bar.set_y(0)
                bar.set_height(0)
                result.append(bar)
        return result

    def update_animation(self, n):
        """Update function called by FuncAnimation
        """
        # Axis
        y_max = 1.5*self.dx * self.xmax * self.L
        padding = 5
        self.timestep()

        # self.ask_bars[index].set_height(self.get_ask_volumes()[index])
        # self.bid_bars[index].set_height(self.get_bid_volumes()[index])
        # Update ask bars
        result = []
        heights = np.zeros(self.Nx)
        for actor_index in range(self.N_actors):
            actor_ask_bars = self.ask_bars[actor_index]
            for x_index in range(self.Nx):
                bar = actor_ask_bars[x_index]
                bar.set_y(heights[x_index] + padding)
                bar.set_height(
                    self.books[actor_index].get_ask_volumes()[x_index])
                heights[x_index] += padding + bar.get_height()
                result.append(bar)
        # Update bid bars
        heights = np.zeros(self.Nx)
        for actor_index in range(self.N_actors):
            actor_bid_bars = self.bid_bars[actor_index]
            for x_index in range(self.Nx):
                bar = actor_bid_bars[x_index]
                bar.set_y(heights[x_index] + padding)
                bar.set_height(
                    self.books[actor_index].get_bid_volumes()[x_index])
                heights[x_index] += padding + bar.get_height()
                result.append(bar)

        # self.best_ask_axis.set_data([self.best_ask, self.best_ask], [0, y_max])
        # self.best_bid_axis.set_data([self.best_bid, self.best_bid], [0, y_max])
        # result = [bar for bar in actor_ask_bars for actor_ask_bars in self.ask_bars] + \
        #     [bar for bar in actor_bid_bars for actor_bid_bars in self.bid_bars]
        return result
