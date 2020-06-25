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

        # Spatial values
        self.xmin = multi_book_args['xmin']
        self.xmax = multi_book_args['xmax']
        self.Nx = multi_book_args['Nx']
        self.X, self.dx = np.linspace(
            self.xmin, self.xmax, num=self.Nx, retstep=True)

        # Elementary timestep
        self.D = multi_book_args['D']
        self.dt = (self.dx)**2 / (2 * self.D)

        # The create the various books
        for n in range(self.N_actors):
            L = L_list[n]
            lambd = lambd_list[n]
            nu = nu_list[n]
            linear = (nu == 0)
            model = LinearDiscreteBook if linear else DiscreteBook
            multi_book_args['L'] = L
            multi_book_args['lambd'] = lambd
            multi_book_args['nu'] = nu
            self.books.append(model(**multi_book_args))

        self.update_price()

        # Measures
        self.actor_trades = np.full(self.N_actors, 1/self.N_actors)

        self.y_max = np.sum([actor_book.y_max for actor_book in self.books])

    def get_ask_volumes(self, index=None):
        volumes = reduce(np.add, [actor_book.get_ask_volumes() for actor_book in self.books])
        if index is not None:
            return volumes[index]
        return volumes

    def get_bid_volumes(self, index=None):
        volumes = reduce(np.add, [actor_book.get_bid_volumes() for actor_book in self.books])
        if index is not None:
            return volumes[index]
        return volumes

    def get_ask_proportions(self, index):
        return self.books[index].get_ask_volumes() / self.get_ask_volumes
        
    def get_bid_proportions(self, index):
        return self.books[index].get_bid_volumes() / self.get_bid_volumes
        
    # ================== TIME EVOLUTION ==================

    def stochastic_timestep(self):
        for actor_book in self.books:
            actor_book.stochastic_timestep()

    def update_price(self):
        for actor_book in self.books:
            actor_book.update_price()
        self.best_ask_index = np.min(
            [actor_book.best_ask_index for actor_book in self.books])
        self.best_ask = self.X[self.best_ask_index]
        self.best_bid_index = np.max(
            [actor_book.best_bid_index for actor_book in self.books])
        self.best_bid = self.X[self.best_bid_index]
        self.bid_volume = self.get_bid_volumes()[self.best_bid_index]
        self.ask_volume = self.get_ask_volumes()[self.best_ask_index]

    def order_reaction(self):
        """Reaction step : matched orders are executed, regardless of
        the book type their are matched with. 
        """
        if self.best_ask_index > self.best_bid_index:
            return

        reaction_volumes = np.minimum(
            self.get_ask_volumes(), self.get_bid_volumes())
        for side in ['ask', 'bid']:
            side_volumes = getattr(self, f'get_{side}_volumes')()
            # Mask array indicating where side's volumes are lower that the other side's
            limiting_volume = np.array(side_volumes <= reaction_volumes)
            for actor_book in self.books:
                actor_side_orders = getattr(actor_book, f'{side}_orders')
                actor_proportion = actor_side_orders.volumes / side_volumes
                # Execute all the side's liquidity where side is lacking volume,
                # else execute a proportional fraction of the other side's liquidit
                executed_volumes = np.where(
                    limiting_volume, reaction_volumes, actor_proportion * reaction_volumes)
                actor_side_orders.execute_orders(executed_volumes)

    def execute_metaorder(self, trade_volume):
        """Execute a meta-order on the total order book, by consuming the volumes
        of the various books in the order of proximity.
        :param trade_volume: algebraic volume to execute
        """
        side = 'bid' if trade_volume < 0 else 'ask'
        sign = 1 if side =='bid' else -1

        executed_volume = 0
        self.actor_trades.fill(0)
        # price_volumes = [getattr(actor_book, f'best_{side}_volume') for actor_book in self.books]
        # The execution price is market's best price
        price_index = getattr(self, f'best_{side}_index')
        total_price_volume = getattr(self, f'{side}_volume')
        # Consume best price orders, price step after price step 
        while executed_volume + total_price_volume < abs(trade_volume):
            for actor_index, actor_book in enumerate(self.books):
                # Ignore actors whose best price is further than
                # the market's best price
                actor_price_index = getattr(actor_book, f'best_{side}_index')
                if actor_price_index != price_index:
                    continue
                actor_volume = getattr(actor_book, f'best_{side}_volume')
                actor_book = self.books[actor_index]
                actor_book.execute_metaorder(- sign * actor_volume)
                self.actor_trades[actor_index] += actor_volume
                executed_volume += actor_volume
            # Increment market price index by one
            price_index -= sign
            self.update_price()
            total_price_volume = getattr(self, f'{side}_volume')
            price_index = getattr(self, f'best_{side}_index')
        # At this point best price volume is greater than what needs to be executed.
        # Exectute each book in proportion to their volume 
        remaining_volume = abs(trade_volume) - executed_volume
        for actor_index, actor_book in enumerate(self.books):
            actor_price_index = getattr(actor_book, f'best_{side}_index')
            if actor_price_index != price_index:
                continue
            actor_volume = getattr(actor_book, f'best_{side}_volume')
            actor_volume = actor_volume / total_price_volume * remaining_volume
            actor_book.execute_metaorder(actor_volume)
            self.actor_trades[actor_index] += actor_volume
        self.actor_trades /= abs(trade_volume)
        # print(f'after execution, price index = {price_index}')


    def get_measures(self):
        measures = {
            'bid': self.best_bid,
            'ask': self.best_ask,
            'actor_trades': np.copy(self.actor_trades)
        }
        return measures

    # ================== ANIMATION ==================

    # Override animation methods to enable a visual
    # distinction between the various actors

    def set_animation(self, fig=None, lims=None):
        """Create subplot axes, lines and texts
        """
        # Ax
        self.volume_ax = fig.add_subplot(1, 2, 1)
        self.volume_ax.set_ylim((0, self.y_max))
        self.volume_ax.set_title('Order volumes')
        width = max((self.xmax-self.xmin)/self.Nx, 0.02)

        # Lines
        # Lines
        self.volume_ax.plot([0, 0], [
            -self.y_max, self.y_max], color='black', lw=0.5, ls='dashed')
        self.best_ask_axis, = self.volume_ax.plot(
            [], [], color='blue', ls='dashed', lw=1, label='best ask')
        self.best_bid_axis, = self.volume_ax.plot(
            [], [], color='red', ls='dashed', lw=1, label='best bid')

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

    def update_animation(self, tstep, volume):
        """Update function called by FuncAnimation
        """
        self.timestep(tstep, volume)

        padding = 0.01 * self.y_max

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

        self.best_ask_axis.set_data(
            [self.best_ask, self.best_ask], [0, self.y_max])
        self.best_bid_axis.set_data(
            [self.best_bid, self.best_bid], [0, self.y_max])

        result.extend([self.best_ask_axis, self.best_bid_axis])
        return result
