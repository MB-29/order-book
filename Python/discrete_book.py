import numpy as np

from limit_orders import LimitOrders


class DiscreteBook:
    """Models an order book in the framework of Latent Order Book agent model.
    """

    def __init__(self, **order_args):

        order_args = order_args
        self.bid_orders = LimitOrders(**order_args, side='BID')
        self.ask_orders = LimitOrders(**order_args, side='ASK')

        # Get spatial values
        self.xmin = order_args['xmin']
        self.xmax = order_args['xmax']
        self.n_diff = order_args['n_diff']
        self.Nx = order_args['Nx']
        self.X, self.dx = np.linspace(
            self.xmin, self.xmax, num=self.Nx, retstep=True)
        self.L = order_args['L']

        # Metaorder
        self.update_price()
        self.dq = 0

        # Animation
        self.y_max = 1.5 * \
            max(abs(self.xmin), abs(self.xmax)) * self.L * self.dx

    def get_ask_volumes(self):
        return self.ask_orders.volumes

    def get_bid_volumes(self):
        return self.bid_orders.volumes

    # ================== TIME EVOLUTION ==================

    def timestep(self):
        # self.execute_metaorder(self.dq)
        self.evolve()

    def evolve(self):
        for n in range(self.n_diff):
            self.stochastic_timestep()
            self.order_reaction()
            self.update_price()
        self.update_price()

    def stochastic_timestep(self):
        for orders in [self.ask_orders, self.bid_orders]:
            orders.order_arrivals()
            orders.order_cancellation()
            orders.order_jumps()
            orders.update_best_price()

    def update_price(self):
        for orders in [self.ask_orders, self.bid_orders]:
            orders.update_best_price()

        self.best_ask = self.X[self.ask_orders.best_price_index - 1]
        self.best_bid = self.X[self.bid_orders.best_price_index + 1]
        self.best_ask_volume = self.get_ask_volumes(
        )[self.ask_orders.best_price_index]
        self.best_bid_volume = self.get_bid_volumes(
        )[self.bid_orders.best_price_index]

    # ------------------ Reaction ------------------

    def order_reaction(self):
        """Reaction step : tradable orders are executed
        """
        best_ask_index = self.ask_orders.best_price_index
        best_bid_index = self.bid_orders.best_price_index
        if best_ask_index > best_bid_index:
            return

        executed_volumes = np.minimum(
            self.get_ask_volumes(), self.get_bid_volumes())
        for orders in [self.ask_orders, self.bid_orders]:
            orders.volumes -= executed_volumes
            orders.total_volume -= executed_volumes.sum()

    # ------------------ Metaorder ------------------

    def execute_metaorder(self, volume):
        orders = self.ask_orders if volume > 0 else self.bid_orders
        orders.consume_best_orders(volume)

  # ================== ANIMATION ==================

    def set_animation(self, fig=None, lims=None):
        """Create subplot axes, lines and texts
        """

        self.volume_ax = fig.add_subplot(1, 2, 1)
        xlims = lims.get('xlim', (self.xmin, self.xmax))
        self.volume_ax.set_xlim(xlims)
        self.volume_ax.set_ylim((0, self.y_max))
        self.volume_ax.plot([0, 0], [
            -self.y_max, self.y_max], color='black', lw=0.5, ls='dashed')
        self.ask_bars = self.volume_ax.bar(
            self.X, self.get_ask_volumes(), align='edge', label='Ask', color='blue', width=0.1, animated='True')
        self.bid_bars = self.volume_ax.bar(
            self.X, self.get_bid_volumes(), align='edge', label='Bid', color='red', width=-0.1, animated='True')
        self.best_ask_axis, = self.volume_ax.plot(
            [], [], color='blue', ls='dashed', lw=1, label='best ask')
        self.best_bid_axis, = self.volume_ax.plot(
            [], [], color='red', ls='dashed', lw=1, label='best bid')
        self.volume_ax.set_title('Order volumes')
        self.volume_ax.legend()

    def init_animation(self):
        """Init function called by FuncAnimation
        """

        # Lines
        for b in self.ask_bars:
            b.set_height(0)
        for b in self.bid_bars:
            b.set_height(0)

        return [bar for bar in self.ask_bars] + [bar for bar in self.bid_bars]

    def update_animation(self, n):
        """Update function called by FuncAnimation
        """
        # Axis
        y_max = 1.5*self.dx * self.xmax * self.L

        self.timestep()

        lists = zip(self.ask_bars, self.bid_bars,
                    self.get_ask_volumes(), self.get_bid_volumes())
        for a_bar, b_bar, ask_vol, bid_vol in lists:
            a_bar.set_height(ask_vol)
            b_bar.set_height(bid_vol)

        # self.best_ask_axis.set_data([self.best_ask, self.best_ask], [0, y_max])
        # self.best_bid_axis.set_data([self.best_bid, self.best_bid], [0, y_max])

        return [bar for bar in self.ask_bars] + [bar for bar in self.bid_bars] + [self.best_ask_axis, self.best_bid_axis]
