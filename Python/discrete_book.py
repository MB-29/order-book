import numpy as np

from limit_orders import LimitOrders


class DiscreteBook:
    """Models an order book in the framework of LLOB agent model.
    """

    def __init__(self, **order_args):

        self.order_args = order_args
        self.bid_orders = LimitOrders(**order_args, side='BID')
        self.ask_orders = LimitOrders(**order_args, side='ASK')

        self.dt = self.order_args['dt']
        self.lower_bound = self.order_args['lower_bound']
        self.upper_bound = self.order_args['upper_bound']
        self.n_relax = self.order_args.get('n_relax', 1)
        self.Nx = self.order_args['Nx']
        self.X = np.linspace(self.lower_bound, self.upper_bound, num=self.Nx)
        self.dx = (self.upper_bound - self.lower_bound)/float(self.Nx)
        self.lambd = self.order_args['lambd']
        self.nu = self.order_args['nu']
        self.D = (self.dx)**2/(2*self.dt)

        # Theoretical values
        self.L = self.order_args['L']
        if not self.L:
            self.L = self.lambd/(np.sqrt(self.nu * self. D))
        self.J = self.L * self.D
        self.price_range = self.upper_bound - self.lower_bound

        # Metaorder
        self.update_price()
        self.dq = 0

    def stationary_density(self, x):

        y = abs(x - self.price)
        x_crit = np.sqrt(self.D/self.nu)
        return (self.lambd/self.nu) * (1 - np.exp(-y/x_crit))

    # ================== TIME EVOLUTION ==================

    def timestep(self):

        self.execute_metaorder(self.dq)
        for t in range(self.n_relax):
            self.stochastic_timestep()
            self.order_reaction()
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
        self.best_ask = self.X[self.ask_orders.best_price_index]
        self.best_bid = self.X[self.bid_orders.best_price_index]
        self.price_index = (self.ask_orders.best_price_index +
                            self.bid_orders.best_price_index)//2
        self.price = self.X[self.price_index]

    # ------------------ Reaction ------------------

    def order_reaction(self):
        """Reaction step : tradable orders are executed
        """
        best_ask_index = self.ask_orders.best_price_index
        best_bid_index = self.bid_orders.best_price_index
        if best_ask_index > best_bid_index:
            return

        # Tradable orders are those in price range [best_ask, best_bid]
        traded_volume = min(self.ask_orders.get_available_volume(
            best_bid_index), self.bid_orders.get_available_volume(best_ask_index))
        for orders in [self.ask_orders, self.bid_orders]:
            orders.consume_best_orders(traded_volume)

    # ------------------ Metaorder ------------------

    def execute_metaorder(self, volume):
        orders = self.ask_orders if volume > 0 else self.bid_orders
        orders.consume_best_orders(volume)

  # ================== ANIMATION ==================

    def set_animation(self, fig, lims):
        """Create subplot axes, lines and texts
        """

        self.volume_ax = fig.add_subplot(1, 2, 1)
        xlims = lims.get('xlim', (self.lower_bound, self.upper_bound))
        y_max = 1.5 * xlims[1] * self.L * self.dx
        self.volume_ax.set_xlim(xlims)
        self.volume_ax.set_ylim((0, y_max))
        self.ask_bars = self.volume_ax.bar(
            self.X, self.ask_orders.volumes, label='Ask', color='blue', width=0.1, animated='True')
        self.bid_bars = self.volume_ax.bar(
            self.X, self.bid_orders.volumes, label='Bid', color='red', width=0.1, animated='True')
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
        y_max = 1.5*self.dx * self.upper_bound * self.L

        self.timestep()

        for index in range(self.Nx):
            self.ask_bars[index].set_height(self.ask_orders.volumes[index])
            self.bid_bars[index].set_height(self.bid_orders.volumes[index])

        self.best_ask_axis.set_data([self.best_ask, self.best_ask], [0, y_max])
        self.best_bid_axis.set_data([self.best_bid, self.best_bid], [0, y_max])

        return [bar for bar in self.ask_bars] + [bar for bar in self.bid_bars] + [self.best_ask_axis, self.best_bid_axis]
