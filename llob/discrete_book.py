import numpy as np

from limit_orders import LimitOrders


class DiscreteBook:
    """Models an order book in the framework of Latent Order Book agent model.
    """

    def __init__(self, **order_args):

        order_args = order_args
        self.bid_orders = LimitOrders(**order_args, side='bid')
        self.ask_orders = LimitOrders(**order_args, side='ask')

        # Get spatial values
        self.xmin = order_args['xmin']
        self.xmax = order_args['xmax']
        self.Nx = order_args['Nx']
        self.X, self.dx = np.linspace(
            self.xmin, self.xmax, num=self.Nx, retstep=True)

        # Elementary timestep
        self.D = order_args['D']
        self.dt = (self.dx)**2 / (2 * self.D)

        # Set prices
        self.update_price()

        # Animation
        self.y_max = max(np.max(self.get_bid_volumes()),
                         np.max(self.get_ask_volumes()))

    def get_ask_volumes(self):
        return self.ask_orders.volumes

    def get_bid_volumes(self):
        return self.bid_orders.volumes

    # ================== TIME EVOLUTION ==================

    def timestep(self, tstep, volume):
        """Step forward.
        """
        n_steps = int(tstep / self.dt)

        self.execute_metaorder(volume)
        self.update_price()
        for n in range(n_steps):
            self.stochastic_timestep()
            self.update_price()
            self.order_reaction()
            self.update_price()


    def stochastic_timestep(self):
        """Stochastic dynamics step.
        """
        for orders in [self.ask_orders, self.bid_orders]:
            orders.deposition()
            orders.cancellation()
            orders.jumps()

    def update_price(self):
        """Update order best price for both sides of the book.
        """
        for orders in [self.ask_orders, self.bid_orders]:
            orders.update_best_price()
        self.best_ask_index = self.ask_orders.best_price_index
        self.best_bid_index = self.bid_orders.best_price_index
        self.best_ask = self.X[self.best_ask_index - 1]
        self.best_bid = self.X[self.best_bid_index + 1]
        self.best_ask_volume = self.get_ask_volumes(
        )[self.ask_orders.best_price_index]
        self.best_bid_volume = self.get_bid_volumes(
        )[self.bid_orders.best_price_index]

    # ------------------ Reaction ------------------

    def order_reaction(self):
        """Order reaction step : matched orders are executed.
        """
        if self.best_ask_index > self.best_bid_index:
            return

        reaction_volumes = np.minimum(
            self.get_ask_volumes(), self.get_bid_volumes())
        for side_orders in [self.ask_orders, self.bid_orders]:
            side_orders.execute_orders(reaction_volumes)

    # ------------------ Metaorder ------------------

    def execute_metaorder(self, volume):
        """Execute a meta-order of a given volume.

        :param volume: trade volume, positive for ask order, negative for bid order
        :type volume: int
        """
        orders = self.ask_orders if volume > 0 else self.bid_orders
        orders.execute_best_orders(volume)

    def get_measure(self, quantity):
        value = getattr(self, quantity)
        return value

  # ================== ANIMATION ==================

    def set_animation(self, fig=None, lims=None):
        """Create subplot axes, lines and texts
        """
        # Ax
        self.volume_ax = fig.add_subplot(1, 2, 1)

        # Bars
        self.ask_bars = self.volume_ax.bar(
            self.X, self.get_ask_volumes(), align='edge', label='Ask', color='blue', width=0.1, animated='True')
        self.bid_bars = self.volume_ax.bar(
            self.X, self.get_bid_volumes(), align='edge', label='Bid', color='red', width=-0.1, animated='True')

        # Lines
        self.volume_ax.plot([0, 0], [
            -self.y_max, self.y_max], color='black', lw=0.5, ls='dashed')
        self.best_ask_axis, = self.volume_ax.plot(
            [], [], color='blue', ls='dashed', lw=1, label='best ask')
        self.best_bid_axis, = self.volume_ax.plot(
            [], [], color='red', ls='dashed', lw=1, label='best bid')

        # Settings
        xlims = lims.get('xlim', (self.xmin, self.xmax))
        self.volume_ax.set_xlim(xlims)
        self.volume_ax.set_ylim((0, self.y_max))
        self.volume_ax.set_title('Order volumes')
        self.volume_ax.legend()

    def init_animation(self):
        """Init function called by FuncAnimation
        """

        for b in self.ask_bars:
            b.set_height(0)
        for b in self.bid_bars:
            b.set_height(0)

        return [bar for bar in self.ask_bars] + [bar for bar in self.bid_bars]

    def update_animation(self, tstep, volume):
        """Update function called by FuncAnimation
        """

        self.timestep(tstep, volume)

        # Update bars
        for index in range(self.Nx):
            self.ask_bars[index].set_height(self.get_ask_volumes()[index])
            self.bid_bars[index].set_height(self.get_bid_volumes()[index])

        self.best_ask_axis.set_data(
            [self.best_ask, self.best_ask], [0, self.y_max])
        self.best_bid_axis.set_data(
            [self.best_bid, self.best_bid], [0, self.y_max])

        return [bar for bar in self.ask_bars] + [bar for bar in self.bid_bars] + [self.best_ask_axis, self.best_bid_axis]
