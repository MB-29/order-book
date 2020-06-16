import numpy as np
import warnings

from diffusion_schemes import theta_scheme_iteration


class ContinuousBook:
    """Models an order book with its order density in the LLOB framework.
    """

    def __init__(self, dt, D, L, lambd, nu, xmin, xmax, Nx=1000, **kwargs):
        """
        :param dt: Size of time subinterval
        :type dt: float
        :param D: Diffusion constant
        :type D: float
        :param L: Order density slope
        :type L: float
        :param lambd: Lambda parameter
        :type lambd: float
        :param nu: Nu parameter
        :type nu: float
        :param xmin: Price interval lower bound
        :type xmin: float
        :param xmax: Price interval upper bound
        :type xmax: float
        :param Nx: Number of space (price) subintervals, defaults to 1000
        :type Nx: int, optional
        """

        # Structural constants
        self.dt = dt
        self.xmin = xmin
        self.xmax = xmax
        self.price_range = (xmax - xmin)/2
        self.boundary_distance = min(
            abs(self.xmin), abs(self.xmax))
        self.X, self.dx = np.linspace(xmin, xmax, num=Nx, retstep=True)
        self.Nx = Nx

        # Model constants
        self.D = D
        self.L = L
        self.J = D * L
        self.resolution_volume = L * (self.dx)**2
        # Density function
        if self.resolution_volume > self.L * self.dx * self.dx:
            warnings.warn(
                'Resolution volume may be too large and lead to an inaccurate price')

        self.density = self.initial_density(self.X)
        self.update_prices()
        # +1 / -1 ensure corresponding volume isn't partially consumed
        self.best_ask_volume = self.density[self.best_ask_index + 1]
        self.best_bid_volume = self.density[self.best_bid_index - 1]

        # Metaorder
        self.dq = 0

    def initial_density(self, x):
        return -self.L * x

    # ================== Time evolution ==================

    def update_prices(self):
        """ Update best ask, best bid and market price
        """
        bid_indices = np.where(self.density * self.dx >
                               self.resolution_volume)[0]
        ask_indices = np.where(self.density * self.dx <
                               - self.resolution_volume)[0]
        self.best_ask_index = ask_indices[0] if ask_indices.size > 0 else self.Nx-1
        self.best_bid_index = bid_indices[-1] if bid_indices.size > 0 else 0

        self.best_ask = self.X[self.best_ask_index - 1]
        self.best_bid = self.X[self.best_bid_index + 1]

    def execute_metaorder(self, volume):
        """ Execute at current time the quantity volume at the best price which depends on the sign of volume.
        If volume > 0, then the metaorder is a buy and hence ask orders are executed at price best_ask.
        If volume < 0, then the metaorder is a sell and hence bid orders are executed at price best_bid.
        As liquidity vanishes the best price is increasingly shifted.
        """

        if volume == 0:
            return

        dq = volume
        if dq > 0:
            while dq > 0:
                # liquidity > 0 is the absolute available volume at the best ask price
                liquidity = -self.density[self.best_ask_index] * self.dx
                if dq < liquidity:
                    self.density[self.best_ask_index] += dq/self.dx
                    dq = 0
                    break

                self.density[self.best_ask_index] = 0
                self.best_ask_index += 1
                dq -= liquidity
                if self.best_ask_index > self.Nx-1:
                    raise ValueError('Market lacks ask liquidity')
        else:
            while dq < 0:
                liquidity = self.density[self.best_bid_index] * self.dx
                if -dq > liquidity:
                    dq += liquidity
                    self.density[self.best_bid_index] = 0
                    self.best_bid_index -= 1
                    if self.best_bid_index == 0:
                        raise ValueError('Market lacks bid liquidity')
                else:
                    self.density[self.best_bid_index] += dq/self.dx
                    dq = 0

        self.best_ask_volume = self.density[self.best_ask_index + 1]
        self.best_bid_volume = self.density[self.best_bid_index - 1]

    def timestep(self):
        """
        Step forward
        """

        self.execute_metaorder(self.dq)
        # Update density values with one iteration of the numerical scheme
        self.update_prices()
        if self.D != 0:
            self.density = theta_scheme_iteration(
                self.density, self.dx, self.dt, self.D, self.L)
        self.update_prices()

    # ================== ANIMATION ==================

    def set_animation(self, fig, lims):
        """Create subplot axes, lines and texts
        """
        xlims = lims.get('xlim', (self.xmin, self.xmax))
        y_max = self.L * xlims[1]

        self.density_ax = fig.add_subplot(1, 2, 1)
        self.density_ax.set_xlim(xlims)
        self.density_line, = self.density_ax.plot(
            [], [], label='Density', color='gray')
        self.best_ask_axis, = self.density_ax.plot(
            [], [], color='blue', ls='dashed', lw=1, label='best ask')
        self.best_bid_axis, = self.density_ax.plot(
            [], [], color='red', ls='dashed', lw=1, label='best bid')
        self.density_ax.plot([self.xmin, self.xmax], [
                             0, 0], color='black', lw=0.5, ls='dashed')
        self.density_ax.plot([0, 0], [
                             -y_max, y_max], color='black', lw=0.5, ls='dashed')
        self.density_ax.set_title('Algebraic order density')
        self.density_ax.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))
        self.density_ax.set_ylim(-y_max, y_max)

    def init_animation(self):
        """Init function called by FuncAnimation
        """

        self.density_line.set_data([], [])
        return [self.density_line, self.best_ask_axis, self.best_bid_axis]

    def update_animation(self, n):
        """Update function called by FuncAnimation
        """
        # Axis
        y_max = 1.5 * self.xmax * self.L

        self.timestep()
        self.density_line.set_data(self.X, self.density)
        self.best_ask_axis.set_data(
            [self.best_ask, self.best_ask], [-y_max, y_max])
        self.best_bid_axis.set_data(
            [self.best_bid, self.best_bid], [-y_max, y_max])
        return [self.best_ask_axis, self.best_bid_axis, self.density_line]
