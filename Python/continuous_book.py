import numpy as np
import warnings

from diffusion_schemes import theta_scheme_iteration


class ContinuousBook:
    """Models an order book with its order density in the LLOB framework.
    """

    resolution_volume = 1e-8

    def __init__(self, dt, D, L, lower_bound, upper_bound, Nx=1000):
        """

        Arguments:
            dt {float} -- Time subinterval
            D {float} -- Diffusion constant
            L {float} -- Latent liquidity
            lower_bound {float} -- Price interval lower bound
            upper_bound {float} -- Price interval upper bound

        Keyword Arguments:
            Nx {int} -- Number of space (price) subintervals
        """

        # Structural constants
        self.dt = dt
        self.dx = (upper_bound - lower_bound)/float(Nx)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.price_range = (upper_bound - lower_bound)/2
        self.X = np.linspace(lower_bound, upper_bound, num=Nx)
        self.Nx = Nx

        # Model constants
        self.D = D
        self.L = L
        self.J = D * L

        # Set prices
        self.price = (lower_bound + upper_bound)/2
        self.best_bid_index = Nx//2
        self.best_ask_index = Nx//2 + 1
        self.mt = 0

        # Density function
        if ContinuousBook.resolution_volume > self.L * self.dx * self.dx:
            warnings.warn(
                'Resolution volume may be too large and lead to an inaccurate price')

        self.density = self.initial_density(self.X)
        self.update_best_ask()
        self.update_best_bid()
        self.best_ask_density = self.density[self.best_ask_index + 1]
        self.best_bid_density = self.density[self.best_bid_index - 1]


    def initial_density(self, x):
        return -self.L * (x-self.price)

    # ================== Time evolution ==================

    def update_best_ask(self):
        ask_indices = np.where(self.density * self.dx <
                               - ContinuousBook.resolution_volume)[0]
        self.best_ask_index = ask_indices[0] if ask_indices.size > 0 else self.Nx-1
        self.best_ask = self.X[self.best_ask_index]

    def update_best_bid(self):
        bid_indices = np.where(self.density * self.dx >
                               ContinuousBook.resolution_volume)[0]
        self.best_bid_index = bid_indices[-1] if bid_indices.size > 0 else 0
        self.best_bid = self.X[self.best_bid_index]

    def update_prices(self):
        """ Update best ask, best bid and market price
        """
        self.update_best_ask()
        self.update_best_bid()
        self.price = (- self.best_ask * self.best_ask_density + self.best_bid *
                      self.best_bid_density)/(-self.best_ask_density + self.best_bid_density)

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
                    self.density[self.best_ask_index] += dq
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
                    self.best_ask_index -= 1
                    if self.best_ask_index == 0:
                        raise ValueError('Market lacks bid liquidity')
                else:
                    self.density[self.best_bid_index] += dq
                    dq = 0

        self.best_ask_density = self.density[self.best_ask_index]
        self.best_bid_density = self.density[self.best_bid_index]

    def timestep(self):
        """
        Step forward
        """

        self.execute_metaorder(self.mt * self.dt)
        # Update density values with one iteration of the numerical scheme
        self.density = theta_scheme_iteration(
            self.density, self.dx, self.dt, self.D, self.L)
        self.update_prices()

     # ================== ANIMATION ==================

    def set_animation(self, fig):
        """Create subplot axes, lines and texts
        """

        y_max = self.L * self.upper_bound

        self.density_ax = fig.add_subplot(1, 2, 1)
        self.density_ax.set_xlim((self.lower_bound, self.upper_bound))
        self.density_line, = self.density_ax.plot([], [], label='Density')
        self.price_axis, = self.density_ax.plot(
            [], [], label='Price', color='yellow', ls='dashed', lw=1)
        self.density_ax.plot([self.lower_bound, self.upper_bound], [
                             0, 0], color='black', lw=0.5, ls='dashed')
        self.density_ax.set_title('Algebraic order density')
        self.density_ax.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))
        self.density_ax.set_ylim(-y_max, y_max)

    def init_animation(self):
        """Init function called by FuncAnimation
        """

        self.density_line.set_data([], [])
        return [self.density_line]

    def update_animation(self, n):
        """Update function called by FuncAnimation
        """
        # Axis
        y_min, y_max = 0, 1.5 * self.upper_bound * self.L

        self.timestep()
        self.density_line.set_data(self.X, self.density)
        self.price_axis.set_data([self.price, self.price], [-y_max, y_max])
        return [self.price_axis, self.density_line]
