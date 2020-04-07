import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from diffusion_schemes import theta_scheme_iteration


class OrderBook:

    def __init__(self, dt, D, lambd, nu, mt=0, lower_bound=-1, upper_bound=1, resolution=100, diffusion_scheme='implicit'):
        """

        Arguments:
            dt {float} -- Time subinterval
            D {float} -- Diffusion constant
            lambd {float} -- Deposition rate
            nu {float} -- Cancellation rate

        Keyword Arguments:
            mt {float} -- Metaorder trading intensity at current time
            lower_bound {float} -- Price interval lower bound
            upper_bound {float} -- Price interval upper bound
            resolution {int} -- Number of subintervals
        """

        # Structural constants
        self.dt = dt
        self.dx = (upper_bound - lower_bound)/float(resolution)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.X = np.linspace(lower_bound, upper_bound, num=resolution)
        self.N = resolution

        # Model constants
        self.D = D
        self.nu = nu
        self.lambd = lambd
        self.J = lambd * np.sqrt(D/float(nu))
        self.gamma = np.sqrt(nu/float(D))
        self.L = lambd * np.sqrt(1/float(nu*D))
        self.mt = mt

        # Density function, which is solved
        self.density = self.initial_density(self.X)

        # Set prices
        self.best_ask_index = self.N-1
        self.best_bid_index = 0
        self.update_prices()

    def update_best_ask(self):
        ask_indices = np.where(self.density < 0)[0]
        self.best_ask_index = ask_indices[0] if ask_indices.size > 0 else self.N-1
        self.best_ask = self.X[self.best_ask_index]

    def update_best_bid(self):
        bid_indices = np.where(self.density > 0)[0]
        self.best_bid_index = bid_indices[-1] if bid_indices.size > 0 else 0
        self.best_bid = self.X[self.best_bid_index]

    def update_prices(self):
        """ Update best ask, best bid and market price
        """
        self.best_ask = self.density[self.best_ask_index]
        self.best_bid = self.density[self.best_bid_index]
        self.update_best_ask()
        self.update_best_bid()
        self.price = (self.best_ask + self.best_bid)/2

    def execute_metaorder(self):
        """ Execute at current time the quantity dq = mt * dt at the best price which depends on the sign of mt.
        If mt > 0, then the metaorder is a buy and hence ask orders are executed, first at price best_ask.
        If mt < 0, then the metaorder is a sell and hence bid orders are executed, first at price best_bid.
        Depending on the liquidity, the price is then shifted 

        """

        dq = self.mt * self.dt
        if dq > 0:
            # liquidity > 0 is the absolute available volume at the best ask price
            while dq > 0:
                liquidity = -self.density[self.best_ask_index] * self.dx
                if dq > liquidity:
                    dq -= liquidity
                    self.density[self.best_ask_index] = 0
                    self.best_ask_index = min(
                        self.N-1, self.best_ask_index + 1)
                else:
                    dq = 0
                    self.density[self.best_ask_index] += dq
        else:
            while dq < 0:
                liquidity = self.density[self.best_bid_index] * self.dx
                if -dq > liquidity:
                    dq += liquidity
                    self.density[self.best_bid_index] = 0
                    self.best_ask_index = max(0, self.best_bid_index - 1)
                else:
                    dq = 0
                    self.density[self.best_bid_index] += dq

    def stationary_density(self, x):
        y = x-self.price
        return - np.sign(y) * (self.lambd/float(self.nu)) * (1-np.exp(-abs(y)*self.gamma))

    def initial_density(self, x):
        return -x**3

    def timestep(self):
        """
        Step forward
        """

        self.update_prices()
        self.execute_metaorder()

        # Update density values with one iteration of the numerical scheme
        self.density = theta_scheme_iteration(
            self.density, self.dx, self.dt, self.D, self.L)
