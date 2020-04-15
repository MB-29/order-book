import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from diffusion_schemes import theta_scheme_iteration


class OrderBook:

    def __init__(self, dt, D, L, mt=0, lower_bound=-1, upper_bound=1, Nx=100, diffusion_scheme='implicit'):
        """

        Arguments:
            dt {float} -- Time subinterval
            D {float} -- Diffusion constant
            L {float} -- Latent liquidity

        Keyword Arguments:
            mt {float} -- Metaorder trading intensity at current time
            lower_bound {float} -- Price interval lower bound
            upper_bound {float} -- Price interval upper bound
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
        self.mt = mt

        # Set prices
        self.price = (lower_bound + upper_bound)/2
        self.best_bid_index = Nx//2
        self.best_ask_index = Nx//2 + 1

        # Density function, which is solved
        self.density = self.initial_density(self.X)

    def update_best_ask(self):
        ask_indices = np.where(self.density < 0)[0]
        self.best_ask_index = ask_indices[0] if ask_indices.size > 0 else self.Nx-1
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
        Depending on the liquidity, the price is then shifted.

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
                       self.Nx-1, self.best_ask_index + 1)
                    if self.best_ask_index == self.Nx-1:
                        raise ValueError('Market lacks ask liquidity')
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
                    if self.best_ask_index == 0:
                        raise ValueError('Market lacks bid liquidity')
                else:
                    dq = 0
                    self.density[self.best_bid_index] += dq


    def initial_density(self, x):
        return -self.L * (x-self.price)

    def timestep(self):
        """
        Step forward
        """

        self.execute_metaorder()
        # Update density values with one iteration of the numerical scheme
        self.density = theta_scheme_iteration(
            self.density, self.dx, self.dt, self.D, self.L)
        self.update_prices()

    def __str__(self):
        parameters_string = f'D= {self.D}\nL = {self.L}\ndt= {self.dt}'
        return f'Order book with following parameters\n{parameters_string}'