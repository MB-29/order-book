import numpy as np
from numba import njit, int64, float64


class LimitOrders:
    """Orders of one side - bid or ask - with their volume depending on the price and stochastic dynamics.
    """

    def __init__(self, dt, lambd, nu, side, xmin, xmax, Nx=1000, L=None, **kwargs):
        """

        Arguments:
            dt {float} -- Time subinterval.
            lambd {float} -- Lambda parameter.
            nu {float} -- Nu parameter.
            side {string} -- Either 'BID' or ASK'
            xmin {float} -- Price interval lower bound
            xmax {float} -- Price interval upper bound

        Keyword Arguments:
            Nx {int} -- Number of space (price) subintervals
        """

        # Structural constants
        self.Nx = Nx
        self.X, self.dx = np.linspace(xmin, xmax, num=Nx, retstep=True)
        self.xmin = xmin
        self.xmax = xmax
        self.dt = dt
        self.t_diff = dt / kwargs['n_diff']

        initial_density = kwargs.get('initial_density', 'stationary')
        boundary_conditions = kwargs.get('boundary_conditions', 'flat')

        # Order side
        self.side = side
        assert (side in ['ASK', 'BID'])
        self.sign = -1 if side == 'ASK' else 1
        self.boundary_index = -1 if side == 'ASK' else 0

        # Model parameters
        self.lambd = lambd
        self.nu = nu
        self.D = (self.dx)**2/(2*self.t_diff)
        self.L = L

        self.initialize_volumes(initial_density)
        self.set_boundary_conditions(boundary_conditions)

    def initialize_volumes(self, initial_density):

        # Initialize volume
        self.volumes = np.zeros(self.Nx, dtype=int)
        initial_densities_dic = {
            'stationary': self.stationary_density,
            'linear': lambda x: self.L * abs(x) if self.sign * x <= 0 else 0,
            'empty': lambda x: 0
        }
        density_function = initial_densities_dic.get(initial_density)
        volumes_function = np.vectorize(
            lambda x: int(self.dx * density_function(x)))
        self.volumes = volumes_function(self.X)

        self.total_volume = np.sum(self.volumes)
        self.update_best_price()
        self.best_price_volume = self.volumes[self.best_price_index]

    def stationary_density(self, x):
        if self.sign * x > 0:
            return 0
        x_crit = np.sqrt(self.D/self.nu)
        return (self.lambd/self.nu) * (1 - np.exp(-abs(x)/x_crit))

    def set_boundary_conditions(self, boundary_conditions):

        boundary_flow_dic = {
            'flat': 0,
            'linear': self.L
        }
        boundary_flow = boundary_flow_dic.get(boundary_conditions)
        self.boundary_flow = boundary_flow

    # ================== TIME EVOLUTION ==================

    # ------------------ Stochastic evolution ------------------

    def order_arrivals(self):
        """Process order deposition stochastic step.
        In order to work properly, method update_price() should be called before this one,
        so that the size of the arrivals vector is correct.
        """

        # Orders are deposited at each side through a lambda intensity Poisson point process
        lam = self.lambd * self.dt * self.dx

        # Number of arrival points for a given side
        size = self.Nx - self.best_price_index % self.Nx if self.side == 'ASK' else self.best_price_index + 1
        padding_size = size - self.Nx if self.side == 'ASK' else self.Nx - size
        # arrivals = np.random.poisson(lam=lam, size=size)
        # arrivals = get_arr(lam, size)

        # No orders are deposited on the rest of the points : pad with 0
        # padding = (self.Nx - size,
        #            0) if self.side == 'ASK' else (0, self.Nx - size)
        # arrivals = np.pad(arrivals, padding,
        #                   mode='constant', constant_values=0)
        self.volumes = add_arrivals(self.volumes, lam, size, padding_size)

        # Add deposited orders
        self.total_volume = np.sum(self.volumes)
        # return arrivals

    def order_cancellation(self):
        """Process order cancellation stochastic step.
        """
        scale = 1/self.nu

        # Vectorize get_cancellation function and apply it to price volumes array
        get_cancellation_vec = np.vectorize(
            lambda volume: self.get_cancellation(volume, scale))
        cancellations = get_cancellation_vec(self.volumes)

        # Subtract cancelled orders
        self.volumes = np.subtract(self.volumes, cancellations)
        self.total_volume -= np.sum(cancellations)
        return cancellations

    def get_cancellation(self, volume, scale):
        """Compute the number of cancellations for a given volume of orders,
        according to an exponential law of given scale.
        """
        if volume == 0:
            return 0
        life_times = np.random.exponential(scale=scale, size=volume)
        cancellations = np.where(life_times < self.dt, 1, 0)
        return np.sum(cancellations)

    def order_jumps(self):
        """Process order jumps stochastic step.
        """

        # 2D array where rows correspond to the price range, and column are respectively
        # the number of jumps left and jumps right
        self.volumes = add_flow(self.volumes, self.dx,
                        self.boundary_index, self.boundary_flow)
        # flow[n] is the algebraic number of particles crossing from n-1 to n

        # update volumes : dV/dt = -dj/dx
        # self.volumes = self.volumes - np.diff(flow)
        # self.total_volume += flow[0] - flow[self.Nx]

    # ------------------ Price ------------------

    def update_best_price(self):
        end_index = 0 if self.side == 'ASK' else -1
        indices = np.nonzero(self.volumes)[0]
        if indices.size == 0:
            indices = [self.boundary_index]
        self.best_price_index = indices[end_index]
        self.best_price = self.X[self.best_price_index]
        self.best_price_volume = self.volumes[self.best_price_index]
        return self.best_price_index

    def consume_best_orders(self, volume):
        """Consume a given quantity of orders at the book's best price, which is updated at each step

        Arguments:
            volume {int} -- Quantity of orders to be consumed

        Raises:
            ValueError: In the case the book lacks liquidity
        """

        if volume == 0:
            return

        index_increment = -self.sign

        trade_volume = abs(volume)
        if trade_volume > self.total_volume:
            raise ValueError(f'{self.side} book lacks liquidity.')

        while trade_volume > 0:
            liquidity = self.volumes[self.best_price_index]
            if trade_volume < liquidity:
                self.volumes[self.best_price_index] -= trade_volume
                break

            self.volumes[self.best_price_index] = 0
            self.best_price_index += index_increment
            trade_volume -= liquidity

            if self.best_price_index > self.Nx - 1 or self.best_price_index < 0:
                raise ValueError(f'Market lacks {self.side} liquidity')

        self.total_volume -= volume
        self.best_price = self.X[self.best_price_index]
        self.best_price_volume = self.volumes[self.best_price_index]

    def get_available_volume(self, price_index):
        """Compute order volume between prices of indices price_index and best_price
        """
        price = self.X[price_index]
        if self.sign * (price - self.best_price) > 0:
            return 0
        lower = min(self.best_price_index, price_index)
        upper = max(self.best_price_index, price_index)
        return np.sum(self.volumes[lower: upper+1])


@njit(int64[:](int64[:], float64, int64, float64))
def add_flow(volumes, dx, boundary_index, boundary_flow):
    Nx = len(volumes)
    # 2D array where rows correspond to the price range, and column are respectively
    # the number of jumps left and jumps right
    jumps = np.zeros((Nx, 2), dtype=int64)
    for index, order_volume in enumerate(volumes):
        jumps_left = np.random.binomial(order_volume, 0.5)
        jumps[index, :] = [jumps_left, order_volume - jumps_left]

    boundary_volume = volumes[boundary_index] + boundary_flow * (dx)**2
    boundary_jumps = np.random.binomial(boundary_volume, 0.5)

    # Set boundary flow
    boundary_jumps_left = boundary_jumps if boundary_index == -1 else 0
    boundary_jumps_right = boundary_jumps if boundary_index == 0 else 0
    jumps_left = np.concatenate((jumps[:, 0], np.array([boundary_jumps_left], dtype=int64)))
    jumps_right = np.concatenate( (np.array([boundary_jumps_right], dtype=int64), jumps[:, 1]) )
    return jumps_right - jumps_left
    # flow[n] is the algebraic number of particles crossing from n-1 to n


@njit(int64[:](int64[:], float64, int64, int64))
def add_arrivals(volumes, lam, size, padding_size):

    # Number of arrival points for a given side
    arrivals = np.random.poisson(lam=lam, size=size)
    padding = np.zeros(abs(padding_size), dtype=int64)
    arrays = (padding, arrivals) if padding_size < 0 else (arrivals, padding)
    arrivals = np.concatenate(arrays)

    return np.add(volumes, arrivals)

@njit(int64[:](float64, int64))
def get_arr(lam, size):

    # Number of arrival points for a given side
    return np.random.poisson(lam=lam, size=size)
