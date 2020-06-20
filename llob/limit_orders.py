import numpy as np
from numba import njit, int64, float64

use_numba = True
# use_numba = False


class LimitOrders:
    """Orders volumes of one side - bid or ask - with stochastic dynamics."""

    def __init__(self, lambd, nu, D, side, xmin, xmax, Nx, L=None, **kwargs):
        """     
        :param lambd: Lambda parameter
        :type lambd: float
        :param nu: Nu parameter
        :type nu: float
        :param side: Either 'bid' or ask'
        :type side: string
        :param xmin: Price interval lower bound
        :type xmin: float
        :param xmax: Price interval upper bound
        :type xmax: float
        :param L: Order density slope, defaults to None
        :type L: float, optional
        :param Nx: Number of space (price) subintervals, defaults to 1000
        :type Nx: int, optional
        """

        # Structural constants
        self.Nx = Nx
        self.X, self.dx = np.linspace(xmin, xmax, num=Nx, retstep=True)
        self.xmin = xmin
        self.xmax = xmax

        # Smoluchowski random walk : D =  dx**2 / (2 * dt)
        self.dt = (self.dx)**2 / (2 * D)

        initial_density = kwargs.get('initial_density', 'stationary')
        boundary_conditions = kwargs.get('boundary_conditions', 'flat')

        # Order side
        self.side = side
        assert (side in ['ask', 'bid'])
        self.sign = -1 if side == 'ask' else 1
        self.boundary_index = -1 if side == 'ask' else 0

        # Model parameters
        self.lambd = lambd
        self.nu = nu
        self.D = D
        self.L = L if L != None else lambd / np.sqrt(nu * self.D)

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

    def deposition(self):
        """Process order deposition stochastic step.
        In order to work properly, method update_price() should be called before this one,
        so that the size of the deposition price range is correct.
        """

        # Orders are deposited at each side through a lambda intensity Poisson point process
        lam = self.lambd * self.dt * self.dx

        # Number of arrival points for a given side
        size = self.Nx - self.best_price_index % self.Nx if self.side == 'ask' else self.best_price_index + 1
        padding_size = size - self.Nx if self.side == 'ask' else self.Nx - size
        if use_numba:
            self.volumes = add_arrivals(self.volumes, lam, size, padding_size)
            return
        arrivals = np.random.poisson(lam=lam, size=size)

        # No orders are deposited on the rest of the points : pad with 0
        padding = (self.Nx - size,
                   0) if self.side == 'ask' else (0, self.Nx - size)
        arrivals = np.pad(arrivals, padding,
                          mode='constant', constant_values=0)

        # Add deposited orders
        self.volumes += arrivals

    def cancellation(self):
        """Process order cancellation stochastic step."""
        scale = 1/self.nu
        if use_numba:
            self.volumes = substract_cancellations(
                self.volumes, scale, self.dt)
            return

        # Vectorize get_cancellation function and apply it to price volumes array
        get_cancellation_vec = np.vectorize(
            lambda volume: self.get_cancellation(volume, scale))
        cancellations = get_cancellation_vec(self.volumes)

        # Subtract cancelled orders
        self.volumes = np.subtract(self.volumes, cancellations)

    def get_cancellation(self, volume, scale):
        """Compute the number of cancellations for a given volume of orders,
        according to an exponential law of given scale.

        """
        if volume == 0:
            return 0
        life_times = np.random.exponential(scale=scale, size=volume)
        cancellations = np.where(life_times < self.dt, 1, 0)
        return np.sum(cancellations)

    def jumps(self):
        """Process order jumps stochastic step."""

        # 2D array where rows correspond to the price range, and column are respectively
        # the number of jumps left and jumps right
        if use_numba:
            self.volumes = add_flow(self.volumes, self.dx,
                                    self.boundary_index, self.boundary_flow)
            return
        jumps = np.zeros((self.Nx, 2), dtype=int)
        for index, order_volume in enumerate(self.volumes):
            jumps_left = np.random.binomial(order_volume, 0.5)
            jumps[index, :] = [jumps_left, order_volume - jumps_left]

        boundary_volume = self.volumes[self.boundary_index] + \
            self.boundary_flow * (self.dx)**2
        boundary_jumps = np.random.binomial(boundary_volume, 0.5)

        # Set boundary flow
        boundary_jumps_left = boundary_jumps if self.side == 'ask' else 0
        boundary_jumps_right = boundary_jumps if self.side == 'bid' else 0
        jumps_left = np.append(jumps[:, 0], boundary_jumps_left)
        jumps_right = np.insert(jumps[:, 1], 0, boundary_jumps_right)
        flow = jumps_right - jumps_left
        # flow[n] is the algebraic number of particles crossing from n-1 to n

        # update volumes : dV/dt = -dj/dx
        self.volumes = self.volumes - np.diff(flow)

    # ------------------ Price ------------------

    def update_best_price(self):
        end_index = 0 if self.side == 'ask' else -1
        indices = np.nonzero(self.volumes)[0]
        if indices.size == 0:
            indices = [self.boundary_index]
        self.best_price_index = indices[end_index]
        self.best_price = self.X[self.best_price_index]
        self.best_price_volume = self.volumes[self.best_price_index]
        return self.best_price_index

    def execute_best_orders(self, volume):
        """Consume a given quantity of orders at the book's best price, which is updated at each step

        :param volume: Order volume to execute
        :type volume: int
        :raises ValueError: In the case the book lacks liquidity

        """

        if volume == 0:
            return
        if self.best_price_index in {0, self.Nx}:
            raise ValueError(f'Market lacks {self.side} liquidity')

        index_increment = -self.sign
        trade_volume = abs(volume)

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

    def execute_orders(self, volumes):
        """Execute given order volumes within the limit
            of the book volumes

        :param volumes: volumes to execute
        :type volumes: array of floats
        :return: actually executed volumes
        :rtype: array
        """

        trade_volumes = np.minimum(volumes, self.volumes)
        self.volumes -= trade_volumes.astype(int)
        return trade_volumes

    def get_available_volume(self, price_index):
        """Compute order volume between prices of indices price_index and self.best_price

        """
        price = self.X[price_index]
        if self.sign * (price - self.best_price) > 0:
            return 0
        lower = min(self.best_price_index, price_index)
        upper = max(self.best_price_index, price_index)
        return np.sum(self.volumes[lower: upper+1])

# ================== NUMBA-OPTIMIZED FUNCTIONS ==================


@njit(int64[:](int64[:], float64, int64, float64))
def add_flow(volumes, dx, boundary_index, boundary_flow):
    """Numba-accelerated function that computes order jump flow.

    :param volumes: Order volumes
    :type volumes: int64[:]
    :param dx: Size of the price subinterval
    :type dx: float64
    :param boundary_index: Depending on the size ask or bid,
    :type boundary_index: int64
    :param boundary_index: The algebraic index of the exterior boundary.
        -1 or O depending on the size ask or bid
    :type boundary_index: int64
    :param boundary_flow: The corresponding flow
    :type boundary_flow: float64
    :returns: Updated volumes
    :rtype: int64[:]

    """
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
    jumps_left = np.concatenate(
        (jumps[:, 0], np.array([boundary_jumps_left], dtype=int64)))
    jumps_right = np.concatenate(
        (np.array([boundary_jumps_right], dtype=int64), jumps[:, 1]))
    flow = jumps_right - jumps_left
    # flow[n] is the algebraic number of particles crossing from n-1 to n

    # update volumes according to dV/dt = -dj/dx
    volumes -= np.diff(flow)
    return volumes


@njit(int64[:](int64[:], float64, int64, int64))
def add_arrivals(volumes, lam, size, padding_size):
    """Numba-accelerated function that computes and adds the arrivals of the orders, which 
    are deposited with a Poissonian law in the corresponding price range.

    :param volumes: Order volumes
    :type volumes: int64[:]
    :param lam: Deposition intensity
    :type lam: float64
    :param size: Size of the deposition price range.
    :type size: int64[:]
    :param padding_size: Complementary size. Its sign indicates whether
        padding should be applied before or after the deposition price range.
    :type padding_size: int64
    :returns: Updated volumes
    :rtype: int64[:]

    """

    # Number of arrival points for a given side
    arrivals = np.random.poisson(lam=lam, size=size)
    padding = np.zeros(abs(padding_size), dtype=int64)
    arrays = (padding, arrivals) if padding_size < 0 else (arrivals, padding)
    arrivals = np.concatenate(arrays)
    return np.add(volumes, arrivals)


@njit(int64[:](int64[:], float64, float64))
def substract_cancellations(volumes, scale, dt):
    """Numba-accelerated function that computes the cancellations of the orders
    according to an exponential law.

    :param volumes: Order volumes
    :type volumes: int64[:]
    :param scale: time scale for exponential law
    :type scale: float64
    :param dt: size of time subinterval
    :type dt: float64
    :returns: Updated volumes
    :rtype: int64[:]

    """
    total_volume = np.sum(volumes)
    # Draw exponential times once for all orders for performance matters
    life_times = np.random.exponential(scale=scale, size=total_volume)
    cancellations = np.zeros(volumes.size, dtype=int64)
    volume_index = 0
    # Compare progressively dt to the drawn exponential times, price by price
    # One loop performs the comparison for the the orders of one price subinterval,
    # whose count is order_volume
    for index, order_volume in enumerate(volumes):
        cancellations[index] = np.sum(
            np.where(life_times[volume_index: volume_index+order_volume] < dt, 1, 0))
        volume_index += order_volume
    return volumes - cancellations
