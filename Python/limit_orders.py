import numpy as np


class LimitOrders:
    """Orders of one side - bid or ask - with their volume depending on the price and stochastic dynamics.
    """

    def __init__(self, dt, lambd, nu, side, lower_bound, upper_bound, Nx=1000, L=None, **kwargs):
        """

        Arguments:
            dt {float} -- Time subinterval.
            lambd {float} -- Lambda parameter.
            nu {float} -- Nu parameter.
            side {string} -- Either 'BID' or ASK'

        Keyword Arguments:
            lower_bound {float} -- Price interval lower bound
            upper_bound {float} -- Price interval upper bound
            Nx {int} -- Number of space (price) subintervals
        """

        # Structural constants
        self.Nx = Nx
        self.dt = dt
        self.dx = (upper_bound - lower_bound)/float(Nx)
        self.X = np.linspace(lower_bound, upper_bound, num=Nx)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

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
        self.D = (self.dx)**2/(2*self.dt)
        self.L = lambd / np.sqrt(nu * self.D) if not L else L
        self.J = self.D * self.L

        self.initialize_volumes(initial_density)
        self.set_boundary_conditions(boundary_conditions)

    def initialize_volumes(self, initial_density):

        # Initialize price
        initial_price_index_dic = {
            'stationary': self.Nx//2,
            'linear': self.Nx//2,
            'empty': self.boundary_index % self.Nx
        }
        self.best_price_index = initial_price_index_dic.get(initial_density)
        self.best_price = self.X[self.best_price_index]

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
        self.best_price_volume = self.volumes[self.best_price_index - self.sign]

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

    def stochastic_timestep(self):
        self.order_arrivals()
        self.order_cancellation()
        self.order_jumps()

    def order_arrivals(self):
        """Process order deposition stochastic step.
        """

        self.update_best_price()

        # Orders are deposited at each side through a lambda intensity Poisson point process
        lam = self.lambd * self.dt * self.dx

        # Number of reachable price points for a given side
        size = self.Nx - self.best_price_index % self.Nx if self.side == 'ASK' else self.best_price_index + 1
        arrivals = np.random.poisson(lam=lam, size=size)

        # No orders are deposited on the rest of the points : pad with 0
        padding = (self.Nx - size,
                   0) if self.side == 'ASK' else (0, self.Nx - size)
        arrivals = np.pad(arrivals, padding,
                          mode='constant', constant_values=0)

        # Add deposited orders
        self.volumes += arrivals
        self.total_volume += np.sum(arrivals)
        return arrivals

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
        jumps = np.zeros((self.Nx, 2), dtype=int)
        for index, order_volume in enumerate(self.volumes):
            jumps_left = np.random.binomial(order_volume, 0.5)
            jumps[index, :] = [jumps_left, order_volume - jumps_left]

        boundary_volume = self.volumes[self.boundary_index] + \
            self.boundary_flow * (self.dx)**2
        boundary_jumps = np.random.binomial(boundary_volume, 0.5)

        # Set boundary flow
        boundary_jumps_left = boundary_jumps if self.side == 'ASK' else 0
        boundary_jumps_right = boundary_jumps if self.side == 'BID' else 0

        jumps_left = np.append(jumps[:, 0], boundary_jumps_left)
        jumps_right = np.insert(jumps[:, 1], 0, boundary_jumps_right)
        flow = jumps_right - jumps_left
        # flow[n] is the algebraic number of particles crossing from n-1 to n

        # update volumes : dV/dt = -dj/dx
        self.volumes = self.volumes - np.diff(flow)
        self.total_volume += flow[0] - flow[self.Nx]

    # def get_jumps(self, volume):
    #     """Compute the number of jumps for a given volume of orders, in a given direction.

    #     Arguments:
    #         volume {int} -- The volume of orders at a certain price

    #     Returns:
    #         Numpy array of size 2 -- [jumps left, jumps right]
    #     """

    #     # Random choice of jumps for each order : either jump left, stay or jump right
    #              self.jump_probabilities[-1],
    #              self.jump_probabilities[1]]

    #     return np.random.multinomial(volume, pvals=pvals)[:2]

    # ------------------ Price ------------------

    def update_best_price(self):
        end_index = 0 if self.side == 'ASK' else -1
        indices = np.nonzero(self.volumes)[0]
        if indices.size == 0:
            indices = [self.boundary_index]
        self.best_price_index = indices[end_index]
        self.best_price = self.X[self.best_price_index]
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
        self.update_best_price()

        index_increment = -self.sign

        if volume > self.total_volume:
            raise ValueError(f'{self.side} book lacks liquidity.')

        trade_volume = volume
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
        """Compute order volume between price of index price_index and best_price
        """
        price = self.X[price_index]
        if self.sign * (price - self.best_price) > 0:
            return 0
        lower = min(self.best_price_index, price_index)
        upper = max(self.best_price_index, price_index)
        return np.sum(self.volumes[lower: upper+1])
