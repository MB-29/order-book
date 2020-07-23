import numpy as np
from matplotlib.animation import FuncAnimation, writers
import warnings

from linear_discrete_book import LinearDiscreteBook
from discrete_book import DiscreteBook
from linear_continuous_book import LinearContinuousBook
from multi_discrete_book import MultiDiscreteBook


class Simulation:
    """Implements a simulation of Latent Order Book time evolution.
    """

    def __init__(self, model_type, metaorder=[0], **kwargs):
        """
        :param model_type: 'discrete' or 'continuous'
        :type model_type: string
        :param metaorder: Meta-order intensity over time, defaults to [0].
            If its length is 1 then the it will be converted to
            a constant meta-order with the corresponding value.
        :type metaorder: list, optional
        """

        # Model type
        assert model_type in ['discrete', 'continuous']
        self.model_type = model_type

        # Time
        self.T = kwargs.get('T', 1)
        self.Nt = kwargs.get('Nt', 100)
        self.time_interval, self.tstep = np.linspace(
            0, self.T, num=self.Nt, retstep=True)

        # Space
        self.xmin = kwargs['xmin']
        self.xmax = kwargs['xmax']
        self.price_range = self.xmax - self.xmin
        self.Nx = kwargs.get('Nx', 100)
        self.dx = self.price_range / self.Nx
        self.boundary_distance = min(abs(self.xmin), self.xmax)

        # Prices and measures
        self.asks = np.zeros(self.Nt)
        self.bids = np.zeros(self.Nt)
        self.prices = np.zeros(self.Nt)
        self.measured_quantities = kwargs.get('measured_quantities', [])
        self.measurement_indices = kwargs.get('measurement_indices', [])
        self.measurements = {}
        for quantity in self.measured_quantities:
            self.measurements[quantity] = []

        # Orders
        self.L = kwargs['L']
        self.is_multi_book = (not np.isscalar(self.L))
        self.D = kwargs['D']
        self.nu = kwargs.get('nu', 0)
        self.lambd = self.L * np.sqrt(self.nu * self.D)
        self.J = self.D * self.L
        self.book = self.set_order_book(model_type, kwargs)

        # Price
        self.price_formula = kwargs.get('price_formula', 'middle')
        price_formula_choice = {
            'middle': lambda a, b: (a+b)/2,
            'best_ask': lambda a, b: a,
            'best_bid': lambda a, b: b,
            'vwap': lambda a, b: self.compute_vwap(a, b)
        }
        self.compute_price = price_formula_choice[self.price_formula]

        # Meta-order
        self.n_start = kwargs.get('n_start', 0)
        self.n_end = kwargs.get('n_end', self.Nt)

        if len(metaorder) == 1:
            self.m0 = metaorder[0]
            self.metaorder = np.zeros(self.Nt)
            self.metaorder[self.n_start:self.n_end].fill(self.m0)
        else:
            assert len(metaorder) == self.Nt
            self.metaorder = metaorder
            self.m0 = metaorder.mean()

        self.t_start = self.n_start * self.tstep
        self.t_end = self.n_end * self.tstep
        self.time_interval_shifted = self.time_interval-self.t_start

        self.theoretical_values()

    def set_order_book(self, model_type, args):
        """Create an instance of an order book with the given parameters

        :param model_type: 'discrete' or 'continuous'
        :type model_type: string
        :param args: book args, see documentation of the corresponding
            order book class 
        :type args: dictionary
        :returns: the instance of the order book
        :rtype: OrderBook object
        """

        args['lambd'] = self.lambd

        # If L is an array, create a multi-actor book
        if self.is_multi_book:
            return MultiDiscreteBook(**args)

        linear = (self.nu == 0)
        model_name = 'linear_' + model_type if linear else model_type

        model_choice = {
            'discrete': DiscreteBook,
            'linear_discrete': LinearDiscreteBook,
            'linear_continuous': LinearContinuousBook,
        }

        return model_choice.get(model_name)(**args)
    
    def get_density(self):
        return {'bid': self.book.get_ask_volumes(), 'ask': self.book.get_bid_volumes()}

    # ================== THEORY ==================

    def theoretical_values(self):
        # Theoretical values

        # Take the dominant slope in the case of a multi-actor book
        L = np.max(self.L)
        self.dt = (self.dx ** 2) / (2 * self.D)
        self.n_steps = int(self.tstep / self.dt)
        self.boundary_factor = np.sqrt(self.D*self.T)/(self.boundary_distance)
        self.infinity_density = L * self.xmax
        self.impact_th = np.sqrt(2*abs(self.m0)*self.T/L)
        self.density_shift_th = np.sqrt(
            abs(self.m0)*self.T*L)  # impact_th * L
        self.participation_rate = self.m0 / \
            (self.D * L) if self.D != 0 else float("inf")
        self.r = abs(self.participation_rate)
        self.scheme_constant = self.D * self.tstep / (self.dx * self.dx)
        self.lower_impact = np.sqrt(
            abs(self.r)/(2*np.pi)) * self.impact_th
        self.first_volume = L * self.dx * self.dx

        # Warnings and errors

        if self.boundary_factor > 1:
            warnings.warn('Boundary effects')
        if self.model_type == 'discrete':
            if self.r < 1 and self.n_steps < 100:
                warnings.warn(
                    f'Low number of diffusion steps {self.n_steps} < 100,'
                    'try increasing spatial resolution.')
            if self.n_steps < 1 and self.r < float('inf'):
                raise ValueError(
                    f'Order diffusion is not possible because diffusion distance is smaller that space subinterval.'
                    'Try decreasing participation rate'
                    f' dt = {self.dt}, tstep = {self.tstep}')
            if self.n_steps > 200:
                raise ValueError(
                    f'Many diffusion steps : ~ {int(self.n_steps)},'
                    f' dt = {self.dt}, tstep = {self.tstep}')

    def compute_vwap(self, best_ask, best_bid):
        """Apply vwap formula
        """
        return (abs(self.book.best_ask_volume) * best_ask
                + abs(self.book.best_bid_volume) * best_bid)/(
                    abs(self.book.best_ask_volume) + abs(self.book.best_bid_volume))

    def get_growth_th(self):
        """Return theoretical price impact, starting from price 0
        """
        A = self.m0/(self.L*np.sqrt(self.D * np.pi)
                     ) if self.r < 1 else np.sqrt(2)*np.sqrt(self.m0/self.L)
        growth = A * \
            np.sqrt(self.time_interval_shifted[self.n_start: self.n_end])
        return growth

    # ================== RUN ==================

    def run(self, fig=None, animation=False, save=False):
        """Run the Nt steps of the simulation

        :param fig: The figure the animation is displayed in, defaults to None
        :type fig: matplotlib figure, optional
        :param animation: Set True to display an animation, defaults to False
        :type animation: bool, optional
        :param save: Set True to save the animation under ./animation.mp4, defaults to False
        :type save: bool, optional
        """

        if animation:
            self.run_animation(fig, save)
            return

        for n in range(self.Nt):
            # Update metaorder intensity
            self.asks[n] = self.book.best_ask
            self.bids[n] = self.book.best_bid
            self.prices[n] = self.compute_price(
                self.book.best_ask, self.book.best_bid)
            self.measure(n)

            volume = self.metaorder[n] * self.tstep
            self.book.timestep(self.tstep, volume)

    def measure(self, n):
        if n not in self.measurement_indices :
            return
        for quantity in self.measured_quantities:
            value = self.book.get_measure(quantity)
            self.measurements[quantity].append(np.copy(value))

    # ================== ANIMATION ==================

    def run_animation(self, fig, save=False):
        """Run the simulation and display a matplotlib animation

        :param fig: The figure the animation is displayed in, defaults to None
        :type fig: matplotlib figure, optional
        :param save: Set True to save the animation under ./animation.mp4, defaults to False
        :type save: bool, optional
        """
        self.set_animation(fig)
        self.animation = FuncAnimation(
            fig, self.update_animation, init_func=self.init_animation, repeat=False, frames=self.Nt, blit=True)
        if save:
            Writer = writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(
                artist='Me'), bitrate=1800)
            self.animation.save('../animation.mp4', writer=writer)

    def set_animation(self, fig):
        """Create subplot axes, lines and texts
        """
        lims = {}
        # if self.r < 0.4:
        #     lims['xlim'] = (-3 * self.lower_impact, 3*self.lower_impact)

        self.book.set_animation(fig, lims)
        self.price_ax = fig.add_subplot(2, 1, 2)
        self.price_ax.set_title('Price evolution')
        self.best_ask_line, = self.price_ax.plot(
            [], [], label='Best Ask', color='blue', ls='--')
        self.best_bid_line, = self.price_ax.plot(
            [], [], label='Best Bid', color='red', ls='--')
        self.price_line, = self.price_ax.plot(
            [], [], label=f'Price ({self.price_formula})', color='yellow')
        self.price_ax.plot([0, self.T], [0, 0],
                           ls='dashed', lw=0.5, color='black')
        self.price_ax.legend()
        # self.price_ax.set_ylim(
        #     (- self.impact_th, 1.5 * self.impact_th))
        self.price_ax.set_ylim(
            (self.xmin, self.xmax))
        self.price_ax.set_xlim((0, self.T))

        # fig.suptitle(self.parameters_string + self.constant_string)

    def init_animation(self):
        """Init function called by matplotlib's FuncAnimation
        """
        self.price_line.set_data([], [])
        self.best_bid_line.set_data([], [])
        self.best_ask_line.set_data([], [])

        return self.book.init_animation() + [self.price_line, self.best_ask_line, self.best_bid_line]

    def update_animation(self, n):
        """Update function called by matplotlib's FuncAnimation
        """
        if n % 10 == 0:
            print(f'Step {n}')
        volume = self.metaorder[n] * self.tstep
        self.asks[n] = self.book.best_ask
        self.bids[n] = self.book.best_bid
        self.prices[n] = self.compute_price(
            self.book.best_ask, self.book.best_bid)
        self.measure(n)

        self.price_line.set_data(
            self.time_interval[:n+1], self.prices[:n+1])
        self.best_ask_line.set_data(
            self.time_interval[:n+1], self.asks[:n+1])
        self.best_bid_line.set_data(
            self.time_interval[:n+1], self.bids[:n+1])
        return self.book.update_animation(self.tstep, volume) + [self.price_line, self.best_ask_line, self.best_bid_line]

    def __str__(self):

        # In the case of a mutli-actor book, the largest value
        # for L is used for all variables related to L.

        string = f""" Order book simulation.
        Time parameters :
                        T = {self.T},
                        Nt = {self.Nt},
                        tstep = {self.tstep:.1e},
                        dt = {self.dt:.1e},
                        n_steps = {self.n_steps:.1e}.

        Space parameters :
                        Price interval = [{self.xmin}, {self.xmax}],
                        Nx = {self.Nx},
                        dx = {self.dx:.1e}.

        Model constants:
                        D = {self.D:.1e},
                        lambda = {self.lambd},
                        nu = {self.nu},
                        L = {self.L},
                        J = {self.J}.

        Metaorder:
                        m0 = {self.m0:.1e},
                        dq = {self.m0 * self.tstep:.1e}.
                        n_start, n_end = ({self.n_start}, {self.n_end}).

        Theoretical values:
                        Participation rate = {self.participation_rate:.1e},
                        impact = {self.impact_th:.1e},
                        lower impact = {self.lower_impact},
                        alpha = {self.scheme_constant:.1e},
                        boundary factor = {self.boundary_factor:.1e},
                        first volume = {self.first_volume:.1f},
                        lower resolution = {self.lower_impact / self.dx :.1e}.
                        """
        return string


def standard_parameters(participation_rate, model_type, xmin=None, xmax=None, Nt=None, T=None):
    """Returns standard argument dictionary for a Simulation instance for
    a given participation rate and a model type
    .. warning:: Participation rates greater than r~2500 will most likely cause an error
    with the current settings.

    """
    if Nt == None:
        Nt = 100
    if T == None:
        T = Nt * 50
    r = abs(participation_rate)
    D = 0.5
    I = np.sqrt(2 * r * D * T)
    if xmin == None:
        xmin = -1.1 * I
    if xmax == None:
        xmax = 1.1 * I
    if r <= 1 :
        boundary_distance = np.sqrt(D * T)
        xmax = max(np.sqrt(r) * xmax, boundary_distance)
        xmin = min(np.sqrt(r) * xmin, -boundary_distance)

    Nx = int(xmax - xmin)
    X = max(abs(xmin), abs(xmax))

    dx = (xmax - xmin) / Nx

    L = 10/(dx * dx)
    if r == float('inf'):
        D = 0
        m0 = (L * X) / (5 * T)

    else:
        D = 0.5
        m0 = D * L * participation_rate

    simulation_args = {
        "model_type": model_type,
        "T": T,
        "Nt": Nt,
        "Nx": Nx,
        "xmin": xmin,
        "xmax": xmax,
        "D": D,
        "metaorder": [m0],
        "L": L,
        'nu': 0
    }
    return simulation_args
