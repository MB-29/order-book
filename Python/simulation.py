import numpy as np
from matplotlib.animation import FuncAnimation, writers
import warnings

from linear_discrete_book import LinearDiscreteBook
from discrete_book import DiscreteBook
from continuous_book import ContinuousBook
from multi_discrete_book import MultiDiscreteBook


class Simulation:
    """Implements a simulation of Latent Order Book time evolution.
    """

    def __init__(self, model_type, metaorder=[0], **kwargs):
        """

        Arguments:
            model_type {str} -- 'discrete' or 'continuous' 
            metaorder {array} -- Meta-order intensity values.
                                An array of size 1 will be converted into a
                                constant meta-order with the corresponding value.
            kwargs -- T, Nt, xmin, xmax, Nx, L, D, price_formula
        """

        # Model type
        assert model_type in ['discrete', 'continuous']
        self.model_type = model_type

        # Time
        self.T = kwargs.get('T', 1)
        self.Nt = kwargs.get('Nt', 100)
        self.time_interval, self.dt = np.linspace(
            0, self.T, num=self.Nt, retstep=True)

        # Space
        self.xmin = kwargs.get('xmin', -1)
        self.xmax = kwargs.get('xmax', 1)
        self.price_range = self.xmax - self.xmin
        self.Nx = kwargs.get('Nx', 100)
        self.dx = self.price_range / self.Nx
        self.boundary_distance = min(abs(self.xmin), self.xmax)

        # Prices
        self.asks = np.zeros(self.Nt)
        self.bids = np.zeros(self.Nt)
        self.prices = np.zeros(self.Nt)

        # Orders
        self.L = kwargs['L']
        self.D = kwargs['D']
        self.nu = kwargs.get('nu', 0)
        self.lambd = self.L * np.sqrt(self.nu * self.D)
        self.J = self.D * self.L
        self.book = self.set_order_book(model_type)

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

        self.t_start = self.n_start * self.dt
        self.t_end = self.n_end * self.dt
        self.time_interval_shifted = self.time_interval-self.t_start

        # self.theoretical_values()

    def set_order_book(self, model_type):
        
        # Order Book
        book_args = {'xmin': self.xmin,
                     'xmax': self.xmax,
                     'Nx': self.Nx,
                     'dt': self.dt,
                     'L': self.L,
                     'D': self.D,
                     'lambd': self.lambd,
                     'nu': self.nu
                     }
        # Allow a certain number of timesteps for the Smoluchowski random walk
        # to reach diffusion : impose n_diff such that
        # D = dx**2 / (dt / n_diff)
        self.n_diff = int(self.dt * self. D / (self.dx)**2)
        if model_type == 'discrete':
            book_args['n_diff'] = self.n_diff
            print(f'n_diff = {self.n_diff}')

        if not np.isscalar(self.L):
            return MultiDiscreteBook(**book_args)

        linear = (self.nu == 0)
        model_name = 'linear_' + model_type if linear else model_type

        model_choice = {
            'discrete': DiscreteBook,
            'linear_discrete': LinearDiscreteBook,
            'linear_continuous': ContinuousBook,
        }



        return model_choice.get(model_name)(**book_args)

    def theoretical_values(self):
        # Theoretical values
        self.boundary_factor = np.sqrt(self.D*self.T)/(self.boundary_distance)
        self.infinity_density = self.L * self.xmax
        self.impact_th = np.sqrt(2*abs(self.m0)*self.T/self.L)
        self.density_shift_th = np.sqrt(
            abs(self.m0)*self.T*self.L)  # impact_th * L
        self.participation_rate = self.m0 / \
            (self.D * self.L) if self.D != 0 else float("inf")
        self.r = abs(self.participation_rate)
        self.alpha = self.D * self.dt / (self.dx * self.dx)
        self.lower_impact = np.sqrt(
            abs(self.r)/(2*np.pi)) * self.impact_th

        # Strings
        self.parameters_string = fr'$m_0={self.m0:.2e}$, d$t={self.dt:.2e}$ '
        self.constant_string = fr'$\Delta p={self.impact_th:.2f}$, boundary factor = {self.boundary_factor:.2f}, $r={self.r:.2e}$, $x \in [{self.xmin}, {self.xmax}]$'

        # Warnings and errors
        if self.model_type == 'discrete':
            if self.r < 1 and self.n_diff < 100:
                warnings.warn(
                    f'Low number of diffusion steps {self.n_diff} < 100,'
                    'try increasing spatial resolution.')
            if self.n_diff < 1 and self.r < float('inf'):
                raise ValueError(
                    'Order diffusion is not possible because diffusion distance is smaller that space subinterval.'
                    'Try decreasing participation rate')
            if self.n_diff > 200:
                raise ValueError(
                    f'Many diffusion steps : ~ {int(self.n_diff)}.')

    def compute_vwap(self, best_ask, best_bid):
        return (abs(self.book.best_ask_volume) * best_ask
                + abs(self.book.best_bid_volume) * best_bid)/(
                    abs(self.book.best_ask_volume) + abs(self.book.best_bid_volume))

    def run(self, fig=None, animation=False, save=False):
        """Run the Nt steps of the simulation

        Keyword Arguments:
            fig {pyplot figure} -- The figure the animation is displayed in
            animation {bool} -- Set True to display an animation
            save {bool} -- Set True to save the animation under ./animation.mp4
        """

        if animation:
            self.run_animation(fig, save)
            return

        for n in range(self.Nt):
            # Update metaorder intensity
            self.book.dq = self.metaorder[n] * self.dt
            self.asks[n] = self.book.best_ask
            self.bids[n] = self.book.best_bid
            self.prices[n] = self.compute_price(
                self.book.best_ask, self.book.best_bid)
            self.book.timestep()

    # ================== COMPUTATIONS ==================

    def get_growth_th(self):
        """Return theoretical price impact, starting from price 0
        """
        A = self.m0/(self.book.L*np.sqrt(self.D * np.pi)
                     ) if self.r < 1 else np.sqrt(2)*np.sqrt(self.m0/self.L)
        growth = A * \
            np.sqrt(self.time_interval_shifted[self.n_start: self.n_end])
        return growth
        # self.growth_th_low = self.prices[self.n_start] + self.A_low * \
        #     np.sqrt(
        #     self.time_interval_shifted[self.n_start:self.n_end])
        # self.growth_th_high = self.prices[self.n_start] + self.A_high * np.sqrt(
        #     self.time_interval_shifted[self.n_start:self.n_end])
        # self.growth_th = self.growth_th_low if self.m0 < self.J else self.growth_th_high

    # ================== ANIMATION ==================

    def run_animation(self, fig, save=False):
        """
        Arguments:
            fig {pyplot figure} -- The figure the animation is displayed in
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
        self.price_ax = fig.add_subplot(1, 2, 2)
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
        """Init function called by FuncAnimation
        """
        self.price_line.set_data([], [])
        self.best_bid_line.set_data([], [])
        self.best_ask_line.set_data([], [])

        return self.book.init_animation() + [self.price_line, self.best_ask_line, self.best_bid_line]

    def update_animation(self, n):
        """Update function called by FuncAnimation
        """
        if n % 10 == 0:
            print(f'Step {n}')
        self.book.dq = self.metaorder[n] * self.dt
        self.asks[n] = self.book.best_ask
        self.bids[n] = self.book.best_bid
        self.prices[n] = self.compute_price(
            self.book.best_ask, self.book.best_bid)
        self.price_line.set_data(
            self.time_interval[:n+1], self.prices[:n+1])
        self.best_ask_line.set_data(
            self.time_interval[:n+1], self.asks[:n+1])
        self.best_bid_line.set_data(
            self.time_interval[:n+1], self.bids[:n+1])
        return self.book.update_animation(n) + [self.price_line, self.best_ask_line, self.best_bid_line]

    def __str__(self):
        string = f""" Order book simulation.
        Time parameters :
                        T = {self.T},
                        Nt = {self.Nt},
                        dt = {self.dt:.1e},
                        n_diff = {self.n_diff:.1e}.,
                        dt = {self.dt:.1e}.

        Space parameters :
                        Price interval = [{self.xmin}, {self.xmax}],
                        Nx = {self.Nx},
                        dx = {self.dx:.1e}.

        Model constants:
                        D = {self.D:.1e},
                        J = {self.J:.1e},
                        L = {self.L:.1e}.

        Metaorder:
                        m0 = {self.m0:.1e},
                        dq = {self.m0 * self.dt:.1e}.
                        n_start, n_end = ({self.n_start}, {self.n_end}).

        Theoretical values:
                        Participation rate = {self.participation_rate:.1e},
                        impact = {self.impact_th:.1e},
                        lower impact = {self.lower_impact},
                        alpha = {self.alpha:.1e},
                        boundary factor = {self.boundary_factor:.1e},
                        first volume = {self.L * self.dx * self.dx:.1e},
                        lower resolution = {self.lower_impact / self.dx :.1e}.
                        """
        return string


def standard_parameters(participation_rate, model_type, T=1, xmin=-0.25, xmax=1, Nx=None, Nt=100):
    """Returns standard argument dictionary for a Simulation instance for
    a given participation rate and a model type

    Arguments:
        participation_rate {float} 
        model_type {string} -- 'discrete' or 'continuous'

    """
    dt = T / Nt
    r = abs(participation_rate)
    if Nx == None:
        Nx = 5001 if model_type == 'continuous' else 501
    boundary_dist = min(abs(xmin), abs(xmax))
    X = max(abs(xmin), abs(xmax))
    dx = (xmax - xmin) / Nx
    L = 1/(dx * dx)
    side_formula = 'best_ask' if participation_rate >= 0 else 'best_bid'
    if r == float('inf'):
        price_formula = side_formula
        D = 0
        m0 = (L * X) / (5 * T)
    elif r >= 5:
        m0 = (L * X) / (5 * T)
        D = m0 / (L*r)
        price_formula = side_formula
    elif r > 0.5:
        price_formula = 'vwap'
        D = boundary_dist**2/(2 * T)
        m0 = L * D * participation_rate
    else:
        price_formula = 'vwap'
        D = boundary_dist**2 / (2 * T)
        m0 = L * D * participation_rate

    simulation_args = {
        "model_type": model_type,
        "T": T,
        "Nt": Nt,
        "price_formula": price_formula,
        "Nx": Nx,
        "xmin": xmin,
        "xmax": xmax,
        "D": D,
        "metaorder": [m0],
        "L": L,
        # "L": np.array([L]),
        'nu': 0
        # 'nu': np.array([0.1])
    }
    return simulation_args
