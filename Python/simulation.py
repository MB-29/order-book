import numpy as np
from matplotlib.animation import FuncAnimation, writers
import warnings

from linear_discrete_book import LinearDiscreteBook
from continuous_book import ContinuousBook


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

        model_choice = {
            'discrete': LinearDiscreteBook,
            'continuous': ContinuousBook
        }

        # Time
        self.T = kwargs.get('T', 1)
        self.Nt = kwargs.get('Nt', 100)
        self.time_interval, self.tstep = np.linspace(
            0, self.T, num=self.Nt, retstep=True)
        self.obs_rate = kwargs.get('obs_rate', 1)
        self.dt = self.tstep / self.obs_rate

        # Space
        self.xmin = kwargs.get('xmin', -1)
        self.xmax = kwargs.get('xmax', 1)
        self.price_range = self.xmax - self.xmin
        self.Nx = kwargs.get('Nx', 100)
        self.dx = self.price_range / self.Nx
        self.boundary_distance = min(abs(self.xmin), self.xmax)

        # Order Book
        self.L = kwargs.get('L', 1e4)
        # In the case of a discrete simulation, diffusion constant is that of a Smoluchowski random walk
        self.D = kwargs.get(
            'D', 0.1) if model_type == 'continuous' else self.dx**2/(2*self.dt)
        book_args = {'xmin': self.xmin,
                     'xmax': self.xmax,
                     'Nx': self.Nx,
                     'L': self.L,
                     'D': self.D,
                     'dt': self.dt}
        if model_type == 'discrete':
            book_args['obs_rate'] = self.obs_rate
        self.book = model_choice.get(model_type)(**book_args)
        self.J = self.book.J
        self.dx = self.book.dx

        # Prices
        self.best_asks = np.zeros(self.Nt)
        self.best_bids = np.zeros(self.Nt)
        self.prices = np.zeros(self.Nt)

        self.price_formula = kwargs.get('price_formula', 'middle')
        price_formula_choice = {
            'middle': lambda a, b: (a+b)/2,
            'best_ask': lambda a, b: a,
            'best_bid': lambda a, b: b,
            'vwap': lambda a, b: self.compute_vwap(a, b)
        }
        self.compute_price = price_formula_choice[self.price_formula]
        
        # Metaorder
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

        # Theoretical values
        self.boundary_factor = np.sqrt(self.D*self.T)/(self.boundary_distance)
        self.infinity_density = self.book.L * self.book.xmax
        self.impact_th = np.sqrt(2*abs(self.m0)*self.T/self.L)
        self.density_shift_th = np.sqrt(
            abs(self.m0)*self.T*self.L)  # impact_th * L
        self.participation_rate = self.m0 / \
            (self.D * self.L) if self.D*self.L != 0 else float("inf")
        self.r = abs(self.participation_rate)
        self.alpha = self.D * self.tstep / (self.dx * self.dx)
        self.lower_impact = np.sqrt(
            abs(self.r)/(2*np.pi)) * self.impact_th

        # Strings
        self.parameters_string = fr'$m_0={self.m0:.2e}$, $J={self.J:.2f}$, d$t={self.dt:.2e}$ '
        self.constant_string = fr'$\Delta p={self.impact_th:.2f}$, boundary factor = {self.boundary_factor:.2f}, $r={self.r:.2e}$, $x \in [{self.xmin}, {self.xmax}]$'

        # Warnings and errors
        if model_type == 'discrete':
            if self.r < 1 and self.obs_rate < 100:
                warnings.warn(
                    f'Low number of diffusion steps {self.obs_rate} < 100,'
                    'try increasing spatial resolution.')
            if self.obs_rate < 1 and self.r < float('inf'):
                raise ValueError(
                    'Order diffusion is not possible because diffusion distance is smaller that space subinterval.'
                    'Try decreasing participation rate')
            if self.obs_rate > 200:
                raise ValueError(
                    f'Many diffusion steps : ~ {int(self.obs_rate)}.')

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
            self.book.dq = self.metaorder[n] * self.tstep
            self.best_asks[n] = self.book.best_ask
            self.best_bids[n] = self.book.best_bid
            self.prices[n] = self.compute_price(
                self.book.best_ask, self.book.best_bid)
            self.book.timestep()

    # ================== COMPUTATIONS ==================

    def get_growth_th(self):
        """Return theoretical price impact, starting from price 0
        """
        A = self.m0/(self.book.L*np.sqrt(self.D * np.pi)
                     ) if self.r < 1 else np.sqrt(2)*np.sqrt(self.m0/self.L)
        growth = A * np.sqrt(self.time_interval_shifted[self.n_start: self.n_end])
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
        self.price_ax.set_ylim(
            (- self.impact_th, 1.5 * self.impact_th))
        self.price_ax.set_xlim((0, self.T))

        fig.suptitle(self.parameters_string + self.constant_string)

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
        self.book.dq = self.metaorder[n] * self.tstep
        self.best_asks[n] = self.book.best_ask
        self.best_bids[n] = self.book.best_bid
        self.prices[n] = self.compute_price(
            self.book.best_ask, self.book.best_bid)
        self.price_line.set_data(
            self.time_interval[:n+1], self.prices[:n+1])
        self.best_ask_line.set_data(
            self.time_interval[:n+1], self.best_asks[:n+1])
        self.best_bid_line.set_data(
            self.time_interval[:n+1], self.best_bids[:n+1])
        return self.book.update_animation(n) + [self.price_line, self.best_ask_line, self.best_bid_line]

    def __str__(self):
        string = f""" Order book simulation.
        Time parameters :
                        T = {self.T},
                        Nt = {self.Nt},
                        tstep = {self.tstep:.1e},
                        obs_rate = {self.obs_rate:.1e}.,
                        dt = {self.dt:.1e}.

        Space parameters :
                        Price interval = [{self.xmin}, {self.xmax}],
                        Nx = {self.Nx},
                        dx = {self.dx:.1e}.

        Model constants:
                        D = {self.book.D:.1e},
                        J = {self.J:.1e},
                        L = {self.book.L:.1e}.

        Metaorder:
                        m0 = {self.m0:.1e},
                        dq = {self.m0 * self.tstep:.1e}.
                        n_start, n_end = ({self.n_start}, {self.n_end}).

        Theoretical values:
                        Participation rate = {self.participation_rate:.1e},
                        impact = {self.impact_th:.1e},
                        lower impact = {self.lower_impact},
                        alpha = {self.alpha:.1e},
                        boundary factor = {self.boundary_factor:.1e},
                        first volume = {self.book.L * self.dx * self.dx:.1e},
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
    tstep = T / Nt
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
    elif r >= 1:
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

    # Ensures diffusion constant is consistent with that of Smoluchowski random walk
    dt = dx * dx / (2 * D) if D != 0 else float('inf')
    obs_rate = tstep / dt
    simulation_args = {
        "model_type": model_type,
        "T": T,
        "Nt": Nt,
        "price_formula": price_formula,
        "Nx": Nx,
        "xmin": xmin,
        "xmax": xmax,
        "L": L,
        "D": D,
        "metaorder": [m0]
    }
    if model_type == 'discrete':
        simulation_args['obs_rate'] = obs_rate
    return simulation_args
