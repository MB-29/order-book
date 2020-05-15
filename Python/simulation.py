import numpy as np
from matplotlib.animation import FuncAnimation, writers

from linear_discrete_book import LinearDiscreteBook
from continuous_book import ContinuousBook

# TODO record best ask and best bid instead of price ?


class Simulation:
    """Implements a simulation of Latent Order Book time evolution.
    """

    def __init__(self, model_type, metaorder=[0], **kwargs):
        """

        Arguments:
            book_args {dictionary} -- Arguments for the instance of order book.
            T {float} -- Time horizon
            Nt {int} -- Number of time steps
            metaorder_args {dictionary} -- Sets metaorder variables.
            Metaorder trading intensity is defined by key 'metaorder' whose value
            must be an array either of size 1, in which case it stands for the value of a constant
            metaorder, or of size Nt.
            Key 'm0' denotes the temporal mean of the metaorder.
            Keys 'n_start' and 'n_end' set the start and end steps.

        Keyword Arguments:
            model_type {str} -- 'discrete' or 'continuous' (default: 'discrete')
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
        self.n_relax = kwargs.get('n_relax', 1)
        self.dt = self.tstep/self.n_relax

        # Space
        self.xmin = kwargs.get('xmin', -1)
        self.xmax = kwargs.get('xmax', 1)
        self.price_range = self.xmax - self.xmin
        self.Nx = kwargs.get('Nx', 100)
        self.dx = self.price_range / self.Nx
        self.boundary_distance = min(abs(self.xmin), self.xmax)

        # Order Book
        self.L = kwargs.get('L', 1e4)
        self.D = kwargs.get('D', 0.1)
        book_args = {'xmin': self.xmin,
                     'xmax': self.xmax,
                     'Nx': self.Nx,
                     'L': self.L,
                     'D': self.D}
        book_args['dt'] = self.dt

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
        if len(metaorder) == 1:
            self.metaorder = np.full(self.Nt, metaorder)
            self.m0 = metaorder[0]
        else:
            assert len(metaorder) == self.Nt
            self.metaorder = metaorder
            self.m0 = metaorder.mean()
        print(metaorder)
        self.boundary_rate = np.sqrt(self.D*self.T)/self.boundary_distance
        self.n_start = kwargs.get('n_start', 0)
        self.n_end = kwargs.get('n_end', self.Nt)
        self.t_start = self.n_start * self.tstep
        self.t_end = self.n_end * self.tstep
        self.time_interval_shifted = self.time_interval-self.t_start

        # Theoretical values
        self.infinity_density = self.book.L * self.book.xmax
        self.impact_th = np.sqrt(2*abs(self.m0)*self.T/self.L)
        self.density_shift_th = np.sqrt(
            abs(self.m0)*self.T*self.L)  # impact_th * L
        self.A_low = self.m0/(self.book.L*np.sqrt(self.D * np.pi))
        self.A_high = np.sqrt(2)*np.sqrt(self.m0/self.L)
        self.participation_rate = self.m0 / \
            self.J if self.J != 0 else float("inf")
        # self.compute_theoretical_growth()
        self.alpha = self.D * self.dt / (self.dx * self.dx)
        self.lower_impact = np.sqrt(
            self.participation_rate/(2*np.pi)) * self.impact_th

        # Plot
        self.parameters_string = r'$m_0={{{0}}}$, $J={{{1}}}$, d$t={{{2}}}$, $X = {{{3}}}$ '.format(
            round(self.m0, 2), round(self.J, 4),  round(self.dt, 5), self.book.xmax)
        self.constant_string = r'$\Delta p={{{0}}}$, $\beta={{{1}}}$, $r={{{2}}}$, d$x={{{3}}}$'.format(
            round(self.impact_th, 2), round(self.boundary_rate, 2), round(self.participation_rate, 2), round(self.dx, 3))

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

    def compute_theoretical_growth(self):
        self.growth_th_low = self.prices[self.n_start] + self.A_low * \
            np.sqrt(
            self.time_interval_shifted[self.n_start:self.n_end])
        self.growth_th_high = self.prices[self.n_start] + self.A_high * np.sqrt(
            self.time_interval_shifted[self.n_start:self.n_end])
        self.growth_th = self.growth_th_low if self.m0 < self.J else self.growth_th_high

    # ================== PLOTS ==================

    def plot_price(self, ax, symlog=False, low=False, high=False):
        """

        Arguments:
            ax {matplotlib ax} --

        Keyword Arguments:
            symlog {bool} -- Use symlog scale (default: {False})
            low {bool} -- Plot low participation regime theoretical impact (default: {False})
            high {bool} -- Plot high participation regime theoretical impact (default: {False})
        """

        # Lines
        ax.plot(self.time_interval_shifted,
                self.prices, label=f'price ({self.price_formula})', color='yellow')
        ax.plot(self.time_interval_shifted,
                self.best_asks, label='best ask', color='blue', ls='--')
        ax.plot(self.time_interval_shifted,
                self.best_bids, label='best bid', color='red', ls='--')
        if low:
            ax.plot(self.time_interval_shifted[self.n_start:self.n_end],
                    self.growth_th_low, label='low regime', lw=1, color='green')
        if high:
            ax.plot(self.time_interval_shifted[self.n_start:self.n_end],
                    self.growth_th_high, label='high regime', lw=1, color='magenta')

        # Scale
        if symlog:
            ax.set_yscale('symlog', linthreshy=1e-1)
            ax.set_xscale('symlog', linthreshx=self.tstep)

        ax.legend(loc='lower right')
        ax.set_title(self.parameters_string)

        return ax

    def plot_err(self, ax, relative=False, symlog=False):
        """

        Arguments:
            ax {matplotlib ax} --

        Keyword Arguments:
            relative {bool} -- Plot relative error
            symlog {bool} -- Use symlog scale (default: {False})
        """
        self.growth_error_abs = self.prices[self.n_start:self.n_end] - \
            self.growth_th

        self.growth_error_rel = self.growth_error_abs/self.growth_th
        self.growth_error = self.growth_error_rel if relative else self.growth_error_abs
        # Curves
        label = 'relative error' if relative else 'absolute error'
        ax.plot(self.time_interval_shifted[self.n_start: self.n_end],
                self.growth_error, label=label)

        # Scale
        if symlog:
            ax.set_yscale('symlog', linthreshy=1e-1)
            ax.set_xscale('symlog', linthreshx=self.tstep)

        ax.legend(loc='lower right')
        ax.set_title('Error')

        return ax

    # ================== ANIMATION ==================

    def run_animation(self, fig, save=False):
        """
        Arguments:
            fig {pyplot figure} -- The figure the animation is displayed in
        """
        save = False
        self.set_animation(fig)
        self.animation = FuncAnimation(
            fig, self.update_animation, init_func=self.init_animation, repeat=False, frames=self.Nt, blit=True)
        if save:
            Writer = writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(
                artist='Me'), bitrate=1800)
            self.animation.save('animation.mp4', writer=writer)

    def set_animation(self, fig):
        """Create subplot axes, lines and texts
        """
        lims = {}
        if self.participation_rate < 0.4:
            lims['xlim'] = (-3 * self.lower_impact, 3*self.lower_impact)

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
                        t_step = {self.tstep:.1e},
                        dt = {self.dt:.1e}.

        Space parameters :
                        Price interval = [{self.xmin}, {self.xmax}],
                        dx = {self.dx:.1e}.

        Model constants:
                        D = {self.book.D:.1e},
                        J = {self.J:.1e},
                        L = {self.book.L:.1e}.

        Metaorder:
                        m0 = {self.m0:.1e},
                        dq = {self.m0 * self.dt:.1e}.

        Theoretical values:
                        Participation rate = {self.participation_rate:.1e},
                        impact = {self.impact_th:.1e},
                        lower impact = {self.lower_impact},
                        alpha = {self.alpha:.1e},
                        beta = {self.boundary_rate:.1e},
                        first volume = {self.book.L * self.dx * self.dx:.1e},
                        lower resolution = {self.lower_impact / self.dx :.1e}.
                        """
        return string

def standard_parameters(participation_rate, model_type, Nx=5001):
    r = abs(participation_rate)
    xmax = 1 if participation_rate > 0 else 0.1
    xmin = -0.1 if participation_rate > 0 else 1
    dx = (xmax - xmin) / Nx
    L = 1/(dx * dx)
    if r >= 1:
        m0 = L / 5
        D = m0 / (L*r)
        price_formula = 'best_ask'
    elif r > 0.5:
        price_formula = 'vwap'
        D = 0.1
        m0 = L * D * r
    else:
        price_formula = 'middle'
        D = min(abs(xmin)**2, abs(xmax)**2)
        m0 = L * D * r

    if model_type == 'discrete':
        dx = (xmax - xmin)/float(Nx)
        dt = dx * dx / (2 * D)

    simulation_args = {
        "model_type": model_type,
        "T": 1,
        "Nt": 100,
        "price_formula": price_formula,
        "Nx": Nx,
        "xmin": xmin,
        "xmax": xmax,
        "L": L,
        "D": D,
        "metaorder" : [m0]
    }

    return simulation_args
