import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import os

from discrete_book import DiscreteBook
from linear_discrete_book import LinearDiscreteBook
from continuous_book import ContinuousBook


class Simulation:
    """Implements a simulation of Latent Order Book time evolution.
    """

    def __init__(self, book_args, T, Nt, metaorder_args, model_type='discrete'):
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

        # Order book
        self.book_args = book_args
        self.book = model_choice.get(model_type)(**book_args)
        self.J = self.book.J
        self.dx = self.book.dx

        # Time evolution
        self.T = T
        self.Nt = Nt
        self.dt = T/float(Nt)
        self.prices = np.zeros(Nt)
        self.time_interval, self.tstep = np.linspace(
            0, T, num=Nt, retstep=True)

        # Metaorder
        metaorder = metaorder_args.get('metaorder', 0)
        if len(metaorder) == 1:
            self.metaorder = np.full(Nt, metaorder)
        else:
            assert len(metaorder) == Nt
            self.metaorder = metaorder

        self.m0 = metaorder_args.get('m0', 0)
        self.beta = np.sqrt(2*self.m0*T/self.book.L)/self.book.price_range
        self.n_start = metaorder_args.get('n_start', Nt//10+1)
        self.n_end = metaorder_args.get('n_end', 9*Nt//10)
        self.t_start = self.n_start * self.tstep
        self.t_end = self.n_end * self.tstep
        self.time_interval_shifted = self.time_interval-self.t_start

        # Theoretical values
        self.infinity_density = self.book.L * self.book.upper_bound
        self.price_shift_th = np.sqrt(self.m0*T/self.book.L)
        self.density_shift_th = np.sqrt(
            self.m0*T*self.book.L)  # price_shift_th * L
        self.A_low = self.m0/(self.book.L*np.sqrt(self.book.D * np.pi))
        self.A_high = np.sqrt(2)*np.sqrt(self.m0/self.book.L)
        self.participation_rate = self.m0/self.J if self.J !=0 else float("inf")
        self.compute_theoretical_growth()
        self.alpha = self.book.D * self.dt / (self.dx * self.dx)
        self.lower_impact = np.sqrt(self.participation_rate/(2*np.pi)) * self.price_shift_th

        # Plot
        self.parameters_string = r'$m_0={{{0}}}$, $J={{{1}}}$, d$t={{{2}}}$, $X = {{{3}}}$ '.format(
            round(self.m0, 2), round(self.J, 4),  round(self.dt, 5), self.book.upper_bound)
        self.constant_string = r'$\Delta p={{{0}}}$, $\beta={{{1}}}$, $r={{{2}}}$, d$x={{{3}}}$'.format(
            round(self.price_shift_th, 2), round(self.beta, 2), round(self.participation_rate, 2), round(self.dx, 3))

        print(self)

    def run(self, fig=plt.gcf(), animation=False, save=False):
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
            mt = self.m0 if (n >= self.n_start and n <= self.n_end) else 0

            self.prices[n] = self.book.price
            self.book.mt = mt
            self.book.timestep()

    # ================== COMPUTATIONS ==================

    def compute_theoretical_growth(self):
        self.growth_th_low = self.prices[self.n_start-1] + self.A_low * \
            np.sqrt(
            self.time_interval_shifted[self.n_start:self.n_end])
        self.growth_th_high = self.prices[self.n_start-1] + self.A_high * \
            np.sqrt(
            self.time_interval_shifted[self.n_start:self.n_end])
        self.growth_th = self.growth_th_low if self.m0 < self.J else self.growth_th_high

    def compute_growth_mean_error(self, ord=2):
        self.growth_mean_error = np.linalg.norm(
            self.growth_th - self.prices[self.n_start:self.n_end], ord=ord) / np.power(self.n_end - self.n_start, 1/ord)
        return self.growth_mean_error

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
                self.prices, label='price evolution')
        if low:
            ax.plot(self.time_interval_shifted[self.n_start:self.n_end],
                    self.growth_th_low, label='low regime', lw=1, color='green')
        if high:
            ax.plot(self.time_interval_shifted[self.n_start:self.n_end],
                    self.growth_th_high, label='high regime', lw=1, color='orange')

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
        self.growth_error_abs = abs(
            self.prices[self.n_start:self.n_end] - self.growth_th)

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
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            self.animation.save('animation.mp4', writer=writer)
        plt.show()

    def set_animation(self, fig):
        """Create subplot axes, lines and texts
        """

        self.book.set_animation(fig)
        self.price_ax = fig.add_subplot(1, 2, 2)
        self.price_ax.set_title('Price evolution')
        self.price_line, = self.price_ax.plot([], [], label='Price')
        self.price_ax.plot([0, self.T], [0, 0],
                           ls='dashed', lw=0.5, color='black')
        self.price_ax.legend()
        self.price_ax.set_ylim(
            (- self.price_shift_th, 1.5 * self.price_shift_th))
        self.price_ax.set_xlim((0, self.T))

        fig.suptitle(self.parameters_string + self.constant_string)

    def init_animation(self):
        """Init function called by FuncAnimation
        """
        self.price_line.set_data([], [])

        return self.book.init_animation() + [self.price_line]

    def update_animation(self, n):
        """Update function called by FuncAnimation
        """
        if n % 10 == 0:
            print(f'Step {n}')
        y_min, y_max = -self.density_shift_th, self.density_shift_th

        self.book.mt = self.m0 if (
            n >= self.n_start and n <= self.n_end) else 0

        self.prices[n] = self.book.price
        self.price_line.set_data(

            self.time_interval[:n+1], self.prices[:n+1])
        return self.book.update_animation(n) + [self.price_line]

    def __str__(self):
        string = f""" Order book simulation.
        Time parameters :
                        T = {self.T},
                        Nt = {self.Nt},
                        dt = {self.dt:.1e}.

        Space parameters : 
                        Price range = {self.book.upper_bound - self.book.lower_bound},
                        dx = {self.dx:.1e}.
                        
        Model constants : 
                        D = {self.book.D:.1e},
                        J = {self.J:.1e},
                        L = {self.book.L:.1e}.

        Metaorder : 
                        m0 = {self.m0:.1e},
                        dq = {self.m0 * self.dt:.1e}.

        Theoretical values : 
                        Participation rate = {self.participation_rate:.1e},
                        impact = {self.price_shift_th:.1e},
                        lower impact = {self.lower_impact},
                        alpha = {self.alpha:.1e},
                        beta = {self.beta:.1e},
                        lower resolution = {self.lower_impact / (self.dx * self.participation_rate):.1e}.
                        """
        return string
