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

    def __init__(self, book_parameters, T, Nt, m0, model_type='discrete', n_start=None, n_end=None):
        """

        Arguments:
            book_parameters {dictionary} -- Argument for the instance of order book.
            T {float} -- Time horizon
            Nt {int} -- Number of time steps
            m0 {float} -- Metaorder trading intensity

        Keyword Arguments:
            model_type {str} -- 'discrete' or 'continuous' (default: 'discrete')
            n_start {int} --  Metaorder start step
            n_end {int} -- Metaorder end step
        """

        # Model type
        assert model_type in ['discrete', 'continuous']

        model_dic = {
            'discrete': LinearDiscreteBook,
            'continuous': ContinuousBook
        }

        # Order book
        self.book_parameters = book_parameters
        self.book = model_dic.get(model_type)(**book_parameters)
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
        self.m0 = m0
        self.beta = np.sqrt(2*m0*T/self.book.L)/self.book.price_range
        self.n_start = Nt//10+1 if not n_start else n_start
        self.n_end = 9*Nt//10 if not n_end else n_end
        self.t_start = self.n_start * self.tstep
        self.t_end = self.n_end * self.tstep
        self.time_interval_shifted = self.time_interval-self.t_start

        # Theoretical values
        self.infinity_density = self.book.L * self.book.upper_bound
        self.price_shift_th = np.sqrt(m0*T/self.book.L)
        self.density_shift_th = np.sqrt(m0*T*self.book.L)  # price_shift_th * L
        self.A_low = m0/(self.book.L*np.sqrt(self.book.D * np.pi))
        self.A_high = np.sqrt(2)*np.sqrt(m0/self.book.L)
        self.participation_rate = self.m0/self.J
        self.compute_theoretical_growth()

        # Plot
        self.parameters_string = r'$m_0={{{0}}}$, $J={{{1}}}$, d$t={{{2}}}$, $X = {{{3}}}$ '.format(
            round(self.m0, 2), round(self.J, 4),  round(self.dt, 5), self.book.upper_bound)
        self.constant_string = r'$\Delta p={{{0}}}$, $\beta={{{1}}}$, $r={{{2}}}$, d$x={{{3}}}$'.format(
            round(self.price_shift_th, 2), round(self.beta, 2), round(self.participation_rate, 2), round(self.dx, 3))

    def run(self, fig=plt.gcf(), animation=False, save=False):
        """Run the Nt steps of the simulation

        Keyword Arguments:
            fig {pyplot figure} -- The figure the animation is displayed in
            animation {bool} -- Set True to display an animation
            save {bool} -- Set True to save the animation under ./animation.m4
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
            (-1 - self.price_shift_th, 1 + 1.5 * self.price_shift_th))
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
            print(f'Iteration {n}')
        y_min, y_max = -self.density_shift_th, self.density_shift_th

        self.book.mt = self.m0 if (
            n >= self.n_start and n <= self.n_end) else 0

        self.prices[n] = self.book.price
        self.price_line.set_data(

            self.time_interval[:n+1], self.prices[:n+1])
        return self.book.update_animation(n) + [self.price_line]
