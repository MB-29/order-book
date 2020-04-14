import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from order_book import OrderBook


class Simulation:

    def __init__(self, book_parameters, T, Nt, m0, n_start=None, n_end=None, plot=False):

        # Order book
        self.book_parameters = book_parameters
        self.book = OrderBook(**book_parameters)
        self.J = self.book.J

        # Time evolution
        self.T = T
        self.Nt = Nt
        self.prices = np.zeros(Nt)
        self.time_interval, self.tstep = np.linspace(
            0, T, num=Nt, retstep=True)
        self.n_start = Nt//10 if not n_start else n_start
        self.n_end = 9*Nt//10 if not n_end else n_end
        self.t_start = self.n_start * self.tstep
        self.t_end = self.n_end * self.tstep
        self.time_interval_shifted = self.time_interval-self.t_start

        # Metaorder
        self.m0 = m0
        self.beta = np.sqrt(2*m0*T/self.book.D*self.J)/self.book.price_range

        # Theoretical values
        self.infinity_density = self.book.L * self.book.upper_bound
        self.price_shift_th = np.sqrt(m0*T/self.book.L)
        self.density_shift_th = np.sqrt(m0*T*self.book.L)  # price_shift_th * L
        self.A_low = m0/(self.J*np.sqrt(np.pi))
        self.A_high = np.sqrt(2*m0/self.J)
        self.participation_rate = self.m0/self.J
        self.compute_theoretical_growth()

        # Plot
        self.plot = plot

    def run(self, animation=False):

        for n in range(self.Nt):

            # Update metaorder intensity
            mt = self.m0 if (n >= self.n_start and n <= self.n_end) else 0

            self.book.mt = mt
            self.book.timestep()
            self.prices[n] = self.book.price

            if animation:
                plt.title(
                    f'iteration {n}, mt = {mt}, price = {self.book.price}')
                ymin, ymax = -self.infinity_density, self.infinity_density
                plt.ylim((ymin, ymax))
                plt.xlim((-0.1, 0.1))
                plt.plot(self.book.X, self.book.density,
                         label='order density', linewidth=1)
                plt.vlines(self.book.price, -self.infinity_density, self.infinity_density, label='price',
                           color='red', linewidth=1)
                plt.hlines(0, self.book.lower_bound, self.book.upper_bound, color='black',
                           linewidth=0.5, linestyle='dashed')

                plt.legend()
                plt.pause(0.001)
                plt.cla()

        print('Simulation completed')

    def compute_theoretical_growth(self):
        self.growth_th_low = self.prices[self.n_start-1] + self.A_low * \
            np.sqrt(self.book.D *
                    self.time_interval_shifted[self.n_start:self.n_end])
        self.growth_th_high = self.prices[self.n_start-1] + self.A_high * \
            np.sqrt(self.book.D *
                    self.time_interval_shifted[self.n_start:self.n_end])
        self.growth_th = self.growth_th_high if self.m0 > self.J else self.growth_th_low

    def compute_growth_mean_error(self, ord=2):
        self.growth_mean_error = np.linalg.norm(
            self.growth_th - self.prices[self.n_start:self.n_end], ord=ord) / np.power(self.n_end - self.n_start, 1/ord)
        return self.growth_mean_error

    def plot_price(self, ax, symlog=False, low=False, high=False):

        # Curves
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
        ax.set_title('Price evolution')

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

    def set_animation(self, fig):
        self.density_ax = fig.add_subplot(1, 2, 1)
        self.price_ax = fig.add_subplot(1, 2, 2)
        self.density_line, = self.density_ax.plot([], [], label='Density')
        self.price_line, = self.price_ax.plot([], [], label='Price')

        self.density_text = self.price_ax.text(0.02, 0.95, '', transform=self.density_ax.transAxes)
        self.price_text = self.price_ax.text(0.02, 0.95, '', transform=self.price_ax.transAxes)
    
    def init_animation(self):
        y_min, y_max = -self.density_shift_th, self.density_shift_th
        x_min, x_max = -self.price_shift_th, self.price_shift_th

        self.density_line.set_data([], [])
        self.density_ax.set_xlim((x_min, x_max))
        self.density_ax.set_ylim((y_min, y_max))

        self.price_ax.set_xlim((0, self.T))
        self.price_ax.set_ylim((x_min, x_max))

        self.density_ax.hlines(0, self.book.lower_bound, self.book.upper_bound, color='black',
                           linewidth=0.5, linestyle='dashed')
        self.density_ax.hlines(0, y_min,y_max, color='black',
                           linewidth=0.5, linestyle='dashed')

        self.price_ax.hlines(0, 0, self.T, color='black',
                           linewidth=0.5, linestyle='dashed')

        self.density_text.set_text('Density')
        self.price_text.set_text('Market price')
        return self.density_line, self.price_line

    def update_animation(self, n):
        mt = self.m0 if (n >= self.n_start and n <= self.n_end) else 0

        self.book.mt = mt
        self.book.timestep()
        self.prices[n] = self.book.price

        self.density_line.set_data(self.book.X, self.book.density)
        self.density_line.set_data(self.book.X, self.book.density)
        self.price_line.set_data(self.time_interval_shifted[:n+1], self.prices[:n+1])
        return self.density_line, self.price_line

    def run_animation(self, fig):
        self.set_animation(fig)
        self.animation = FuncAnimation(
            fig, self.update_animation, init_func=self.init_animation, repeat=False, frames=self.Nt, blit=True)
        plt.show()