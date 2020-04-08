import numpy as np
import matplotlib.pyplot as plt

from order_book import OrderBook


class Simulation:

    def __init__(self, book_parameters, T, Nt, m0, n_start=None, n_end=None, plot=False):

        # Order book
        self.book_parameters = book_parameters
        self.book = OrderBook(**book_parameters)
        self.J = self.book.J

        # Time evolution
        self.Nt = Nt
        self.prices = np.zeros(Nt)
        self.time_interval, self.tstep = np.linspace(
            0, T, num=Nt, retstep=True)
        self.n_start = Nt//10 if not n_start else n_start
        self.n_end = Nt//2 if not n_end else n_end
        self.t_start = self.n_start * self.tstep
        self.t_end = self.n_end * self.tstep
        self.time_interval_shifted = self.time_interval-self.t_start

        # Metaorder
        self.m0 = m0

        # Theoretical constants
        self.A_low = m0/(self.J*np.sqrt(np.pi))
        self.A_high = np.sqrt(2*m0/self.J)
        self.participation_rate = self.m0/self.J

        # Plot
        self.plot = plot

    def run(self, animation=False):

        self.growth_th = self.A_low * \
            np.sqrt(self.book.D *
                    self.time_interval_shifted[self.n_start:self.n_end])

        for n in range(self.Nt):

            # Update metaorder intensity
            mt = self.m0 if (n >= self.n_start and n <= self.n_end) else 0

            self.book.mt = mt
            self.book.timestep()
            self.prices[n] = self.book.price

            if animation:
                ymin, ymax = -1, 1
                plt.ylim((ymin, ymax))
                plt.plot(self.book.X, self.book.density,
                         label='order density', linewidth=1)
                plt.vlines(self.book.price, -1, 1, label='price', color='red', linewidth=1)
                plt.hlines(0, -1, 1, color='black', linewidth=0.5, linestyle='dashed')

                plt.legend()
                plt.pause(1)
                plt.cla()

        self.peak_value = self.prices[self.n_end]
        

    def compute_growth_mean_error(self, ord=2):
        self.growth_mean_error = np.linalg.norm(
            self.growth_th - self.prices[self.n_start:self.n_end], ord=ord) / np.power(self.n_end - self.n_start, 1/ord)
        return self.growth_mean_error

    def plot_price(self, ax, symlog=False):

        # Curves
        ax.plot(self.time_interval_shifted,
                 self.prices, label='price evolution')
        ax.plot(self.time_interval_shifted[self.n_start:self.n_end],
                 self.growth_th, label='theoretical impact', lw=1, color='green')

        # Scale
        if symlog:
            ax.set_yscale('symlog', linthreshy=1e-1)
            ax.set_xscale('symlog', linthreshx=self.tstep)


        ax.legend(loc='lower right')
        ax.set_title('Price evolution')

        return ax

    def plot_err(self, ax, relative=False, symlog=False):

        self.growth_error = abs(self.prices[self.n_start:self.n_end] - self.growth_th)
        if relative :
            self.growth_error = self.growth_error/self.growth_th

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
