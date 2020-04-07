import numpy as np
import matplotlib.pyplot as plt

from order_book import OrderBook


class Simulation:

    def __init__(self, book_parameters, T, Nt, m0, plot=False):

        # Order book
        self.book_parameters = book_parameters
        self.book = OrderBook(**book_parameters)
        self.J = self.book.J

        # Time evolution
        self.Nt = Nt
        self.prices = np.zeros(Nt)
        self.time_interval, self.tstep = np.linspace(0, T, num=Nt, retstep=True)
        self.n_start = Nt//5
        self.n_end = 2 * Nt//3
        self.t_start = self.n_start * self.tstep

        # Metaorder
        self.m0 = m0

        # Theoretical constants
        self.A_low = m0/(self.J*np.sqrt(np.pi))
        self.A_high = np.sqrt(2*m0/self.J)

        # Plot
        self.plot = plot

    def run(self):

        self.growth_th = self.A_low * \
            np.sqrt(self.book.D *
                    self.time_interval[0:self.n_end-self.n_start])

        for n in range(self.Nt):

            # Update metaorder intensity
            mt = self.m0 if (n >= self.n_start and n <= self.n_end) else 0

            self.book.mt = 1
            self.book.mt = mt
            self.book.timestep()
            self.prices[n] = self.book.price

            if self.plot:
                ymin, ymax = -1, 1
                plt.ylim((ymin, ymax))
                plt.plot(self.book.X, self.book.density,
                         label='order density', linewidth=1)
                plt.legend()
                plt.pause(0.001)
                plt.cla()

    def compute_growth_MSE(self, ord=2):
        self.growth_mean_error = np.linalg.norm(
            self.growth_th - self.prices[self.n_start:self.n_end], ord=ord) / np.sqrt(self.n_end- self.n_start)
        return self.growth_mean_error

    def plot_vs_time(self, symlog=True):

        ax = plt.gca()

        plt.plot(self.time_interval-self.t_start, self.prices, label='price evolution')
        plt.plot(self.time_interval[self.n_start:self.n_end]-self.t_start,
         self.growth_th, label='theoretical impact', lw=1, color='green')

        if symlog:
            plt.yscale('symlog', linthreshy = 1e-1)
            plt.xscale('symlog', linthreshx = self.tstep)

        # Titles

        title = r'd$t={{{dt}}}$, $\lambda={{{lambd}}}$, $\nu={{{nu}}}$, $D={{{D}}}$'.format(
            **self.book_parameters)
        text = r'$m_0={{{0}}}$, $J={{{1}}}$'.format(self.m0, round(self.J, 2))

        plt.legend(loc='lower right')
        plt.title(title)
        plt.text(0.01,0.92,text, transform=ax.transAxes)

        plt.show()