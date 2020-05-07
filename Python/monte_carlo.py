import numpy as np
from fbm import fgn

from simulation import Simulation


class MonteCarlo:
    """Implements a Monte Carlo simulation of order book dynamics with noisy meta-order
    """

    # TODO : support n_start, n_end

    def __init__(self, N_samples, noise_args, simulation_args):
        """
        Arguments:
            N_samples {int} -- number of samples
            noise_args {dictionary} -- {'sigma' : noise size, 'hurst' : hurst exponent}
            simulation_args {dictionary} -- see class Simulation
        """

        self.N_samples = N_samples
        self.Nt = simulation_args.get('Nt')
        self.T = simulation_args.get('T')
        self.time_interval = np.linspace(0, self.T, num=self.Nt)

        self.simulation_args = simulation_args

        self.m0 = simulation_args['metaorder_args'].get('m0')
        self.sigma = noise_args.get('sigma')
        self.hurst = noise_args.get('hurst')
        self.gamma = 2*(1 - self.hurst)
        self.noise = np.zeros((self.Nt, N_samples))
        self.price_samples = np.zeros((self.Nt, N_samples))
        self.vanilla_prices = np.zeros((self.Nt, N_samples))

    def generate_noise(self):

        # Standard fractional Gaussian noise
        for sample_index in range(self.N_samples):
            self.noise[:, sample_index] = fgn(
                n=self.Nt, hurst=self.hurst, length=self.T)

        # Scale and translate
        self.noisy_metaorders = self.m0 + self.m0 * self.sigma * self.noise

    def run(self):
        self.generate_noise()
        args = self.simulation_args
        for k in range(self.N_samples):
            # without noise
            args['metaorder_args']['metaorder'] = [self.m0]
            self.simulation = Simulation(**args)
            self.simulation.run()
            self.vanilla_prices[:, k] = self.simulation.prices

            # with noise
            args['metaorder_args']['metaorder'] = self.noisy_metaorders[:, k]
            self.simulation = Simulation(**args)
            self.simulation.run(animation=False)
            self.price_samples[:, k] = self.simulation.prices

            if (k+1) % (max(self.N_samples//10, 1)) == 0:
                print(f'Performed {k+1} simulations')

        self.compute_statistics()
        self.compute_theory()

    def compute_statistics(self):
        self.price_mean = self.price_samples.mean(axis=1)
        self.vanilla_price_mean = self.vanilla_prices.mean(axis=1)
        self.price_variance = self.price_samples.var(axis=1)

    def compute_theory(self):

        self.growth_th_low = self.simulation.A_low * \
            np.sqrt(
                self.time_interval)
        self.growth_th_high = self.simulation.A_high * \
            np.sqrt(
                self.time_interval)

    def plot_price(self, ax, scale='linear', low=False, high=False):

        # Lines
        ax.plot(self.time_interval,
                self.price_mean, label='mean price')

        self.simulation.compute_theoretical_growth()
        if low:
            ax.plot(self.time_interval,
                    self.growth_th_low, label='low regime', lw=1, color='green')
        if high:
            ax.plot(self.time_interval,
                    self.growth_th_high, label='high regime', lw=1, color='orange')

        # Scale
        ax.set_yscale(scale)
        ax.set_xscale(scale)

        ax.legend()
        ax.set_title('Price evolution')

        return ax

    def plot_variance(self, ax, scale='linear'):

        # Lines
        ax.plot(self.time_interval[10:],
                self.price_variance[10:], label='price variance')

        # Scale
        ax.set_yscale(scale)
        ax.set_xscale(scale)

        ax.legend()
        ax.set_title('Variance evolution')

        return ax
