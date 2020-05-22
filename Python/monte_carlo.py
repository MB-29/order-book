import numpy as np
from fbm import fgn
from tqdm.auto import tqdm
from simulation import Simulation


class MonteCarlo:
    """Implements a Monte Carlo simulation of order book dynamics with noisy meta-order
    """

    # TODO : support n_start, n_end

    def __init__(self, N_samples, noise_args, simulation_args):
        """
        Arguments:
            N_samples {int} -- number of samples
            noise_args {dictionary} -- {'m0' : noise mean, sigma' : noise size, 'hurst' : hurst exponent}
            simulation_args {dictionary} -- see class Simulation
        """

        self.N_samples = N_samples
        self.Nt = simulation_args.get('Nt')
        self.T = simulation_args.get('T')
        self.time_interval, self.tstep = np.linspace(
            0, self.T, num=self.Nt, retstep=True)

        self.simulation_args = simulation_args

        self.m0 = noise_args.get('m0', 0)
        self.sigma = noise_args.get('sigma', 0)
        self.hurst = noise_args.get('hurst', 0.75)
        self.gamma = 2*(1 - self.hurst)
        self.noise = np.zeros((self.Nt, N_samples))
        self.price_samples = np.zeros((self.Nt, N_samples))

    def generate_noise(self):

        # Standard fractional Gaussian noise
        for sample_index in range(self.N_samples):
            self.noise[:, sample_index] = fgn(
                n=self.Nt, hurst=self.hurst, length=self.T)

        # Scale and translate
        self.scale = self.m0 * self.sigma / (self.tstep ** self.hurst)
        self.noisy_metaorders = self.m0 + self.scale * self.noise
        print(
            f'Generated noise has mean {self.noisy_metaorders.mean().mean():.2f} and variance {self.noisy_metaorders.var(axis=1).mean():.2f}')

    def run(self):
        self.generate_noise()
        args = self.simulation_args
        self.simulation = Simulation(**args)
        print(self.simulation)
        for k in tqdm(range(self.N_samples)):
            args['metaorder'] = self.noisy_metaorders[:, k]
            self.simulation = Simulation(**args)
            self.simulation.run(animation=False)
            self.price_samples[:, k] = self.simulation.prices

        self.compute_statistics()
        self.compute_theory()

    def compute_statistics(self):
        self.price_mean = self.price_samples.mean(axis=1)
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
                    self.growth_th_high + self.price_mean[0], label='high regime', lw=1, color='orange')

        # Scale
        ax.set_yscale(scale)
        ax.set_xscale(scale)

        ax.legend()
        # ax.set_title('Price evolution')

        return ax

    def plot_variance(self, ax, scale='linear'):

        # Lines
        ax.plot(self.time_interval[10:],
                self.price_variance[10:], label='price variance')

        # Scale
        ax.set_yscale(scale)
        ax.set_xscale(scale)

        ax.legend()
        # ax.set_title('Variance evolution')

        return ax

    def gather_results(self):

        return {'mean': self.price_mean,
                'variance': self.price_variance,
                'sigma': self.sigma,
                'N_samples': self.N_samples,
                'm0': self.m0,
                'hurst': self.hurst,
                'params': self.simulation_args}
