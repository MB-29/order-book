import numpy as np
from fbm import fgn
from tqdm.auto import tqdm
from simulation import Simulation


class MonteCarlo:
    """Implements a Monte Carlo simulation of order book dynamics with noisy meta-order
    """

    def __init__(self, N_samples, noise_args, simulation_args):
        """
        :param N_samples: [description]
        :type N_samples: int
        :param noise_args: {'m0': noise mean , 'm1': noise std, 'hurst': hurst exponent}
        :type noise_args: dictionary
        :param simulation_args: See class Simulation
        :type simulation_args: dictionary
        """

        self.N_samples = N_samples
        self.Nt = simulation_args.get('Nt')
        self.T = simulation_args.get('T')
        self.time_interval, self.tstep = np.linspace(
            0, self.T, num=self.Nt, retstep=True)

        self.simulation_args = simulation_args

        self.m0 = noise_args.get('m0', 0)
        self.m1 = noise_args.get('m1', 0)
        self.hurst = noise_args.get('hurst', 0.75)
        self.gamma = 2*(1 - self.hurst)
        self.noise = np.zeros((self.Nt, N_samples))
        self.price_samples = np.zeros((self.Nt, N_samples))
        self.ask_samples = np.zeros((self.Nt, N_samples))
        self.bid_samples = np.zeros((self.Nt, N_samples))

    def generate_noise(self):

        # Standard fractional Gaussian noise
        for sample_index in range(self.N_samples):
            self.noise[:, sample_index] = fgn(
                n=self.Nt, hurst=self.hurst, length=self.T)

        # Scale and translate
        self.scale = self.m1 / (self.tstep ** self.hurst)
        self.noisy_metaorders = self.m0 + self.scale * self.noise
        print(
            f'Generated noise has mean {self.noisy_metaorders.mean().mean():.2f} '
            'and variance {self.noisy_metaorders.var(axis=1).mean():.2f}')

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
            self.ask_samples[:, k] = self.simulation.asks
            self.bid_samples[:, k] = self.simulation.bids

        self.compute_statistics()

    def compute_statistics(self):

        self.price_mean = self.price_samples.mean(axis=1)
        self.ask_mean = self.ask_samples.mean(axis=1)
        self.bid_mean = self.bid_samples.mean(axis=1)

        self.price_variance = self.price_samples.var(axis=1)
        self.ask_variance = self.ask_samples.var(axis=1)
        self.bid_variance = self.bid_samples.var(axis=1)

    def gather_results(self):

        return {'price_mean': self.price_mean,
                'price_variance': self.price_variance,
                'ask_mean': self.ask_mean,
                'ask_variance': self.ask_variance,
                'bid_mean': self.bid_mean,
                'bid_variance': self.bid_variance,
                'm1': self.m1,
                'N_samples': self.N_samples,
                'm0': self.m0,
                'hurst': self.hurst,
                'params': self.simulation_args}
