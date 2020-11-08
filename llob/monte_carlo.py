import numpy as np
import pandas as pd
from fbm import fgn
from tqdm.auto import tqdm
from retry import retry

from simulation import Simulation


class MonteCarlo:
    """Implements a Monte Carlo simulation of order book dynamics with noisy meta-order
    """

    def __init__(self, N_samples, noise_args, simulation_args, **kwargs):
        """
        :param N_samples: [description]
        :type N_samples: int
        :param noise_args: {'m0': noise mean , 'm1': noise std, 'hurst': hurst exponent}
        :type noise_args: dictionary
        :param simulation_args: See class Simulation
        :type simulation_args: dictionary
        """

        self.N_samples = N_samples
        self.Nt = simulation_args['Nt']
        self.T = simulation_args['T']

        # Measurements
        self.simulation_args = simulation_args
        self.measured_quantities = simulation_args.get( 
            'measured_quantities', [])
        self.sample_measurements = simulation_args.get(
            'sample_measurements', [])
        self.measured_samples = {}
        for quantity in self.measured_quantities:
            self.measured_samples[quantity] = []
        self.measurement_indices = simulation_args.get(
            'measurement_indices', [])
        self.measurement_slice = simulation_args.get(
            'measurement_slice', 1)

        # Input
        self.m0 = noise_args.get('m0', 0)
        self.m1 = noise_args.get('m1', 0)
        self.hurst = noise_args['hurst']
        self.noise = np.zeros((self.T, N_samples))

        # Output
        self.price_samples = np.zeros((self.T, N_samples))
        self.ask_samples = np.zeros((self.T, N_samples))
        self.bid_samples = np.zeros((self.T, N_samples))

    def generate_noise(self):

        self.noisy_metaorders = np.full(
            (self.T, self.N_samples), self.m0, dtype=float)
        if self.m1 == 0:
            return

        # Standard fractional Gaussian noise
        for sample_index in range(self.N_samples):
            self.noise[:, sample_index] =fgn(
                n=self.T, hurst=self.hurst, length=self.T)

        # Scale and translate
        self.scale = self.m1
        self.noisy_metaorders += self.scale * self.noise
        print(
            f'Generated meta-order has mean {self.noisy_metaorders.mean().mean():.2f} '
            f'and variance {self.noisy_metaorders.var(axis=1).mean():.2f}')

    def run(self):
        self.generate_noise()
        args = self.simulation_args
        self.simulation = Simulation(**args)
        print(self.simulation)
        for k in tqdm(range(self.N_samples)):
            args['metaorder'] = self.noisy_metaorders[:, k]

            self.try_running(args)

            self.price_samples[:, k] = self.simulation.prices
            self.ask_samples[:, k] = self.simulation.asks
            self.bid_samples[:, k] = self.simulation.bids

            for quantity in self.measured_quantities:
                self.measured_samples[quantity].append(
                    np.copy(self.simulation.measurements[quantity]))

        self.compute_statistics()

    def compute_statistics(self):

        self.price_mean = self.price_samples.mean(axis=1)
        self.ask_mean = self.ask_samples.mean(axis=1)
        self.bid_mean = self.bid_samples.mean(axis=1)

        self.price_variance = self.price_samples.var(axis=1)
        self.ask_variance = self.ask_samples.var(axis=1)
        self.bid_variance = self.bid_samples.var(axis=1)

        self.measurement_means = {}
        self.measurement_vars = {}
        for quantity in self.measured_quantities:
            samples = np.array(self.measured_samples[quantity])
            self.measurement_means[quantity] = np.mean(samples, axis=0)
            self.measurement_vars[quantity] = np.var(samples, axis=0)

    def gather_results(self):

        result = {'price_mean': self.price_mean,
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

        for quantity in self.measured_quantities:
            result[f'{quantity}_mean'] = self.measurement_means[quantity]
            result[f'{quantity}_variance'] = self.measurement_vars[quantity]

        for sample_measurement in self.sample_measurements:
            measurement = [getattr(self, f'{sample_measurement}_samples')[time_index: time_index+self.measurement_slice, :] for time_index in self.measurement_indices]
            result[sample_measurement] = measurement

        return result


    @retry((ValueError, IndexError), tries=1)
    def try_running(self, args):
        """Run a simulation, retrying in case of error.
        """
        self.simulation = Simulation(**args)
        self.simulation.run()
