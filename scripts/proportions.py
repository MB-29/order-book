import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from cycler import cycler
import numpy as np
import json
import os
from fbm import fgn
import time
import pickle

from simulation import Simulation, standard_parameters
from monte_carlo import MonteCarlo


plt.style.use('seaborn-deep')


# rc('text', usetex=True)
# rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])
# rcParams['axes.prop_cycle'] = cycler(color=['#465987', '#3F7A46', 'red', 'y'])


# Run
model_type = 'continuous'
model_type = 'discrete'
# standard_args = standard_parameters(participation_rate, model_type, xmin=-1, Nx=8000)
# print(f'Standard arguments : {standard_args}')
# standard_args['nu'] = 0.1

T, Nt = 15000, 300
L, nu = np.array([50, 1]), np.array([1, 0])
m0 = 100
X = 1800
xmin, xmax = -20, X-20
Nx = int(xmax - xmin)
simulation_args = {
    'Nx': Nx,
    'Nt': Nt,
    'xmin': xmin,
    'xmax': xmax,
    'nu': nu,
    'L': L,
    'D': 0.5,
    'T': T,
    'model_type': model_type,
    'measured_quantities': ['actor_trades'],
    'metaorder': [m0]
}

# Sample


# simulation = Simulation(**simulation_args)
# print(simulation)
# # fig = plt.figure(figsize=(12, 6))
# # simulation.run(animation=True, fig=fig, save=False)
# simulation.run(animation=False, fig=None, save=False)
# # plt.show()

# proportions = simulation.measurements['actor_trades']
# plt.plot(time_interval, proportions)
# plt.show()

# Monte Carlo
N_samples = 5
noise_args = {
    'm0': m0,
    'm1': 0,
    'hurst': 0.7
}
noisy_simulation = MonteCarlo(N_samples, noise_args, simulation_args)

noisy_simulation.run()
output = noisy_simulation.gather_results()
proportions_mean = output['actor_trades_mean']
yerr = np.sqrt(output['actor_trades_variance'] / N_samples)

with open(f'../proportions_long_time.pkl', 'wb') as pickle_file:
    pickle.dump(output, pickle_file)

fig = plt.figure()
ax1 = fig.add_subplot('211')
ax2 = fig.add_subplot('212')

ax1.errorbar(time_interval, proportions_mean[:, 0], yerr=yerr[:, 0], label='fast')
ax1.errorbar(time_interval, proportions_mean[:, 1], yerr=yerr[:, 1], label='slow')
ax1.legend()
ax2.plot(proportions_mean)
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.show()

