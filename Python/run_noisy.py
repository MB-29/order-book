import matplotlib.pyplot as plt
import json
import os

from monte_carlo import MonteCarlo

args_path = os.path.join('..', 'presets', 'high', 'discrete2.json')
with open(args_path, 'r') as args_file:
    simulation_args = json.load(args_file)

N_samples = 10
hurst = 0.75
sigma = 1

noise_args = {
    'sigma': sigma,
    'hurst': hurst
}


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

noisy_simulation = MonteCarlo(N_samples, noise_args, simulation_args)

noisy_simulation.run()

noisy_simulation.plot_price(ax1, high=True)
noisy_simulation.plot_price(ax2, scale='log')
noisy_simulation.plot_variance(ax3)
noisy_simulation.plot_variance(ax4, scale='log')

plt.show()
