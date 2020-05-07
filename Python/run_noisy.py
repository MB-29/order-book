import matplotlib.pyplot as plt
import json
import os

from monte_carlo import MonteCarlo


args_path = os.path.join('..', 'presets', 'high', 'discrete2.json')
with open(args_path, 'r') as args_file:
    simulation_args = json.load(args_file)

T = simulation_args.get('T')
N_samples = 200
hurst = 0.75
sigma = 1e-2

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

variance_exp = 2 * hurst - 1
noisy_simulation.plot_price(ax1, high=True)
noisy_simulation.plot_price(ax2, scale='log')

var_rescaling = noisy_simulation.price_variance[-1] / (T ** variance_exp)
offset = 10

ax3.plot(noisy_simulation.time_interval[offset:],
         noisy_simulation.price_variance[offset:], label='price variance')
ax3.plot(noisy_simulation.time_interval[offset:], var_rescaling *
         (noisy_simulation.time_interval[offset:] ** variance_exp), label=r'$y=Ct ^ {1 - \gamma}$')
ax3.set_title('Variance')
# noisy_simulation.plot_variance(ax3)

ax4.plot(noisy_simulation.time_interval[offset:],
         noisy_simulation.price_variance[offset:], label='price variance')
ax4.plot(noisy_simulation.time_interval[offset:], var_rescaling *
         (noisy_simulation.time_interval[offset:] ** variance_exp), label=r'$y=Ct ^ {1 - \gamma}$')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_title('Variance')
# noisy_simulation.plot_variance(ax4, scale = 'log')

plt.suptitle(
    fr'$ H={hurst: .2f}$, $ m_0={noisy_simulation.m0: .1f}$'
    fr'$, r={noisy_simulation.simulation.participation_rate: .1e}$'
    fr', $\sigma={sigma: .1f}$,  {N_samples} samples')

plt.show()
