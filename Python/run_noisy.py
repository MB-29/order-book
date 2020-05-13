import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import os

from monte_carlo import MonteCarlo


args_path = os.path.join('..', 'presets', 'high', 'continuous.json')
with open(args_path, 'r') as args_file:
    simulation_args = json.load(args_file)
m0 = simulation_args['metaorder_args']['m0']
T = simulation_args.get('T')
N_samples = 500
hurst = 0.65
sigma = 0

noise_args = {
    'sigma': sigma,
    'hurst': hurst
}

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

noisy_simulation = MonteCarlo(N_samples, noise_args, simulation_args)
pure_simulation = MonteCarlo(1, {}, simulation_args)

noisy_simulation.run()
noisy_simulation.price_mean[0] = 0
pure_simulation.run()

variance_exp = 2 * hurst - 1
err_exp = 2 * hurst - 3/2

ax1.plot(pure_simulation.time_interval, pure_simulation.price_mean, label='no noise', color='blue', lw=2)
noisy_simulation.plot_price(ax1, high=True)

A = noisy_simulation.simulation.A_low if noisy_simulation.simulation.participation_rate < 0.5 else noisy_simulation.simulation.A_high
price_th = A * np.sqrt(noisy_simulation.time_interval)

error = price_th - noisy_simulation.price_mean - (1/2) * (price_th  - (pure_simulation.price_mean ** 2) / price_th)

var_rescaling = noisy_simulation.price_variance[-1] / (T ** variance_exp)
err_rescaling = abs(error[-1] / (T ** err_exp))
offset = 10

_, X, _ = ax2.hist(
    noisy_simulation.noisy_metaorders[0, :], bins='sqrt', density=True)
# _, X, _ = ax2.hist(noisy_simulation.noise[0, :], bins='sqrt', density=True)
gaussian = np.vectorize(lambda x: norm.pdf(x, m0, sigma * m0))(X)
# gaussian = np.vectorize(lambda x : norm.pdf(x, 0, 0.1))(X)
ax2.plot(X, gaussian, color='red', label='gaussian')
ax2.legend()

ax3.plot(noisy_simulation.time_interval[offset:],
         noisy_simulation.price_variance[offset:], label='price variance')
ax3.plot(noisy_simulation.time_interval[offset:], var_rescaling *
         (noisy_simulation.time_interval[offset:] ** variance_exp), label=r'$y=Ct ^ {2H - 1}$')
ax3.set_title('Variance')
ax3.legend()
# noisy_simulation.plot_variance(ax3)

ax4.plot(noisy_simulation.time_interval[offset:],
         noisy_simulation.price_variance[offset:], label='price variance')
ax4.plot(noisy_simulation.time_interval[offset:], var_rescaling *
         (noisy_simulation.time_interval[offset:] ** variance_exp), label=r'$y=Ct ^ {2H - 1}$')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_title('Variance')
ax4.legend()
# noisy_simulation.plot_variance(ax4, scale = 'log')


ax5.plot(noisy_simulation.time_interval, error,
         label=r'$I_t - \mathbb{E}(p_t) - N_t/2I_t $', color='red')
# ax5.plot(pure_simulation.time_interval, pure_simulation.price_mean -
#          noisy_simulation.price_mean, label='noise term', color='blue')
ax5.legend()

ax6.plot(noisy_simulation.time_interval, error,
         label=r'$I_t - \mathbb{E}(p_t) - N_t/2I_t $', color='red')
# ax6.plot(noisy_simulation.time_interval, err_rescaling *
#          (noisy_simulation.time_interval ** err_exp), label=r'$y=Ct ^ {2H - 3/2}$')
# ax6.plot(pure_simulation.time_interval[offset:], (pure_simulation.price_mean - price_th)
#          [offset:], label='price error (no noise)', color='blue')
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.legend()
# ax6.plot(time_interval[offset:], (price_mean - vanilla_price_mean -
#                                   price_th)[offset:], label='noise contribution')


plt.suptitle(
    fr'$ H={hurst: .2f}$, $ m_0={noisy_simulation.m0: .1f}$'
    fr'$, r={noisy_simulation.simulation.participation_rate: .1e}$'
    fr', $\sigma={sigma:.2f}$,  {N_samples} samples')

# plt.legend()
plt.show()
