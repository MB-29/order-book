import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from cycler import cycler
from scipy.stats import norm
import json
import os
import pickle
import pandas as pd
from scipy.optimize import curve_fit


def power_law(x, a, b):
    return a*np.power(x, b)

# Styles

plt.style.use('seaborn-deep')


# rc('text', usetex=True)
# rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])
# rcParams['axes.prop_cycle'] = cycler(color=['#465987', '#3F7A46', 'red', 'y'])

# Params

path = '../data/multi/proportions_short_time_T10.pkl'
path = '../data/multi/proportions_long_time_frozen.pkl'
path = '../data/multi/proportions_long_time_N10.pkl'
with open(path, 'rb') as pickle_file:
    output = pickle.load(pickle_file)

params = output['params']
print(params)
D, L, nu, m0 = params['D'], params['L'], params['nu'], output['m0']
T, Nt = params['T'], params['Nt']
time_interval, dt = np.linspace(0, T, Nt, retstep=True)
N_samples = output['N_samples']
t_star = D *L[0]**2/ (2 * nu[0] *L[1] * m0)
xc = L[0] * np.sqrt(D / nu[0]) / L[1]
print(f't_star = { t_star}')
print(f'xc = { xc}')
print(f'1/(2 t_star) = {1/(2 * t_star)}')
print(f'price factor = {xc/(2 * t_star)}')

# Output

proportions_mean = output['actor_trades_mean']
price = output['price_mean']
proportion_yerr = 4*np.sqrt(output['actor_trades_variance'] / N_samples)
price_yerr = np.sqrt(output['price_variance'] / N_samples)

# Regression

regression_params, cov = curve_fit(f=power_law, xdata=time_interval[5:30], ydata=proportions_mean[5:30, 0], p0=[
    0, 0], bounds=(-np.inf, np.inf))
stdevs = np.sqrt(np.diag(cov))
print(f'a = {regression_params[0]} +- {stdevs[0]}')
print(f'b= {regression_params[1]} +- {stdevs[1]}')

# Figure

fig = plt.figure()
ax1 = fig.add_subplot('211')
ax2 = fig.add_subplot('212')

# plt.plot(time_interval, price)

# ax1.plot(time_interval, power_law(time_interval, *regression_params))
ax1.errorbar(time_interval, proportions_mean[:, 0], yerr=proportion_yerr[:, 0], label='slow')

ax2.plot(time_interval, power_law(time_interval, *regression_params))
ax2.errorbar(time_interval, proportions_mean[:, 0], yerr=proportion_yerr[:, 0], label='slow')
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.show()

# # Regression

# regression_params, cov = curve_fit(f=power_law, xdata=time_interval[200:], ydata=price[200:], p0=[
#     0, 0], bounds=(-np.inf, np.inf))
# stdevs = np.sqrt(np.diag(cov))
# print(f'a = {regression_params[0]} +- {stdevs[0]}')
# print(f'b= {regression_params[1]} +- {stdevs[1]}')

# # Figure

# fig = plt.figure()
# ax1 = fig.add_subplot('211')
# ax2 = fig.add_subplot('212')

# # plt.plot(time_interval, price)

# ax1.plot(time_interval, power_law(time_interval, *regression_params), ls='--')
# ax1.errorbar(time_interval, price, yerr=price_yerr, label='slow')

# ax2.plot(time_interval, power_law(time_interval, *regression_params))
# ax2.errorbar(time_interval, price, yerr=price_yerr, label='slow')
# ax2.set_xscale('log')
# ax2.set_yscale('log')
# plt.show()
