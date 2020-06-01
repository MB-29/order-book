import matplotlib.pyplot as plt
import numpy as np
import json
import os
from fbm import fgn
import time

from simulation import Simulation, standard_parameters

participation_rate = 500
# participation_rate = float('inf')

# Run
model_type = 'continuous'
model_type = 'discrete'
standard_args = standard_parameters(participation_rate, model_type, xmin=-1, Nx=8000)
print(f'Standard arguments : {standard_args}')

# Add noise. To ignore noise, comment out or set m1 = 0 
# T, Nt = standard_args['T'], standard_args['Nt']
# m0 = standard_args['metaorder'][0]
# m1, hurst = m0, 0.75
# noise = fgn(n=Nt, hurst=hurst, length=T)
# standard_args['metaorder'] = m1 / \
#     (standard_args['T']/standard_args['Nt'] ** hurst) * noise

simulation = Simulation(**standard_args)
# print(simulation)

fig = plt.figure(figsize=(12, 6))
tic = time.perf_counter()
simulation.run(animation=False, fig=fig, save=False)
toc = time.perf_counter()
print(f'Execution time : {toc - tic}')
plt.show()

fig2 = plt.figure(figsize=(10, 6))
ax1 = fig2.add_subplot(2, 1, 1)

ax1.plot(simulation.time_interval_shifted, simulation.prices, label='price')
# ax1.plot(simulation.time_interval_shifted[simulation.n_start: simulation.n_end],
#          simulation.get_growth_th(), label='theoretical')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
plt.show()
