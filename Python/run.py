import matplotlib.pyplot as plt
import numpy as np
import json
import os
from fbm import fgn

from simulation import Simulation, standard_parameters

participation_rate = 1e5

# Run
model_type = 'continuous'
model_type = 'discrete'
standard_args = standard_parameters(participation_rate, model_type)
print(f'Standard arguments : {standard_args}')

# Add noise, to ignore comment out or set sigma = 0
T, Nt = standard_args['T'], standard_args['Nt']
sigma, hurst = 0, 0.75
noise = fgn(n=Nt, hurst=hurst, length=T)
m0 = standard_args['metaorder'][0]
standard_args['metaorder'] = m0 + sigma * m0 / \
    (standard_args['T']/standard_args['Nt'] ** hurst) * noise

simulation = Simulation(**standard_args)
print(simulation)

fig = plt.figure(figsize=(12, 6))
simulation.run(animation=True, fig=fig)
plt.show()

fig2 = plt.figure(figsize=(10, 6))
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)
simulation.compute_theoretical_growth()
simulation.plot_price(ax1, high=True)
simulation.plot_err(ax2, relative=False)
plt.show()
