import matplotlib.pyplot as plt
import numpy as np
import json
import os

from simulation import Simulation, standard_parameters

participation_rate = float('inf')

# Run
model_type = 'continuous'
# model_type = 'discrete'
standard_args = standard_parameters(participation_rate, model_type)
print(f'Standard arguments : {standard_args}')

simulation = Simulation(**standard_args)


print(simulation)

fig = plt.figure(figsize=(12, 6))
simulation.run(animation=True, fig=fig)
plt.show()

fig2 = plt.figure(figsize=(10, 6))
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)
simulation.compute_theoretical_growth()
simulation.plot_price(ax1, low=True)
simulation.plot_err(ax2, relative=False)
plt.show()
