import matplotlib.pyplot as plt
import numpy as np
import json
import os

from simulation import Simulation

T = 1000
Nt = 100
time_interval, tstep = np.linspace(0, T, num=Nt, retstep=True)
n_relax = 100
dt = tstep/n_relax

size = 1
Nx = 501

n_start, n_end = 0, Nt
L = 100000
m0 = 1
metaorder_args = {
    'metaorder': [m0],
    'm0': m0,
    'n_start': n_start,
    'n_end': n_end,
}

dbook_args = {
    'Nx': Nx,
    'lower_bound': -size,
    'upper_bound': size,
    'initial_density': 'empty',
    'L': L,
    'n_relax': n_relax
}

cbook_args = {
    'dt': T/float(Nt),
    'D': 0.0,
    'L': L,
    'Nx': Nx,
    'lower_bound': -size,
    'upper_bound': size
}


# Run
model_type = 'continuous'
# model_type = 'discrete'

args_path = os.path.join('..', 'presets', 'high', 'discrete.json')
with open(args_path, 'r') as args_file:
    json_args = json.load(args_file)

book_args = dbook_args if model_type == 'discrete' else cbook_args
args = {'Nt': Nt,
        'T': T,
        'book_args': book_args,
        'metaorder_args': metaorder_args,
        'model_type': model_type}

simulation = Simulation(**args)
simulation = Simulation(**json_args)

print(simulation)

fig = plt.figure(figsize=(12, 6))
simulation.run(animation=True, fig=fig)
plt.show()

fig2 = plt.figure(figsize=(10, 8))
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)
simulation.compute_theoretical_growth()
simulation.plot_price(ax1, high=True)
simulation.plot_err(ax2, relative=True)
plt.show()
