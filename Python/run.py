import matplotlib.pyplot as plt
import numpy as np
import json
import os

from simulation import Simulation, standard_parameters

T = 1
Nt = 100
time_interval, tstep = np.linspace(0, T, num=Nt, retstep=True)
n_relax = 100
dt = tstep/n_relax

size = 1
Nx = 10001

n_start, n_end = 0, Nt
L = 5
m0 = 0.1
D = 0.5
metaorder_args = {
    'metaorder': [m0],
    'm0': m0,
    'n_start': n_start,
    'n_end': n_end,
}

dbook_args = {
    'Nx': Nx,
    'xmin': -size/10,
    'xmax': size,
    'initial_density': 'empty',
    'L': L,
    'n_relax': n_relax
}

cbook_args = {
    'dt': T/float(Nt),
    'D': D,
    'L': L,
    'Nx': Nx,
    'xmin': -size,
    'xmax': size
}

participation_rate = 200
# Run
model_type = 'continuous'
# model_type = 'discrete'
standard_args = standard_parameters(participation_rate, model_type, Nx=Nx)
print(standard_args)
args_path = os.path.join('..', 'presets', 'high', 'discrete.json')
with open(args_path, 'r') as args_file:
    json_args = json.load(args_file)

book_args = dbook_args if model_type == 'discrete' else cbook_args
args = {'Nt': Nt,
        'T': T,
        'book_args': book_args,
        'metaorder_args': metaorder_args,
        'model_type': model_type,
        'price_formula': 'vwap'}

# simulation = Simulation(**args)
# simulation = Simulation(**json_args)
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
