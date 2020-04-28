import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation

T = 1
Nt = 100
dt = T/float(Nt)
time_interval = np.linspace(0, T, num=Nt)

size = 50
Nx = 1000
dx = 2*size/Nx

lambd = 10
nu = 1
L = 10

# Metaorder
n_start, n_end = 10, Nt
m0 = 1000
metaorder_args = {
    'metaorder': [m0],
    'm0': m0,
    'n_start': n_start,
    'n_end': n_end
}

order_args = {
    'nu': nu,
    'lambd': lambd,
    'dt': dt,
    'Nx': Nx,
    'lower_bound': -size,
    'upper_bound': size,
    'initial_density': 'linear',
    'L': L
}

order_args2 = {
    'dt': T/float(Nt),
    'D': dx*dx/dt,
    'L': L,
    'Nx': Nx,
    'lower_bound': -size,
    'upper_bound': size
}


# Run
model_type = 'continuous'
model_type = 'discrete'
args = order_args if model_type == 'discrete' else order_args2
simulation = Simulation(args, T, Nt, metaorder_args, model_type=model_type)
fig = plt.figure(figsize=(12, 6))
simulation.run(animation=True, fig=fig)
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
simulation.plot_price(ax1, high=True)
simulation.plot_err(ax2, relative=True)
plt.show()
