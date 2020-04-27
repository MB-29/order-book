import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation

T = 1
Nt = 100
dt = T/float(Nt)
time_interval = np.linspace(0, T, num=Nt)

size = 1
Nx = 1000
dx = 2*size/Nx

lambd = 10
nu = 1

n_start, n_end = 10, Nt
m0 = 10000
L = 10

order_args = {
    'nu': nu,
    'lambd': lambd,
    'dt': dt,
    'Nx': Nx,
    'lower_bound': -50,
    'upper_bound': 50,
    'initial_density': 'linear',
    'L': L
}
order_args2 = {
    'dt': T/float(Nt),
    'D': dx * dx / (2 * dt),
    'L': L,
    'Nx': Nx,
    'lower_bound': -50,
    'upper_bound': 50
}


# Run
model_type = 'continuous'
model_type = 'discrete'
args = order_args if model_type == 'discrete' else order_args2
simulation = Simulation(args, T, Nt, m0, n_start=0,
                        n_end=Nt, model_type=model_type)
fig = plt.figure(figsize=(12, 6))
simulation.run(animation=True, fig=fig)
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
simulation.plot_price(ax1, high=True)
simulation.plot_err(ax2, relative=True)
plt.show()
