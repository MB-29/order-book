import matplotlib.pyplot as plt
import numpy as np

from order_book import OrderBook
from simulation import Simulation

# Simulation parameters
T = 1
Nt = 50

# Model parameters
book_parameters = {
    'dt': T/float(Nt),
    'D': 1,
    'L' : 1,
    'Nx': 2000,
    'lower_bound': -100,
    'upper_bound': 100
    }

# Metaorder
m0 = 100

# Run
simulation = Simulation(book_parameters, T, Nt, m0)
simulation.run(animation=True)
simulation.plot_price(plt.gca())
plt.show()