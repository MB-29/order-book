import matplotlib.pyplot as plt
import numpy as np

from order_book import OrderBook
from simulation import Simulation

# Simulation parameters
T = 1
Nt = 100

# Model parameters
book_parameters = {
    'dt': T/float(Nt),
    'lambd': 1,
    'nu': 1e-2,
    'D': 1,
    'Nx': 500
    }

# Metaorder
m0 = 1

# Run
simulation = Simulation(book_parameters, T, Nt, m0)
simulation.run()
simulation.plot_vs_time()
