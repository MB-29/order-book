"""
LLOB: Locally Linear Order Book simulations.

Implementation for the article "Market impact in a multiple metaorder landscape"
by Blanke, Moran, Crépin, Bouchaud, and Benzaquen.

Example usage:
    from llob import Simulation, courant_dt

    book = LinearDiscreteBook.from_params(D=0.5, L=10.0, xmin=-50, xmax=50, n_grid=100)
    sim = Simulation(
        book=book,
        duration=100.0,
        n_frames=100,
        dt_step=courant_dt(book.dx, book.D),
        metaorder=[1.0],
    )
    sim.run()
    print(sim.prices[-1])
"""

from .books import (
    DiscreteBook,
    IntegerFloorWarning,
    LimitOrders,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
)
from .monte_carlo import MonteCarlo
from .simulation import Simulation, courant_dt, standard_parameters

__all__ = [
    "Simulation",
    "MonteCarlo",
    "courant_dt",
    "standard_parameters",
    "DiscreteBook",
    "LinearDiscreteBook",
    "LinearContinuousBook",
    "MultiDiscreteBook",
    "LimitOrders",
    "IntegerFloorWarning",
]
__author__ = "Matthieu Blanke"
__version__ = "1.0.0"
