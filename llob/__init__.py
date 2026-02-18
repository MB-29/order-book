"""
LLOB: Locally Linear Order Book simulations.

Implementation for the article "Market impact in a multiple metaorder landscape"
by Blanke, Moran, Crépin, Bouchaud, and Benzaquen.

Example usage:
    from llob import Simulation, standard_parameters

    params = standard_parameters(participation_rate=1.0, model_type='discrete')
    sim = Simulation.from_params(**params)
    sim.run()
    print(sim.prices[-1])
"""
from .books import (
    DiscreteBook,
    LimitOrders,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
)
from .monte_carlo import MonteCarlo
from .simulation import Simulation, standard_parameters

__all__ = [
    # Main entry points
    "Simulation",
    "MonteCarlo",
    "standard_parameters",
    # Book classes (for advanced usage)
    "DiscreteBook",
    "LinearDiscreteBook",
    "LinearContinuousBook",
    "MultiDiscreteBook",
    "LimitOrders",
]
__author__ = "Matthieu Blanke"
__version__ = "1.0.0"
