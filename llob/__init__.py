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

Using pydantic configs (preferred):
    from llob import Simulation, SimulationConfig, GridConfig

    config = SimulationConfig(
        model_type='discrete',
        grid=GridConfig(xmin=-50, xmax=50, Nx=100),
        D=1.0,
        L=1.0,
        T=100,
        Nt=100,
        metaorder=[0.5],
    )
    sim = Simulation.from_config(config)
    sim.run()
"""
from .books import (
    DiscreteBook,
    LimitOrders,
    LinearContinuousBook,
    LinearDiscreteBook,
    MultiDiscreteBook,
)
from .configs import (
    BookConfig,
    DiscreteBookConfig,
    GridConfig,
    LimitOrdersConfig,
    LinearContinuousBookConfig,
    LinearDiscreteBookConfig,
    MonteCarloConfig,
    MultiDiscreteBookConfig,
    NoiseConfig,
    SimulationConfig,
)
from .monte_carlo import MonteCarlo
from .simulation import Simulation, standard_parameters

__all__ = [
    # Main entry points
    "Simulation",
    "MonteCarlo",
    "standard_parameters",
    # Configuration classes
    "GridConfig",
    "SimulationConfig",
    "MonteCarloConfig",
    "NoiseConfig",
    # Book configs
    "BookConfig",
    "DiscreteBookConfig",
    "LinearDiscreteBookConfig",
    "LinearContinuousBookConfig",
    "MultiDiscreteBookConfig",
    "LimitOrdersConfig",
    # Book classes (for advanced usage)
    "DiscreteBook",
    "LinearDiscreteBook",
    "LinearContinuousBook",
    "MultiDiscreteBook",
    "LimitOrders",
]
__author__ = "Matthieu Blanke"
__version__ = "1.0.0"
