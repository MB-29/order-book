"""
Agent-based LLOB model with boundary conditions and linear stationary state.
"""
__author__ = 'Matthieu Blanke'
__version__ = '1.0.0'

from discrete_book import DiscreteBook


class LinearDiscreteBook(DiscreteBook):
    """ Implements a discrete order book in the linear regime.
    """

    def __init__(self, **order_args):

        # Check that lambd and nu are either 0 or not defined
        order_args['lambd'] = order_args.get('lambd', 0)
        order_args['nu'] = order_args.get('nu', 0)
        assert order_args['lambd'] == 0 and order_args['nu'] == 0,\
            'Linear model settings'

        order_args['initial_density'] = 'linear'
        order_args['boundary_conditions'] = 'linear'

        DiscreteBook.__init__(self, **order_args)

    # No arrivals nor cancellations, only jumps
    def stochastic_timestep(self):
        for orders in [self.ask_orders, self.bid_orders]:
            orders.jumps()
            orders.update_best_price()
