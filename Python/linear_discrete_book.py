from discrete_book import DiscreteBook


class LinearDiscreteBook(DiscreteBook):
    """ Implements a discrete order book in the linear regime.
    """

    def __init__(self, **order_args):
        """Latent liquidity L must be provided.
        """

        order_args['nu'] = 0
        order_args['lambd'] = 0
        L = order_args.get('L')
        assert L, 'In the linear model L must be specified.'

        order_args['initial_density'] = 'linear'
        order_args['boundary_conditions'] = 'linear'

        DiscreteBook.__init__(self, **order_args)

    # Only allowed operation is order jumps
    def stochastic_timestep(self):
        for orders in [self.ask_orders, self.bid_orders]:
            orders.order_jumps()
            orders.update_best_price()
