from discrete_book import DiscreteBook

class LinearDiscreteBook(DiscreteBook):

    def __init__(self, **order_args):

        order_args['nu'] = 0
        order_args['lambd'] = 0
        L = order_args.get('L')
        assert L, 'In the linear model L must be specified.'

        order_args['initial_density'] = 'linear'
        order_args['boundary_conditions'] = 'linear'

        DiscreteBook.__init__(self, **order_args)

    # Only allowed timestep is order jumps
    def stochastic_timestep(self):
        for orders in [self.ask_orders, self.bid_orders]:
            orders.order_jumps()        