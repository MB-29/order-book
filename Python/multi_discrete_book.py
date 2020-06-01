import numpy as np
from functools import reduce 

from limit_orders import LimitOrders
from discrete_book import DiscreteBook
from linear_discrete_book import LinearDiscreteBook

class MultiDiscreteBook(DiscreteBook) :

    def __init__(self, **multi_book_args):
        L_list = multi_book_args['L']
        lambd_list = multi_book_args['lambd']
        nu_list = multi_book_args['nu']
        self.N_actors = len(L_list)
        self.actor_books = []

        # Get spatial values
        self.xmin = multi_book_args['xmin']
        self.xmax = multi_book_args['xmax']
        self.n_diff = multi_book_args['n_diff']
        self.Nx = multi_book_args['Nx']
        self.X, self.dx = np.linspace(
            self.xmin, self.xmax, num=self.Nx, retstep=True)
        self.L = multi_book_args['L']

        for n in range(self.N_actors):
            L = L_list[n]
            lambd = lambd_list[n]
            nu = nu_list[n]
            linear = nu == 0
            model = LinearDiscreteBook if linear else DiscreteBook
            multi_book_args['L'] = L
            multi_book_args['lambd'] = lambd
            multi_book_args['nu'] = nu
            self.actor_books.append(model(**multi_book_args))

        self.dq = 0

        self.best_ask = np.mean([book.best_ask for book in self.actor_books])
        self.best_bid = np.mean([book.best_bid for book in self.actor_books])


        self.y_max = 1.5 * \
            max(abs(self.xmin), abs(self.xmax)) * np.max(self.L) * self.dx

    
    def get_ask_volumes(self):
        return reduce(np.add, [book.get_ask_volumes() for book in self.actor_books])
    def get_bid_volumes(self):
        return reduce(np.add, [book.get_bid_volumes() for book in self.actor_books])
    
    def timestep(self) :
        self.execute_metaorder(self.dq)
        for book in self.actor_books:
            book.evolve()
        self.best_ask = np.mean([book.best_ask for book in self.actor_books])
        self.best_bid = np.mean([book.best_bid for book in self.actor_books])
            
    
    def execute_metaorder(self, trade_volume):
        executed_volume = 0
        increment = 1
        while increment > 0 :
            increment = 0
            for book in self.actor_books:
                volume = book.best_ask_volume if trade_volume > 0 else -book.best_bid_volume
                if abs(volume) + executed_volume > abs(trade_volume) :
                    continue
                book.execute_metaorder(volume)
                executed_volume += abs(volume)
                increment = abs(volume)
    

