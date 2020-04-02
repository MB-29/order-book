import matplotlib.pyplot as plt
import numpy as np

from order_book import OrderBook

# Model parameters
dt = 1e-2
lambd = 1e-1
nu = 1e-2
D = 1

book = OrderBook(dt, D=D, lambd=lambd, nu=nu)

# Price timeseries
n = 200
prices = np.zeros(n)

# Metaorder intensity
m0 = 1
t_start, t_end = n/3, 2*n/3

# Simulation
for t in range(n):

    # Plot evolution
    ymin, ymax = -1, 1
    plt.ylim((ymin, ymax))
    plt.plot(book.X, book.density, label='order density', linewidth=1)
    plt.title(f'iteration {t}')

    # Display best_ask/bid prices
    # plt.vlines(book.best_ask, ymin, ymax, label='best ask', color='blue')
    # plt.vlines(book.best_bid, ymin, ymax, label='best bid', color='yellow')
    plt.vlines(book.price, ymin, ymax, label='price', color='red', linewidth=1)
    plt.legend()
    plt.pause(0.001)

    # Add a metaorder
    book.mt = m0 if (t >= t_start and t <= t_end) else 0
    book.timestep()
    plt.cla()

    prices[t] = book.price
plt.close()

# Plot price evoution
parameters_string = r''
# plt.title(r'Price evolution with $\textrm{dt} = {{{}}}$'.format(dt))
plt.title(r'd$t={{{0}}}$, $\lambda={{{1}}}$, $\nu={{{2}}}$, $D={{{3}}}$'.format(dt, lambd, nu, D))
plt.plot(np.arange(n), prices, label='price evolution')
plt.legend()
plt.show()
