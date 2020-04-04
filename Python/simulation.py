import matplotlib.pyplot as plt
import numpy as np

from order_book import OrderBook

# Model parameters
T = 1
Nt = 20
dt = T/float(Nt)
lambd = 1
nu = 1e-2
D = 1
J = lambd * np.sqrt(D/float(nu))
book = OrderBook(dt, D=D, lambd=lambd, nu=nu, resolution=100)

# Price timeseries
prices = np.zeros(Nt)
time_interval = np.linspace(0, T, num=Nt)

# Metaorder intensity
m0 = 5
n_start, n_end = Nt//5, 2*Nt//3
A_low = m0/(J*np.sqrt(np.pi))
A_high = np.sqrt(2*m0/J)

# Simulation
for n in range(Nt):

    # Plot evolution
    ymin, ymax = -1, 1
    plt.ylim((ymin, ymax))
    plt.plot(book.X, book.density, label='order density', linewidth=1)
    plt.title(f'iteration {n}')

    # Display best_ask/bid prices
    plt.vlines(book.price, ymin, ymax, label='price', color='red', linewidth=1)
    plt.hlines(0, -1, 1, color='black', linewidth=0.5, linestyle='dashed')
    plt.vlines(book.best_ask, ymin, ymax, label='best ask', color='blue')
    plt.vlines(book.best_bid, ymin, ymax, label='best bid', color='yellow')
    plt.vlines(0, 0, m0*dt/book.dx, label='traded quantity', color='cyan')

    plt.legend()
    plt.pause(0.001)

    # Add a metaorder
    book.mt = m0 if (n >= n_start and n <= n_end) else 0
    book.timestep()
    plt.cla()

    prices[n] = book.price

plt.close()

p_th = A_low*np.sqrt(D*time_interval[0:n_end-n_start])
plt.plot(time_interval[n_start:n_end],
         p_th, label='Theoretical impact', lw=1, color='green')


# Plot price evoution
# plt.title(r'Price evolution with $\textrm{dt} = {{{}}}$'.format(dt))
# plt.title(r'd$t={{{0}}}$, $\lambda={{{1}}}$, $\nu={{{2}}}$, $D={{{3}}}$'.format(
#     dt, lambd, nu, D))
plt.title(r'$m_0={{{0}}}$, $J={{{1}}}$'.format(m0, round(J, 2)))
plt.plot(time_interval, prices, label='price evolution')
plt.legend()
plt.show()
