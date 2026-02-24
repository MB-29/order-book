"""
Reproduce the execution animation from demo/execution.gif.

Uses standard_parameters to generate well-behaved simulation parameters.
Now that T is physical time and Nt is number of frames, the metaorder
array has length Nt and represents intensity at each output frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from llob import Simulation, standard_parameters

# Use participation rate ~1 as shown in the original (r=1.01e+00)
participation_rate = 1.0
model_type = "discrete"

# Get standard parameters
params = standard_parameters(participation_rate, model_type)

print(f"Parameters: {params}")

# Create simulation
sim = Simulation.from_params(**params)
print(sim)

# Custom animation with side-by-side layout (1 row, 2 columns) like original
fig, (ax_vol, ax_price) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"$m_0$ = {sim.m0:.2e}, $J$ = {sim.J:.2f}, "
    f"dt = {sim.dt:.2e}, "
    f"$\\Delta p$ = {sim.impact_th:.2f}, "
    f"boundary factor = {sim.boundary_factor:.2f}, "
    f"$r$={sim.participation_rate:.2e}, "
    f"$x \\in [{sim.xmin:.1f}, {sim.xmax:.1f}]$"
)

# Setup volume plot (left)
X = sim.book.X
Nx = len(X)
width = (sim.xmax - sim.xmin) / Nx
y_max = sim.book.y_max

ask_bars = ax_vol.bar(X, sim.book.get_ask_volumes(), align="edge", label="Ask",
                       color="blue", width=width)
bid_bars = ax_vol.bar(X, sim.book.get_bid_volumes(), align="edge", label="Bid",
                       color="red", width=-width)
ax_vol.axvline(0, color="black", lw=0.5, ls="dashed")
best_ask_line, = ax_vol.plot([], [], color="blue", ls="dashed", lw=1, label="best ask")
best_bid_line, = ax_vol.plot([], [], color="red", ls="dashed", lw=1, label="best bid")
ax_vol.set_xlim(sim.xmin, sim.xmax)
ax_vol.set_ylim(0, y_max)
ax_vol.set_title("Order volumes")
ax_vol.legend(loc="upper right")

# Setup price plot (right) - scale y-axis to expected impact
time_interval = sim.time_interval
price_line, = ax_price.plot([], [], label="Price (middle)", color="yellow", lw=2)
ask_price_line, = ax_price.plot([], [], label="Best Ask", color="blue", ls="--")
bid_price_line, = ax_price.plot([], [], label="Best Bid", color="red", ls="--")
ax_price.axhline(0, color="black", lw=0.5, ls="dashed")
ax_price.set_xlim(0, time_interval[-1])
# Scale y-axis to show the expected price range
ax_price.set_ylim(-0.2 * sim.impact_th, 1.2 * sim.impact_th)
ax_price.set_title("Price evolution")
ax_price.legend(loc="upper right")

# Arrays to store frame-level results
Nt = sim.n_frames
prices = np.zeros(Nt)
asks_arr = np.zeros(Nt)
bids_arr = np.zeros(Nt)


def init():
    for b in ask_bars:
        b.set_height(0)
    for b in bid_bars:
        b.set_height(0)
    price_line.set_data([], [])
    ask_price_line.set_data([], [])
    bid_price_line.set_data([], [])
    best_ask_line.set_data([], [])
    best_bid_line.set_data([], [])
    return list(ask_bars) + list(bid_bars) + [price_line, ask_price_line, bid_price_line,
                                               best_ask_line, best_bid_line]


def update(frame):
    if frame % 10 == 0:
        print(f"Frame {frame}/{Nt}")

    # Execute metaorder volume for this frame
    # Volume = intensity * dt (time interval per frame)
    dq = sim.metaorder[frame] * sim.dt

    # Advance the book by one frame (dt time, with n_diff internal steps)
    if sim.model_type == "continuous":
        sim.book.timestep(sim.dt, dq)
    else:
        sim.book.execute_metaorder(dq)
        for _ in range(sim.n_diff):
            sim.book.stochastic_timestep()
            sim.book.order_reaction()
            sim.book.update_price()

    # Record prices at this frame
    asks_arr[frame] = sim.book.best_ask
    bids_arr[frame] = sim.book.best_bid
    prices[frame] = (asks_arr[frame] + bids_arr[frame]) / 2

    # Update volume bars
    ask_vols = sim.book.get_ask_volumes()
    bid_vols = sim.book.get_bid_volumes()
    for bar, h in zip(ask_bars, ask_vols):
        bar.set_height(h)
    for bar, h in zip(bid_bars, bid_vols):
        bar.set_height(h)

    # Update best price lines
    best_ask_line.set_data([sim.book.best_ask, sim.book.best_ask], [0, y_max])
    best_bid_line.set_data([sim.book.best_bid, sim.book.best_bid], [0, y_max])

    # Update price plot
    price_line.set_data(time_interval[:frame+1], prices[:frame+1])
    ask_price_line.set_data(time_interval[:frame+1], asks_arr[:frame+1])
    bid_price_line.set_data(time_interval[:frame+1], bids_arr[:frame+1])

    return list(ask_bars) + list(bid_bars) + [price_line, ask_price_line, bid_price_line,
                                               best_ask_line, best_bid_line]


ani = FuncAnimation(fig, update, init_func=init, frames=Nt, blit=True, repeat=False)
plt.tight_layout()
plt.show()
