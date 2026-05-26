"""
Animate the execution of a metaorder on a discrete LLOB.

Parameters are loaded from a named preset in :mod:`scripts.presets`, so the
same regime that drives a Monte-Carlo experiment can be inspected
interactively. The per-frame metaorder intensity is ``m0 + m1 * fgn(H)``,
matching :class:`llob.MonteCarlo`; pass ``--seed`` for a reproducible run
or ``--no-noise`` for a deterministic one.

Usage::

    PYTHONPATH=./ python scripts/execution_animation.py --preset impact_weak_noise
    PYTHONPATH=./ python scripts/execution_animation.py --preset impact_pure_noise --seed 0
    PYTHONPATH=./ python scripts/execution_animation.py --list
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from fbm import fgn
from matplotlib.animation import FuncAnimation

from llob import LinearDiscreteBook
from scripts.presets import get_preset, list_presets

# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--preset", default="impact_balanced",
    help="Preset name from scripts.presets (default: impact_balanced).",
)
parser.add_argument(
    "--list", action="store_true",
    help="List available presets and exit.",
)
parser.add_argument(
    "--m0", type=float, default=None,
    help="Override the preset's m0 (deterministic metaorder rate).",
)
parser.add_argument(
    "--m1", type=float, default=None,
    help="Override the preset's m1 (noise amplitude).",
)
parser.add_argument(
    "--no-noise", action="store_true",
    help="Force m1 = 0 (deterministic animation).",
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="RNG seed for the fractional Gaussian noise (reproducible run).",
)
args = parser.parse_args()

if args.list:
    for name in list_presets():
        print(f"  {name}")
    sys.exit(0)

preset = get_preset(args.preset)
print(f"Loaded preset: {args.preset}  ({preset.description})")

# ============================================================================
# PARAMETERS (from preset)
# ============================================================================
D = preset.D
L = preset.L
XMIN, XMAX = preset.xmin, preset.xmax
N_GRID = preset.n_grid
M0 = float(args.m0) if args.m0 is not None else preset.m0
M1 = 0.0 if args.no_noise else (float(args.m1) if args.m1 is not None else preset.m1)
HURST = preset.hurst

N_PLOT_FRAMES = preset.n_frames
PLOT_INTERVAL = preset.duration / preset.n_frames
DT_STEP = preset.dt_step

# Theoretical impact / noise scale, used for y-axis bounds
DURATION = N_PLOT_FRAMES * PLOT_INTERVAL
IMPACT_TH = np.sqrt(2 * abs(M0) * DURATION / L) if M0 != 0 else 0.0
# Noise contribution: integrated fGn has Var ~ m1^2 t^{2H}; convert to price
# via diffusion 1/L (rough scale for y-axis bounds, not a theoretical claim).
NOISE_SCALE = abs(M1) * DURATION ** HURST / L if M1 != 0 else 0.0
Y_SCALE = max(IMPACT_TH, NOISE_SCALE, np.sqrt(D * DURATION) * 0.1)

# ============================================================================
# NOISE
# ============================================================================
if args.seed is not None:
    np.random.seed(args.seed)
if M1 != 0:
    noise = M1 * fgn(n=N_PLOT_FRAMES, hurst=HURST, length=DURATION)
else:
    noise = np.zeros(N_PLOT_FRAMES)
metaorder = M0 + noise

# ============================================================================
# BOOK
# ============================================================================
book = LinearDiscreteBook.from_params(
    D=D, L=L, xmin=XMIN, xmax=XMAX, n_grid=N_GRID,
)
y_max = max(
    book.bid_orders.stationary_density(XMIN),
    book.ask_orders.stationary_density(XMAX),
) * book.dx

# History buffers
times = np.zeros(N_PLOT_FRAMES)
asks = np.zeros(N_PLOT_FRAMES)
bids = np.zeros(N_PLOT_FRAMES)
prices = np.zeros(N_PLOT_FRAMES)

# ============================================================================
# FIGURE
# ============================================================================
fig, (ax_vol, ax_price) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"[{args.preset}]  $m_0$ = {M0:.2g}, $m_1$ = {M1:.2g}, $H$ = {HURST:.2g}, "
    f"$J$ = {D * L:.2g}, dt = {PLOT_INTERVAL:.2g}, "
    f"$\\Delta p_\\mathrm{{th}}$ = {IMPACT_TH:.2f}"
)

# --- Volume plot (left) ---
width = (XMAX - XMIN) / N_GRID
ask_bars = ax_vol.bar(
    book.X, book.get_ask_volumes(), align="edge",
    label="Ask", color="blue", width=width,
)
bid_bars = ax_vol.bar(
    book.X, book.get_bid_volumes(), align="edge",
    label="Bid", color="red", width=-width,
)
ax_vol.axvline(0, color="black", lw=0.5, ls="dashed")
(best_ask_line,) = ax_vol.plot(
    [], [], color="blue", ls="dashed", lw=1, label="best ask")
(best_bid_line,) = ax_vol.plot(
    [], [], color="red", ls="dashed", lw=1, label="best bid")
ax_vol.set_xlim(XMIN, XMAX)
ax_vol.set_ylim(0, y_max)
ax_vol.set_title("Order volumes")
ax_vol.legend(loc="upper right")

# --- Price plot (right) ---
(price_line,) = ax_price.plot([], [], label="Price (middle)", color="yellow", lw=2)
(ask_price_line,) = ax_price.plot([], [], label="Best Ask", color="blue", ls="--")
(bid_price_line,) = ax_price.plot([], [], label="Best Bid", color="red", ls="--")
ax_price.axhline(0, color="black", lw=0.5, ls="dashed")
ax_price.set_xlim(0, DURATION)
ax_price.set_ylim(-1.5 * Y_SCALE, 1.5 * Y_SCALE)
ax_price.set_title("Price evolution")
ax_price.legend(loc="upper right")


def init():
    for b in ask_bars:
        b.set_height(0)
    for b in bid_bars:
        b.set_height(0)
    for line in (price_line, ask_price_line, bid_price_line, best_ask_line, best_bid_line):
        line.set_data([], [])
    return [*ask_bars, *bid_bars, price_line, ask_price_line, bid_price_line,
            best_ask_line, best_bid_line]


def update(frame: int):
    if frame % 10 == 0:
        print(f"Frame {frame}/{N_PLOT_FRAMES}")

    dq = metaorder[frame] * PLOT_INTERVAL
    book.evolve(PLOT_INTERVAL, dq, DT_STEP)

    times[frame] = (frame + 1) * PLOT_INTERVAL
    asks[frame] = book.best_ask
    bids[frame] = book.best_bid
    prices[frame] = (book.best_ask + book.best_bid) / 2

    for bar, h in zip(ask_bars, book.get_ask_volumes()):
        bar.set_height(h)
    for bar, h in zip(bid_bars, book.get_bid_volumes()):
        bar.set_height(h)

    best_ask_line.set_data([book.best_ask, book.best_ask], [0, y_max])
    best_bid_line.set_data([book.best_bid, book.best_bid], [0, y_max])

    price_line.set_data(times[: frame + 1], prices[: frame + 1])
    ask_price_line.set_data(times[: frame + 1], asks[: frame + 1])
    bid_price_line.set_data(times[: frame + 1], bids[: frame + 1])

    return [*ask_bars, *bid_bars, price_line, ask_price_line, bid_price_line,
            best_ask_line, best_bid_line]


ani = FuncAnimation(
    fig, update, init_func=init, frames=N_PLOT_FRAMES, blit=True, repeat=False,
)
plt.tight_layout()
plt.show()
