"""
Full phase diagram: Map the stability boundary in (m1, alpha) space.

For each (m1, alpha) pair, run a simulation and check if it stays stable.
Uses multiprocessing for parallelization.
"""

import multiprocessing as mp
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from llob import Simulation

# Parameters - can be overridden via command line
PARAMS = {
    "model_type": "discrete",
    "duration": 500.0,
    "n_frames": 100,
    "n_grid": 100,
    "xmin": -50.0,
    "xmax": 50.0,
    "D": 0.5,
    "L": 10.0,
    "nu": 0.1,
}

def set_book_size(book_size: float, n_grid: int = 100):
    """Set book size while keeping other params fixed."""
    PARAMS["xmin"] = -book_size / 2
    PARAMS["xmax"] = book_size / 2
    PARAMS["n_grid"] = n_grid


def run_single_simulation(args: tuple) -> dict:
    """Run a single simulation for given (alpha, m1) pair."""
    alpha, m1, seed = args

    lambd = PARAMS["L"] * np.sqrt(PARAMS["nu"] * PARAMS["D"])
    n_frames = PARAMS["n_frames"]

    np.random.seed(seed)
    metaorder = m1 * np.random.randn(n_frames)

    sim = Simulation.from_params(
        **PARAMS,
        lambd=lambd,
        metaorder=metaorder,
        alpha=alpha,
        seed=seed,
    )

    try:
        sim.run()
        half = n_frames // 2
        stationary_spreads = sim.spreads[half:]
        return {
            "alpha": alpha,
            "m1": m1,
            "success": True,
            "mean_spread": np.mean(stationary_spreads),
            "max_spread": np.max(sim.spreads),
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {
                "alpha": alpha,
                "m1": m1,
                "success": False,
                "mean_spread": np.inf,
                "max_spread": np.inf,
            }
        raise


def main():
    print("=" * 70)
    print("Full Phase Diagram: Stability in (m1, alpha) space")
    print("=" * 70)

    # Fine grid for phase diagram
    m1_values = np.array([5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500])
    alpha_values = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])

    # Create all (alpha, m1) pairs
    tasks = []
    seed = 42
    for alpha in alpha_values:
        for m1 in m1_values:
            tasks.append((alpha, m1, seed))

    print(f"Running {len(tasks)} simulations...")
    print(f"m1 values: {m1_values}")
    print(f"alpha values: {alpha_values}")

    # Run in parallel
    n_workers = min(mp.cpu_count(), 8)
    print(f"Using {n_workers} workers")

    with mp.Pool(n_workers) as pool:
        results = pool.map(run_single_simulation, tasks)

    # Organize results into a grid
    stability = np.zeros((len(alpha_values), len(m1_values)))
    mean_spreads = np.zeros((len(alpha_values), len(m1_values)))

    for result in results:
        i = np.where(alpha_values == result["alpha"])[0][0]
        j = np.where(m1_values == result["m1"])[0][0]
        stability[i, j] = 1 if result["success"] else 0
        mean_spreads[i, j] = result["mean_spread"]

    # Print summary table
    print("\n" + "=" * 70)
    print("Results: 1=Stable, 0=Depleted")
    print("=" * 70)
    header = "alpha\\m1 |" + "".join(f"{m:>5}" for m in m1_values)
    print(header)
    print("-" * len(header))
    for i, alpha in enumerate(alpha_values):
        row = f"{alpha:>7.1f} |" + "".join(
            f"{'  ✓' if stability[i,j] else '  ✗':>5}" for j in range(len(m1_values))
        )
        print(row)

    # Find critical boundary
    critical_m1 = []
    for i, alpha in enumerate(alpha_values):
        stable_m1s = m1_values[stability[i, :] == 1]
        if len(stable_m1s) > 0:
            critical_m1.append(np.max(stable_m1s))
        else:
            critical_m1.append(0)

    print("\n" + "=" * 70)
    print("Critical m1 for each alpha:")
    print("=" * 70)
    for alpha, m1_c in zip(alpha_values, critical_m1):
        print(f"alpha={alpha:>5.1f}: m1_c = {m1_c:>5.0f}")

    # Create figure with multiple plots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Phase diagram heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(
        stability,
        extent=[m1_values[0], m1_values[-1], alpha_values[0], alpha_values[-1]],
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax1.set_xlabel("Noise intensity m1", fontsize=12)
    ax1.set_ylabel("Alpha (spread-sensitivity)", fontsize=12)
    ax1.set_title("Phase Diagram: Green=Stable, Red=Depleted", fontsize=14)

    # Add markers
    for i, alpha in enumerate(alpha_values):
        for j, m1 in enumerate(m1_values):
            marker = "o" if stability[i, j] == 1 else "x"
            color = "darkgreen" if stability[i, j] == 1 else "darkred"
            ax1.plot(m1, alpha, marker, color=color, markersize=5, alpha=0.7)

    # Plot 2: Critical boundary curve
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(alpha_values, critical_m1, "bo-", markersize=8, linewidth=2, label="Critical m1")
    ax2.fill_between(
        alpha_values, 0, critical_m1, alpha=0.3, color="green", label="Stable region"
    )
    ax2.fill_between(
        alpha_values,
        critical_m1,
        max(m1_values) * 1.1,
        alpha=0.3,
        color="red",
        label="Depleted region",
    )
    ax2.set_xlabel("Alpha (spread-sensitivity)", fontsize=12)
    ax2.set_ylabel("Critical noise intensity m1_c", fontsize=12)
    ax2.set_title("Critical Boundary: m1_c(alpha)", fontsize=14)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(alpha_values))
    ax2.set_ylim(0, max(m1_values) * 1.1)

    # Plot 3: alpha_c(m1) - inverse view
    ax3 = fig.add_subplot(2, 2, 3)
    # Find critical alpha for each m1
    critical_alpha = []
    for j, m1 in enumerate(m1_values):
        stable_alphas = alpha_values[stability[:, j] == 1]
        if len(stable_alphas) > 0:
            critical_alpha.append(np.min(stable_alphas))
        else:
            critical_alpha.append(np.inf)

    # Only plot finite values
    valid = np.array(critical_alpha) < np.inf
    ax3.plot(
        m1_values[valid],
        np.array(critical_alpha)[valid],
        "ro-",
        markersize=8,
        linewidth=2,
        label="Critical alpha",
    )
    ax3.fill_between(
        m1_values[valid],
        np.array(critical_alpha)[valid],
        max(alpha_values) * 1.1,
        alpha=0.3,
        color="green",
        label="Stable region",
    )
    ax3.fill_between(
        m1_values[valid],
        0,
        np.array(critical_alpha)[valid],
        alpha=0.3,
        color="red",
        label="Depleted region",
    )
    ax3.set_xlabel("Noise intensity m1", fontsize=12)
    ax3.set_ylabel("Critical alpha_c", fontsize=12)
    ax3.set_title("Critical Boundary: alpha_c(m1)", fontsize=14)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mean spread heatmap (for stable cases)
    ax4 = fig.add_subplot(2, 2, 4)
    # Replace inf with NaN for plotting
    plot_spreads = mean_spreads.copy()
    plot_spreads[~np.isfinite(plot_spreads)] = np.nan
    im4 = ax4.imshow(
        plot_spreads,
        extent=[m1_values[0], m1_values[-1], alpha_values[0], alpha_values[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax4.set_xlabel("Noise intensity m1", fontsize=12)
    ax4.set_ylabel("Alpha", fontsize=12)
    ax4.set_title("Mean Spread (stable cases only)", fontsize=14)
    plt.colorbar(im4, ax=ax4, label="Mean spread")

    plt.tight_layout()

    # Generate filename based on book size
    book_size = PARAMS["xmax"] - PARAMS["xmin"]
    filename = f"depletion_scripts/full_phase_diagram_booksize{int(book_size)}.png"
    plt.savefig(filename, dpi=150)
    print(f"\nSaved to {filename}")

    # Fit a linear relationship for critical boundary
    valid_alpha = np.array(alpha_values)[np.array(critical_m1) > 0]
    valid_m1c = np.array(critical_m1)[np.array(critical_m1) > 0]
    if len(valid_alpha) > 2:
        coeffs = np.polyfit(valid_alpha, valid_m1c, 1)
        print(f"\nLinear fit: m1_c ≈ {coeffs[0]:.1f} * alpha + {coeffs[1]:.1f}")
        print(f"=> alpha_c ≈ (m1 - {coeffs[1]:.1f}) / {coeffs[0]:.1f}")


if __name__ == "__main__":
    import sys

    # Allow setting book size from command line
    if len(sys.argv) > 1:
        book_size = float(sys.argv[1])
        n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        set_book_size(book_size, n_grid)
        print(f"Using book size = {book_size}, n_grid = {n_grid}")

    main()
