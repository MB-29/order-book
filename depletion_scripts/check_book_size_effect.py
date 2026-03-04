"""
Check if the saturation in the phase diagram is due to finite book size.

Hypothesis: The critical m1 saturates because the spread can't exceed the book size.
Test: Run with different book sizes and see if the critical boundary changes.
"""

import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from llob import Simulation


def run_simulation(args: tuple) -> dict:
    """Run a single simulation."""
    alpha, m1, book_size, n_grid, seed = args

    xmin, xmax = -book_size / 2, book_size / 2
    D = 0.5
    L = 10.0
    nu = 0.1
    lambd = L * np.sqrt(nu * D)

    dx = (xmax - xmin) / (n_grid - 1)
    n_frames = 100
    duration = 500.0

    np.random.seed(seed)
    metaorder = m1 * np.random.randn(n_frames)

    sim = Simulation.from_params(
        model_type="discrete",
        duration=duration,
        n_frames=n_frames,
        n_grid=n_grid,
        xmin=xmin,
        xmax=xmax,
        D=D,
        L=L,
        nu=nu,
        lambd=lambd,
        metaorder=metaorder,
        alpha=alpha,
        seed=seed,
    )

    try:
        sim.run()
        half = n_frames // 2
        return {
            "alpha": alpha,
            "m1": m1,
            "book_size": book_size,
            "success": True,
            "mean_spread": np.mean(sim.spreads[half:]),
            "max_spread": np.max(sim.spreads),
            "max_spread_fraction": np.max(sim.spreads) / book_size,
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {
                "alpha": alpha,
                "m1": m1,
                "book_size": book_size,
                "success": False,
                "mean_spread": np.inf,
                "max_spread": np.inf,
                "max_spread_fraction": 1.0,
            }
        raise


def main():
    print("=" * 70)
    print("Checking book size effect on phase diagram")
    print("=" * 70)

    # Test different book sizes
    book_sizes = [100, 200, 400]  # Current is 100
    alpha_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    m1_values = np.array([20, 50, 100, 150, 200, 300, 400, 500])
    seed = 42

    # Keep n_grid proportional to book size to maintain same dx
    base_n_grid = 100
    base_book_size = 100

    results_by_size = {}

    for book_size in book_sizes:
        n_grid = int(base_n_grid * book_size / base_book_size)
        dx = book_size / (n_grid - 1)
        print(f"\nBook size = {book_size}, n_grid = {n_grid}, dx = {dx:.3f}")

        # Create tasks
        tasks = []
        for alpha in alpha_values:
            for m1 in m1_values:
                tasks.append((alpha, m1, book_size, n_grid, seed))

        # Run in parallel
        n_workers = min(mp.cpu_count(), 8)
        with mp.Pool(n_workers) as pool:
            results = pool.map(run_simulation, tasks)

        # Organize results
        stability = np.zeros((len(alpha_values), len(m1_values)))
        max_spread_frac = np.zeros((len(alpha_values), len(m1_values)))

        for result in results:
            i = alpha_values.index(result["alpha"])
            j = np.where(m1_values == result["m1"])[0][0]
            stability[i, j] = 1 if result["success"] else 0
            max_spread_frac[i, j] = result["max_spread_fraction"]

        results_by_size[book_size] = {
            "stability": stability,
            "max_spread_frac": max_spread_frac,
        }

        # Print critical m1 for each alpha
        print(f"  Critical m1 for each alpha (book_size={book_size}):")
        for i, alpha in enumerate(alpha_values):
            stable_m1s = m1_values[stability[i, :] == 1]
            m1_c = np.max(stable_m1s) if len(stable_m1s) > 0 else 0
            print(f"    alpha={alpha}: m1_c = {m1_c}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, book_size in enumerate(book_sizes):
        ax = axes[idx]
        stability = results_by_size[book_size]["stability"]

        # Find critical m1 for each alpha
        critical_m1 = []
        for i, alpha in enumerate(alpha_values):
            stable_m1s = m1_values[stability[i, :] == 1]
            critical_m1.append(np.max(stable_m1s) if len(stable_m1s) > 0 else 0)

        ax.plot(alpha_values, critical_m1, "bo-", markersize=8, linewidth=2)
        ax.fill_between(alpha_values, 0, critical_m1, alpha=0.3, color="green")
        ax.fill_between(
            alpha_values, critical_m1, max(m1_values), alpha=0.3, color="red"
        )
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Critical m1")
        ax.set_title(f"Book size = {book_size}")
        ax.set_ylim(0, max(m1_values) * 1.1)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Critical boundary for different book sizes", fontsize=14)
    plt.tight_layout()
    plt.savefig("depletion_scripts/book_size_effect.png", dpi=150)
    print("\nSaved to depletion_scripts/book_size_effect.png")

    # Check max spread fraction at depletion
    print("\n" + "=" * 70)
    print("Max spread as fraction of book size at high m1")
    print("=" * 70)

    for book_size in book_sizes:
        max_frac = results_by_size[book_size]["max_spread_frac"]
        print(f"\nBook size = {book_size}:")
        for i, alpha in enumerate(alpha_values):
            # Look at highest m1 that succeeded
            stable_mask = results_by_size[book_size]["stability"][i, :] == 1
            if np.any(stable_mask):
                last_stable_idx = np.where(stable_mask)[0][-1]
                frac = max_frac[i, last_stable_idx]
                m1 = m1_values[last_stable_idx]
                print(f"  alpha={alpha}: at m1={m1}, max_spread = {frac*100:.1f}% of book")


if __name__ == "__main__":
    main()
