"""
Test spread evolution over time to see if depletion occurs.

We need to test with:
1. Higher noise intensity (m1 >> typical deposition rate)
2. Longer simulation time
3. Track spread over time to see growth pattern
"""

import matplotlib.pyplot as plt
import numpy as np

from llob import Simulation

# Parameters matching Experiment 1 from new_experiments.md
PARAMS = {
    "model_type": "discrete",
    "duration": 2000.0,  # Long enough to see growth
    "n_frames": 200,
    "n_grid": 200,
    "xmin": -100.0,
    "xmax": 100.0,
    "D": 0.5,
    "L": 10.0,
    "nu": 0.1,
}


def run_spread_evolution(alpha: float, m1: float, seed: int = 42) -> dict:
    """Run simulation and return time series of spread."""
    lambd = PARAMS["L"] * np.sqrt(PARAMS["nu"] * PARAMS["D"])
    n_frames = PARAMS["n_frames"]
    duration = PARAMS["duration"]

    # Create metaorder array - this represents total volume per frame
    # Use larger noise to stress the system
    np.random.seed(seed)
    metaorder = m1 * np.random.randn(n_frames)

    sim = Simulation.from_params(
        **PARAMS,
        lambd=lambd,
        metaorder=metaorder,
        alpha=alpha,
        seed=seed,
    )

    # Time array
    times = np.linspace(0, duration, n_frames)

    try:
        sim.run()
        return {
            "times": times,
            "spreads": sim.spreads.copy(),
            "prices": sim.prices.copy(),
            "asks": sim.asks.copy(),
            "bids": sim.bids.copy(),
            "depleted": False,
            "depletion_time": None,
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            # Find where simulation stopped
            # spreads array will have zeros where not filled
            last_nonzero = np.where(sim.spreads > 0)[0]
            if len(last_nonzero) > 0:
                depletion_idx = last_nonzero[-1]
            else:
                depletion_idx = 0

            return {
                "times": times,
                "spreads": sim.spreads.copy(),
                "prices": sim.prices.copy(),
                "asks": sim.asks.copy(),
                "bids": sim.bids.copy(),
                "depleted": True,
                "depletion_time": times[depletion_idx] if depletion_idx > 0 else 0,
            }
        raise


def main():
    print("=" * 70)
    print("Testing spread evolution over time")
    print("=" * 70)

    # Test with high noise
    m1_values = [10.0, 50.0, 100.0]
    alpha_values = [0.0, 0.5, 2.0]

    fig, axes = plt.subplots(len(m1_values), 1, figsize=(12, 10), sharex=True)

    for i, m1 in enumerate(m1_values):
        ax = axes[i]
        ax.set_title(f"Noise intensity m1 = {m1}")

        for alpha in alpha_values:
            print(f"\nRunning m1={m1}, alpha={alpha}...")
            result = run_spread_evolution(alpha, m1)

            label = f"α={alpha}"
            # Only plot non-zero spreads (zeros mean not yet computed or depleted)
            valid = result["spreads"] > 0
            if np.any(valid):
                ax.plot(result["times"][valid], result["spreads"][valid], label=label)

            # Stats
            valid_spreads = result["spreads"][result["spreads"] > 0]
            if len(valid_spreads) > 0:
                mean_s = np.mean(valid_spreads)
                max_s = np.max(valid_spreads)
                final_s = valid_spreads[-1]
                print(f"  Mean spread: {mean_s:.2f}, Max: {max_s:.2f}, Final: {final_s:.2f}")
            else:
                print(f"  No valid spread data")

            if result["depleted"]:
                print(f"  DEPLETED at t={result['depletion_time']:.1f}!")

        ax.set_ylabel("Spread")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_evolution.png", dpi=150)
    print("\nSaved plot to depletion_scripts/spread_evolution.png")

    # Also check if the book is actually being depleted
    print("\n" + "=" * 70)
    print("Checking book depletion at end of simulation")
    print("=" * 70)

    for m1 in [50.0, 100.0]:
        for alpha in [0.0, 1.0, 5.0]:
            print(f"\nm1={m1}, alpha={alpha}:")
            result = run_spread_evolution(alpha, m1)

            if result["depleted"]:
                print(f"  DEPLETED at t={result['depletion_time']:.1f}!")
                continue

            # Check if ask > bid at end
            final_ask = result["asks"][-1]
            final_bid = result["bids"][-1]
            print(f"  Final ask: {final_ask:.2f}")
            print(f"  Final bid: {final_bid:.2f}")
            print(f"  Final spread: {final_ask - final_bid:.2f}")

            # Check if spread hit boundaries
            max_spread = PARAMS["xmax"] - PARAMS["xmin"]
            valid_spreads = result["spreads"][result["spreads"] > 0]
            if len(valid_spreads) > 0 and np.max(valid_spreads) > 0.5 * max_spread:
                print(
                    f"  WARNING: Spread reached "
                    f"{100*np.max(valid_spreads)/max_spread:.0f}% of max!"
                )


def find_critical_m1():
    """Find the critical m1 that causes depletion for each alpha value."""
    print("\n" + "=" * 70)
    print("Finding critical m1 for different alpha values")
    print("=" * 70)

    alpha_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    m1_test_values = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

    # Store results: alpha -> (max_stable_m1, min_depleted_m1)
    critical_m1 = {}

    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}...")
        max_stable = 0
        min_depleted = float("inf")

        for m1 in m1_test_values:
            result = run_spread_evolution(alpha, m1)
            if result["depleted"]:
                print(f"  m1={m1}: DEPLETED at t={result['depletion_time']:.0f}")
                min_depleted = min(min_depleted, m1)
            else:
                valid_spreads = result["spreads"][result["spreads"] > 0]
                mean_s = np.mean(valid_spreads) if len(valid_spreads) > 0 else 0
                print(f"  m1={m1}: Stable, mean spread = {mean_s:.2f}")
                max_stable = max(max_stable, m1)

        critical_m1[alpha] = (max_stable, min_depleted)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    alphas = list(critical_m1.keys())
    max_stables = [critical_m1[a][0] for a in alphas]
    min_depleteds = [critical_m1[a][1] if critical_m1[a][1] != float("inf") else 250 for a in alphas]

    # Plot the stability boundary
    ax.fill_between(alphas, 0, max_stables, alpha=0.3, color="green", label="Stable region")
    ax.fill_between(
        alphas, max_stables, min_depleteds, alpha=0.3, color="yellow", label="Transition"
    )
    ax.fill_between(alphas, min_depleteds, 250, alpha=0.3, color="red", label="Depleted region")

    ax.plot(alphas, max_stables, "go-", label="Max stable m1", markersize=8)
    ax.plot(
        alphas,
        [m if m != float("inf") else 250 for m in [critical_m1[a][1] for a in alphas]],
        "ro-",
        label="Min depleted m1",
        markersize=8,
    )

    ax.set_xlabel("Alpha (spread-sensitivity)")
    ax.set_ylabel("Noise intensity m1")
    ax.set_title("Critical m1 vs Alpha: Where does depletion occur?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(alphas) + 0.5)
    ax.set_ylim(0, 250)

    plt.tight_layout()
    plt.savefig("depletion_scripts/critical_m1_vs_alpha.png", dpi=150)
    print("\nSaved plot to depletion_scripts/critical_m1_vs_alpha.png")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Critical m1 by alpha")
    print("=" * 70)
    print(f"{'Alpha':>8} | {'Max Stable m1':>15} | {'Min Depleted m1':>15}")
    print("-" * 45)
    for alpha in alphas:
        max_s, min_d = critical_m1[alpha]
        min_d_str = f"{min_d}" if min_d != float("inf") else ">200"
        print(f"{alpha:>8.1f} | {max_s:>15} | {min_d_str:>15}")


if __name__ == "__main__":
    main()
    find_critical_m1()
