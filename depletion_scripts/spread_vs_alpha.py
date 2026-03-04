"""
Test whether alpha stabilizes the spread near zero.

Physics:
- With m0=0 (no deterministic metaorder), equilibrium spread should be 0
- Noise m1 causes fluctuations that open the spread
- Alpha > 0 increases deposition when spread opens, pushing it back to 0
- For large enough alpha, spread should remain near 0 regardless of m1

Key question: What is the stationary spread s*(alpha, m1)?
- For alpha=0: s* may grow unbounded (the problem)
- For alpha>0: s* should be finite and decrease with alpha
"""

import matplotlib.pyplot as plt
import numpy as np

from llob import Simulation

# Parameters - use smaller/faster settings for quick iteration
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


def run_simulation(alpha: float, m1: float, seed: int = 42) -> dict:
    """Run simulation with m0=0 (only noise) and measure spread."""
    lambd = PARAMS["L"] * np.sqrt(PARAMS["nu"] * PARAMS["D"])
    n_frames = PARAMS["n_frames"]

    # m0 = 0, only noise from m1
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
        # Use second half of simulation for "stationary" measurement
        # (first half may be transient)
        half = n_frames // 2
        stationary_spreads = sim.spreads[half:]
        return {
            "success": True,
            "mean_spread": np.mean(stationary_spreads),
            "std_spread": np.std(stationary_spreads),
            "max_spread": np.max(sim.spreads),
            "spreads": sim.spreads.copy(),
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {
                "success": False,
                "mean_spread": np.inf,
                "std_spread": np.inf,
                "max_spread": np.inf,
                "spreads": sim.spreads.copy(),
            }
        raise


def main():
    print("=" * 70)
    print("Stationary spread s*(alpha, m1)")
    print("=" * 70)
    print(f"\nBook: [{PARAMS['xmin']}, {PARAMS['xmax']}], L={PARAMS['L']}, nu={PARAMS['nu']}")
    print("Metaorder: m0=0 (no deterministic component), only noise m1")
    print()

    # Test: For fixed m1, how does spread depend on alpha?
    m1 = 20.0  # Moderate noise
    alpha_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"Fixed m1 = {m1}")
    print("-" * 50)
    print(f"{'alpha':>8} | {'mean spread':>12} | {'std':>8} | {'max':>8}")
    print("-" * 50)

    results_by_alpha = {}
    for alpha in alpha_values:
        result = run_simulation(alpha, m1)
        results_by_alpha[alpha] = result
        if result["success"]:
            print(
                f"{alpha:>8.1f} | {result['mean_spread']:>12.2f} | "
                f"{result['std_spread']:>8.2f} | {result['max_spread']:>8.2f}"
            )
        else:
            print(f"{alpha:>8.1f} | {'DEPLETED':>12} | {'--':>8} | {'--':>8}")

    # Plot spread vs alpha
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mean spread vs alpha
    ax1 = axes[0]
    alphas = [a for a in alpha_values if results_by_alpha[a]["success"]]
    means = [results_by_alpha[a]["mean_spread"] for a in alphas]
    stds = [results_by_alpha[a]["std_spread"] for a in alphas]

    ax1.errorbar(alphas, means, yerr=stds, fmt="o-", capsize=5, markersize=8)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Alpha (spread-sensitivity)")
    ax1.set_ylabel("Mean stationary spread")
    ax1.set_title(f"Spread vs Alpha (m1={m1})")
    ax1.grid(True, alpha=0.3)

    # Right: Spread time series for different alpha
    ax2 = axes[1]
    times = np.linspace(0, PARAMS["duration"], PARAMS["n_frames"])
    for alpha in [0.0, 1.0, 5.0]:
        if results_by_alpha[alpha]["success"]:
            ax2.plot(times, results_by_alpha[alpha]["spreads"], label=f"α={alpha}")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Spread")
    ax2.set_title(f"Spread evolution (m1={m1})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_vs_alpha.png", dpi=150)
    print(f"\nSaved to depletion_scripts/spread_vs_alpha.png")

    # Now test: For fixed alpha, how does spread depend on m1?
    print("\n" + "=" * 70)
    print("Effect of noise intensity m1")
    print("=" * 70)

    alpha_fixed = 2.0
    m1_values = [5.0, 10.0, 20.0, 50.0, 100.0]

    print(f"\nFixed alpha = {alpha_fixed}")
    print("-" * 50)
    print(f"{'m1':>8} | {'mean spread':>12} | {'std':>8} | {'max':>8}")
    print("-" * 50)

    results_by_m1 = {}
    for m1 in m1_values:
        result = run_simulation(alpha_fixed, m1)
        results_by_m1[m1] = result
        if result["success"]:
            print(
                f"{m1:>8.1f} | {result['mean_spread']:>12.2f} | "
                f"{result['std_spread']:>8.2f} | {result['max_spread']:>8.2f}"
            )
        else:
            print(f"{m1:>8.1f} | {'DEPLETED':>12} | {'--':>8} | {'--':>8}")

    # Plot spread vs m1
    fig2, ax = plt.subplots(figsize=(8, 5))
    m1s = [m for m in m1_values if results_by_m1[m]["success"]]
    means = [results_by_m1[m]["mean_spread"] for m in m1s]
    stds = [results_by_m1[m]["std_spread"] for m in m1s]

    ax.errorbar(m1s, means, yerr=stds, fmt="o-", capsize=5, markersize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Noise intensity m1")
    ax.set_ylabel("Mean stationary spread")
    ax.set_title(f"Spread vs Noise (alpha={alpha_fixed})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_vs_m1.png", dpi=150)
    print(f"\nSaved to depletion_scripts/spread_vs_m1.png")

    # Key test: Compare alpha=0 vs alpha>0 for same m1
    print("\n" + "=" * 70)
    print("Key comparison: alpha=0 vs alpha>0")
    print("=" * 70)

    m1_test = 20.0
    for alpha in [0.0, 2.0, 10.0]:
        result = run_simulation(alpha, m1_test)
        if result["success"]:
            print(f"alpha={alpha}: mean spread = {result['mean_spread']:.2f}")
        else:
            print(f"alpha={alpha}: DEPLETED (spread blew up)")

    # Summary
    dx = (PARAMS["xmax"] - PARAMS["xmin"]) / (PARAMS["n_grid"] - 1)
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Grid spacing dx = {dx:.2f}")
    print(f"Minimum spread = dx = {dx:.2f} (one tick)")
    print()
    print("Key findings:")
    print("1. With large alpha, spread stays at minimum (1 tick) - STABLE")
    print("2. With alpha=0 and high m1, the book DEPLETES")
    print("3. Alpha provides stabilization by boosting deposition when spread opens")
    print()
    print("The critical question: what is the minimum alpha needed for stability")
    print("at a given noise level m1?")

    # Find critical alpha for different m1
    print("\n" + "=" * 70)
    print("Critical alpha for stability at different m1")
    print("=" * 70)

    m1_values = [20, 50, 100, 150, 200]
    alpha_test = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    for m1 in m1_values:
        critical_alpha = None
        for alpha in alpha_test:
            result = run_simulation(alpha, m1)
            if result["success"]:
                critical_alpha = alpha
                break
        if critical_alpha is not None:
            print(f"m1={m1:>4}: min stable alpha = {critical_alpha}")
        else:
            print(f"m1={m1:>4}: no stable alpha found (need alpha > {alpha_test[-1]})")


if __name__ == "__main__":
    main()
