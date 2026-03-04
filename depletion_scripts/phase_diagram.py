"""
Phase diagram: For each (m1, alpha) pair, run a simulation and check if it depletes.

This maps out the stability boundary in (m1, alpha) space.
"""

import matplotlib.pyplot as plt
import numpy as np

from llob import Simulation

# Parameters
PARAMS = {
    "model_type": "discrete",
    "duration": 1000.0,
    "n_frames": 100,
    "n_grid": 200,
    "xmin": -100.0,
    "xmax": 100.0,
    "D": 0.5,
    "L": 10.0,
    "nu": 0.1,
}


def run_simulation(alpha: float, m1: float, seed: int = 42) -> dict:
    """Run simulation and return whether it depleted."""
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
        valid_spreads = sim.spreads[sim.spreads > 0]
        return {
            "depleted": False,
            "mean_spread": np.mean(valid_spreads) if len(valid_spreads) > 0 else 0,
            "max_spread": np.max(valid_spreads) if len(valid_spreads) > 0 else 0,
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {"depleted": True, "mean_spread": np.inf, "max_spread": np.inf}
        raise


def main():
    print("=" * 70)
    print("Phase diagram: Stability boundary in (m1, alpha) space")
    print("=" * 70)

    # Grid of (m1, alpha) values to test
    m1_values = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150])
    alpha_values = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])

    # Results matrix: 1 = stable, 0 = depleted
    stability = np.zeros((len(alpha_values), len(m1_values)))
    mean_spreads = np.zeros((len(alpha_values), len(m1_values)))

    for i, alpha in enumerate(alpha_values):
        for j, m1 in enumerate(m1_values):
            print(f"Testing alpha={alpha:.1f}, m1={m1:.0f}...", end=" ")
            result = run_simulation(alpha, m1)
            stability[i, j] = 0 if result["depleted"] else 1
            mean_spreads[i, j] = result["mean_spread"]
            status = "DEPLETED" if result["depleted"] else f"stable (s={result['mean_spread']:.1f})"
            print(status)

    # Plot phase diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Binary stability map
    ax1 = axes[0]
    im1 = ax1.imshow(
        stability,
        extent=[m1_values[0], m1_values[-1], alpha_values[0], alpha_values[-1]],
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax1.set_xlabel("Noise intensity m1")
    ax1.set_ylabel("Alpha (spread-sensitivity)")
    ax1.set_title("Stability: Green=Stable, Red=Depleted")

    # Add markers for each test point
    for i, alpha in enumerate(alpha_values):
        for j, m1 in enumerate(m1_values):
            marker = "o" if stability[i, j] == 1 else "x"
            color = "green" if stability[i, j] == 1 else "red"
            ax1.plot(m1, alpha, marker, color=color, markersize=8)

    # Right: Mean spread heatmap (log scale for visibility)
    ax2 = axes[1]
    # Replace inf with max finite value for plotting
    plot_spreads = mean_spreads.copy()
    finite_mask = np.isfinite(plot_spreads)
    if np.any(finite_mask):
        max_finite = np.max(plot_spreads[finite_mask])
        plot_spreads[~finite_mask] = max_finite * 2

    im2 = ax2.imshow(
        plot_spreads,
        extent=[m1_values[0], m1_values[-1], alpha_values[0], alpha_values[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax2.set_xlabel("Noise intensity m1")
    ax2.set_ylabel("Alpha (spread-sensitivity)")
    ax2.set_title("Mean Spread (darker=smaller)")
    plt.colorbar(im2, ax=ax2, label="Mean spread")

    plt.tight_layout()
    plt.savefig("depletion_scripts/phase_diagram.png", dpi=150)
    print("\nSaved phase diagram to depletion_scripts/phase_diagram.png")

    # Find and plot the critical boundary
    print("\n" + "=" * 70)
    print("Critical boundary: m1_c(alpha)")
    print("=" * 70)

    # For each alpha, find the critical m1 (last stable m1)
    critical_m1 = []
    for i, alpha in enumerate(alpha_values):
        stable_m1s = m1_values[stability[i, :] == 1]
        if len(stable_m1s) > 0:
            critical_m1.append(np.max(stable_m1s))
        else:
            critical_m1.append(0)
        print(f"alpha={alpha:.1f}: m1_c = {critical_m1[-1]:.0f}")

    # Plot critical boundary
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.plot(alpha_values, critical_m1, "bo-", markersize=10, linewidth=2)
    ax.fill_between(alpha_values, 0, critical_m1, alpha=0.3, color="green", label="Stable")
    ax.fill_between(
        alpha_values, critical_m1, max(m1_values), alpha=0.3, color="red", label="Depleted"
    )
    ax.set_xlabel("Alpha (spread-sensitivity)", fontsize=12)
    ax.set_ylabel("Critical noise intensity m1_c", fontsize=12)
    ax.set_title("Critical Boundary: Higher alpha allows higher noise", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(alpha_values))
    ax.set_ylim(0, max(m1_values) * 1.1)

    plt.tight_layout()
    plt.savefig("depletion_scripts/critical_boundary.png", dpi=150)
    print("\nSaved critical boundary to depletion_scripts/critical_boundary.png")


if __name__ == "__main__":
    main()
