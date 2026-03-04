"""
Quick test to verify alpha stabilizes spread in high-noise regime.

This script tests whether alpha > 0 prevents spread blowup when m_1 is large.

Physics analysis:
- Maximum spread: s_max = xmax - xmin (when book is depleted)
- Deposition rate with alpha: λ_eff = λ_0 + α * s
- For stability: deposition must counter consumption

Key constraint for finite book:
- If spread reaches s_max, deposition is λ_eff = λ_0 + α * s_max
- This needs to replenish faster than noise consumes
"""

import numpy as np
from llob import Simulation

# Parameters for high-noise regime
PARAMS = {
    "model_type": "discrete",
    "duration": 500.0,  # Shorter for quick test
    "n_frames": 100,
    "n_grid": 100,
    "xmin": -50.0,
    "xmax": 50.0,
    "D": 0.5,
    "L": 10.0,
    "nu": 0.1,
}


def run_simulation(alpha: float, m1: float, seed: int = 42) -> dict:
    """Run simulation and return spread statistics."""
    # Compute lambd from L, nu, D
    lambd = PARAMS["L"] * np.sqrt(PARAMS["nu"] * PARAMS["D"])

    # Create metaorder with noise but no deterministic component
    # m_t = m_0 + m_1 * dW_t (fractional Brownian motion)
    # For simplicity, use sinusoidal noise to represent m_1 effect
    n_frames = PARAMS["n_frames"]
    np.random.seed(seed)
    metaorder = m1 * np.random.randn(n_frames)  # Random noise

    sim = Simulation.from_params(
        **PARAMS,
        lambd=lambd,
        metaorder=metaorder,
        alpha=alpha,
        seed=seed,
    )

    sim.run()

    return {
        "spreads": sim.spreads.copy(),
        "mean_spread": np.mean(sim.spreads),
        "max_spread": np.max(sim.spreads),
        "final_spread": sim.spreads[-1],
        "spread_growth": sim.spreads[-1] - sim.spreads[0],
    }


def main():
    print("=" * 60)
    print("Testing alpha stabilization for spread in high-noise regime")
    print("=" * 60)

    # Book parameters
    s_max = PARAMS["xmax"] - PARAMS["xmin"]
    print(f"\nBook size: [{PARAMS['xmin']}, {PARAMS['xmax']}]")
    print(f"Maximum possible spread: {s_max}")

    # Test different noise levels and alpha values
    m1_values = [1.0, 5.0, 10.0, 20.0]
    alpha_values = [0.0, 0.01, 0.1, 0.5, 1.0]

    print("\n" + "-" * 60)
    print("Results: Mean spread for different (m1, alpha) combinations")
    print("-" * 60)

    # Header
    header = f"{'m1':>8} |" + "".join(f" α={a:>5} |" for a in alpha_values)
    print(header)
    print("-" * len(header))

    for m1 in m1_values:
        row = f"{m1:>8.1f} |"
        for alpha in alpha_values:
            result = run_simulation(alpha, m1)
            row += f" {result['mean_spread']:>6.2f} |"
        print(row)

    print("-" * 60)

    # Detailed analysis for high noise case
    print("\n" + "=" * 60)
    print("Detailed analysis for m1 = 20 (high noise)")
    print("=" * 60)

    for alpha in alpha_values:
        result = run_simulation(alpha, m1=20.0)
        print(f"\nalpha = {alpha}:")
        print(f"  Mean spread:   {result['mean_spread']:.2f}")
        print(f"  Max spread:    {result['max_spread']:.2f}")
        print(f"  Final spread:  {result['final_spread']:.2f}")
        print(f"  Spread growth: {result['spread_growth']:.2f}")

    # Estimate required alpha
    print("\n" + "=" * 60)
    print("Estimating required alpha for stability")
    print("=" * 60)

    # The deposition rate per grid point is: λ_eff * dt * dx
    # Total deposition rate ~ λ_eff * dt * dx * n_active_points
    # For stability, this should exceed consumption rate m1

    lambd = PARAMS["L"] * np.sqrt(PARAMS["nu"] * PARAMS["D"])
    dx = (PARAMS["xmax"] - PARAMS["xmin"]) / PARAMS["n_grid"]
    D = PARAMS["D"]
    dt = dx**2 / (2 * D)

    print(f"\nGrid spacing dx = {dx:.4f}")
    print(f"Time step dt = {dt:.6f}")
    print(f"Base lambda = {lambd:.4f}")

    # Rough estimate: for spread s, deposition adds ~ (lambd + alpha*s) * dt * dx * n_grid orders
    # Consumption is ~ m1 * dt per timestep
    # Balance: (lambd + alpha*s) * dt * dx * n_grid ~ m1 * dt
    # => alpha*s ~ m1 / (dx * n_grid) - lambd
    # => alpha ~ (m1 / (dx * n_grid) - lambd) / s_target

    for m1 in [10.0, 20.0, 50.0]:
        for s_target in [5.0, 10.0, 20.0]:
            alpha_needed = max(0, (m1 / (dx * PARAMS["n_grid"]) - lambd) / s_target)
            print(f"m1={m1:>5}, target spread={s_target:>5} => alpha needed ~ {alpha_needed:.4f}")


if __name__ == "__main__":
    main()
