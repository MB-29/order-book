"""
Discrimination experiment: Power-law vs Exponential for std(spread) vs alpha.

Uses a finer alpha grid and wider range to better distinguish functional forms.
Focuses on a single m1 value for cleaner analysis.
"""

import multiprocessing as mp
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from llob import Simulation

# Parameters
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

N_SIMS = 50
RESULTS_FILE = Path("depletion_scripts/discrimination_results.pkl")


def run_simulation(args: tuple) -> dict:
    """Run a single simulation."""
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
            "seed": seed,
            "success": True,
            "std_spread": np.std(stationary_spreads),
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {
                "alpha": alpha,
                "m1": m1,
                "seed": seed,
                "success": False,
                "std_spread": np.inf,
            }
        raise


def run_experiment(m1, alpha_values):
    """Run simulations for all alpha values."""
    base_seed = 42
    seeds = list(range(base_seed, base_seed + N_SIMS))

    all_tasks = []
    for alpha in alpha_values:
        for seed in seeds:
            all_tasks.append((alpha, m1, seed))

    print(f"Running {len(all_tasks)} simulations (m1={m1}, {len(alpha_values)} alphas, {N_SIMS} seeds each)")

    n_workers = min(mp.cpu_count(), 8)
    print(f"Using {n_workers} workers")

    with mp.Pool(n_workers) as pool:
        results = pool.map(run_simulation, all_tasks)

    return results


def process_results(results, alpha_values):
    """Process results into summary statistics."""
    data = {"alphas": [], "stds": [], "sems": [], "success_rates": []}

    for alpha in alpha_values:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        successful = [r for r in alpha_results if r["success"]]

        if successful:
            std_vals = [r["std_spread"] for r in successful]
            data["alphas"].append(alpha)
            data["stds"].append(np.mean(std_vals))
            data["sems"].append(np.std(std_vals) / np.sqrt(len(std_vals)))
            data["success_rates"].append(len(successful) / len(alpha_results))

    for k in data:
        data[k] = np.array(data[k])

    return data


def discrimination_analysis(data):
    """Analyze power-law vs exponential fits."""
    # Filter for valid data:
    # - positive alpha
    # - std > floor (exclude numerical zeros where spread is locked at minimum)
    # - std > 0.005 to exclude the "saturated" regime where spread can't decrease further
    std_floor = 0.01  # Exclude points where we've hit the minimum spread floor
    mask = (data["alphas"] > 0) & (data["stds"] > std_floor)
    x = data["alphas"][mask]
    y = data["stds"][mask]

    print(f"Using {len(x)} data points with std > {std_floor}")

    if len(x) < 5:
        print("Not enough valid data points!")
        return None

    log_x = np.log(x)
    log_y = np.log(y)

    # Power law fit: log(y) = log(a) + b*log(x)
    pl_coeffs = np.polyfit(log_x, log_y, 1)
    pl_b, pl_log_a = pl_coeffs
    pl_a = np.exp(pl_log_a)
    pl_pred = pl_log_a + pl_b * log_x
    pl_residuals = log_y - pl_pred
    pl_ss_res = np.sum(pl_residuals**2)
    pl_ss_tot = np.sum((log_y - np.mean(log_y))**2)
    pl_r2 = 1 - pl_ss_res / pl_ss_tot

    # Exponential fit: log(y) = log(a) - b*x
    exp_coeffs = np.polyfit(x, log_y, 1)
    exp_neg_b, exp_log_a = exp_coeffs
    exp_a = np.exp(exp_log_a)
    exp_b = -exp_neg_b
    exp_pred = exp_log_a + exp_neg_b * x
    exp_residuals = log_y - exp_pred
    exp_ss_res = np.sum(exp_residuals**2)
    exp_ss_tot = np.sum((log_y - np.mean(log_y))**2)
    exp_r2 = 1 - exp_ss_res / exp_ss_tot

    return {
        "x": x,
        "y": y,
        "log_x": log_x,
        "log_y": log_y,
        "power_law": {"a": pl_a, "b": pl_b, "r2": pl_r2, "residuals": pl_residuals},
        "exponential": {"a": exp_a, "b": exp_b, "r2": exp_r2, "residuals": exp_residuals},
    }


def make_plots(data, analysis, m1):
    """Create discrimination plots."""
    x = analysis["x"]
    y = analysis["y"]
    pl = analysis["power_law"]
    exp = analysis["exponential"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Data with fits in different scales
    # Only plot data points that were used in the fit (std > floor)
    std_floor = 0.01
    fit_mask = data["stds"] > std_floor
    excluded_mask = (data["stds"] > 0) & (data["stds"] <= std_floor)

    # Log-log (power law should be linear here)
    ax = axes[0, 0]
    ax.errorbar(data["alphas"][fit_mask], data["stds"][fit_mask], yerr=data["sems"][fit_mask],
                fmt='o', color='blue', markersize=6, capsize=3, label='Data (used)')
    if np.any(excluded_mask):
        ax.scatter(data["alphas"][excluded_mask], data["stds"][excluded_mask],
                   marker='x', color='gray', s=40, label='Excluded (floor)')
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_pl = pl["a"] * np.power(x_fit, pl["b"])
    y_exp = exp["a"] * np.exp(-exp["b"] * x_fit)
    ax.plot(x_fit, y_pl, 'k-', linewidth=2, label=f'Power: α^{pl["b"]:.2f}')
    ax.plot(x_fit, y_exp, 'r--', linewidth=2, label=f'Exp: e^(-{exp["b"]:.2f}α)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_title("Log-Log (power law linear here)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Linear-log (exponential should be linear here)
    ax = axes[0, 1]
    ax.errorbar(data["alphas"][fit_mask], data["stds"][fit_mask], yerr=data["sems"][fit_mask],
                fmt='o', color='blue', markersize=6, capsize=3, label='Data (used)')
    if np.any(excluded_mask):
        ax.scatter(data["alphas"][excluded_mask], data["stds"][excluded_mask],
                   marker='x', color='gray', s=40, label='Excluded')
    ax.plot(x_fit, y_pl, 'k-', linewidth=2, label=f'Power: α^{pl["b"]:.2f}')
    ax.plot(x_fit, y_exp, 'r--', linewidth=2, label=f'Exp: e^(-{exp["b"]:.2f}α)')
    ax.set_yscale('log')
    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_title("Linear-Log (exponential linear here)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Linear-linear
    ax = axes[0, 2]
    ax.errorbar(data["alphas"][fit_mask], data["stds"][fit_mask], yerr=data["sems"][fit_mask],
                fmt='o', color='blue', markersize=6, capsize=3, label='Data (used)')
    if np.any(excluded_mask):
        ax.scatter(data["alphas"][excluded_mask], data["stds"][excluded_mask],
                   marker='x', color='gray', s=40, label='Excluded')
    ax.plot(x_fit, y_pl, 'k-', linewidth=2, label=f'Power: α^{pl["b"]:.2f}')
    ax.plot(x_fit, y_exp, 'r--', linewidth=2, label=f'Exp: e^(-{exp["b"]:.2f}α)')
    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_title("Linear-Linear")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: Residual analysis

    # Residuals vs alpha
    ax = axes[1, 0]
    ax.scatter(x, pl["residuals"], marker='o', color='black', s=50, label='Power law', zorder=5)
    ax.scatter(x, exp["residuals"], marker='s', color='red', s=50, label='Exponential', zorder=5)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("α")
    ax.set_ylabel("Residual in log(std)")
    ax.set_title("Residuals vs α")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals vs log(alpha) - to see systematic trends
    ax = axes[1, 1]
    ax.scatter(analysis["log_x"], pl["residuals"], marker='o', color='black', s=50, label='Power law')
    ax.scatter(analysis["log_x"], exp["residuals"], marker='s', color='red', s=50, label='Exponential')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("log(α)")
    ax.set_ylabel("Residual in log(std)")
    ax.set_title("Residuals vs log(α)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
    DISCRIMINATION SUMMARY (m₁ = {m1})

    Power Law:  std = {pl['a']:.4f} × α^({pl['b']:.3f})
                R² = {pl['r2']:.4f}

    Exponential: std = {exp['a']:.4f} × exp(-{exp['b']:.3f} × α)
                 R² = {exp['r2']:.4f}

    ΔR² = {pl['r2'] - exp['r2']:.4f}

    Winner: {'POWER LAW' if pl['r2'] > exp['r2'] else 'EXPONENTIAL'}

    Notes:
    - If power law: should be linear in log-log (top-left)
    - If exponential: should be linear in linear-log (top-middle)
    - Residuals show systematic deviations from fits
    - Look for curvature in residuals to identify wrong model
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle(f"Power-law vs Exponential Discrimination (m₁={m1})", fontsize=14)
    plt.tight_layout()
    plt.savefig("depletion_scripts/discrimination_analysis.png", dpi=150)
    print("Saved discrimination_analysis.png")


def main(rerun=False):
    print("=" * 70)
    print("Discrimination Experiment: Power-law vs Exponential")
    print("=" * 70)

    # Single m1 value, fine alpha grid with wider range
    m1 = 50  # Good balance: high enough success rate, clear signal
    # Finer grid, logarithmically spaced for better coverage
    alpha_values = np.array([
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
        5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0
    ])

    if RESULTS_FILE.exists() and not rerun:
        print(f"Loading from {RESULTS_FILE}")
        with open(RESULTS_FILE, "rb") as f:
            saved = pickle.load(f)
        results = saved["results"]
        m1 = saved["m1"]
        alpha_values = np.array(saved["alpha_values"])
        print(f"Loaded {len(results)} results")
    else:
        results = run_experiment(m1, alpha_values)

        with open(RESULTS_FILE, "wb") as f:
            pickle.dump({
                "results": results,
                "m1": m1,
                "alpha_values": alpha_values,
                "params": PARAMS,
                "n_sims": N_SIMS,
            }, f)
        print(f"Saved to {RESULTS_FILE}")

    # Process
    data = process_results(results, alpha_values)

    print(f"\nData summary (m1={m1}):")
    print(f"  alpha range: {data['alphas'].min():.1f} - {data['alphas'].max():.1f}")
    print(f"  std range: {data['stds'].min():.4f} - {data['stds'].max():.4f}")
    print(f"  success rates: {data['success_rates'].min()*100:.0f}% - {data['success_rates'].max()*100:.0f}%")

    # Analyze
    analysis = discrimination_analysis(data)

    if analysis:
        pl = analysis["power_law"]
        exp = analysis["exponential"]

        print(f"\n{'='*60}")
        print("FIT RESULTS")
        print(f"{'='*60}")
        print(f"Power law:   std = {pl['a']:.4f} × α^({pl['b']:.3f}), R² = {pl['r2']:.4f}")
        print(f"Exponential: std = {exp['a']:.4f} × exp(-{exp['b']:.3f}×α), R² = {exp['r2']:.4f}")
        print(f"\nΔR² = {pl['r2'] - exp['r2']:.4f}")

        if pl['r2'] > exp['r2']:
            print("  --> POWER LAW fits better")
        else:
            print("  --> EXPONENTIAL fits better")

        # Make plots
        make_plots(data, analysis, m1)

    print("\nDone!")


if __name__ == "__main__":
    import sys
    rerun = "--rerun" in sys.argv
    main(rerun=rerun)
