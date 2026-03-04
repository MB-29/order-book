"""
Plot std(spread) vs alpha for fixed m1 values.

This shows how the spread variability decreases with stronger feedback.
Uses many simulations for robust statistics.
"""

import multiprocessing as mp
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

# Number of simulations per (alpha, m1) pair for statistics
N_SIMS = 50

RESULTS_FILE = Path("depletion_scripts/spread_std_results.pkl")


def run_simulation(args: tuple) -> dict:
    """Run a single simulation for given (alpha, m1, seed) triplet."""
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
            "max_spread": np.max(sim.spreads),
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {
                "alpha": alpha,
                "m1": m1,
                "seed": seed,
                "success": False,
                "std_spread": np.inf,
                "max_spread": np.inf,
            }
        raise


def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def exponential(x, a, b):
    """Exponential decay: y = a * exp(-b * x)"""
    return a * np.exp(-b * x)


def run_simulations(m1_values, alpha_values):
    """Run all simulations and return raw results."""
    base_seed = 42
    seeds = list(range(base_seed, base_seed + N_SIMS))

    all_tasks = []
    for m1 in m1_values:
        for alpha in alpha_values:
            for seed in seeds:
                all_tasks.append((alpha, m1, seed))

    print(f"Total simulations to run: {len(all_tasks)}")

    n_workers = min(mp.cpu_count(), 8)
    print(f"Using {n_workers} workers")

    with mp.Pool(n_workers) as pool:
        all_results = pool.map(run_simulation, all_tasks)

    return all_results


def process_results(all_results, m1_values, alpha_values):
    """Process raw results into summary statistics."""
    results_by_m1 = {}

    for m1 in m1_values:
        m1_results = [r for r in all_results if r["m1"] == m1]

        std_spreads = {}
        success_rate = {}

        for alpha in alpha_values:
            alpha_results = [r for r in m1_results if r["alpha"] == alpha]
            successful = [r for r in alpha_results if r["success"]]

            if successful:
                std_vals = [r["std_spread"] for r in successful]
                std_spreads[alpha] = {
                    "mean": np.mean(std_vals),
                    "std": np.std(std_vals) / np.sqrt(len(std_vals)),  # SEM
                    "all_values": std_vals,
                }
            else:
                std_spreads[alpha] = {"mean": np.inf, "std": 0, "all_values": []}

            success_rate[alpha] = len(successful) / len(alpha_results)

        results_by_m1[m1] = {
            "std_spreads": std_spreads,
            "success_rate": success_rate,
        }

    return results_by_m1


def print_summary(results_by_m1, alpha_values):
    """Print summary table."""
    for m1, data in results_by_m1.items():
        print(f"\nm1 = {m1}:")
        print(f"  alpha | std(spread) ± SEM  | success rate")
        print(f"  {'-'*45}")
        for alpha in alpha_values:
            std_val = data["std_spreads"][alpha]["mean"]
            sem = data["std_spreads"][alpha]["std"]
            sr = data["success_rate"][alpha]
            if np.isfinite(std_val):
                print(f"  {alpha:5.1f} | {std_val:6.3f} ± {sem:5.3f}   | {sr*100:5.0f}%")
            else:
                print(f"  {alpha:5.1f} | {'depleted':>17} | {sr*100:5.0f}%")


def extract_plot_data(results_by_m1, alpha_values):
    """Extract plottable data (finite values only) for each m1."""
    plot_data = {}
    for m1, data in results_by_m1.items():
        alphas = []
        stds = []
        errs = []
        for alpha in alpha_values:
            if np.isfinite(data["std_spreads"][alpha]["mean"]):
                alphas.append(alpha)
                stds.append(data["std_spreads"][alpha]["mean"])
                errs.append(data["std_spreads"][alpha]["std"])
        plot_data[m1] = {
            "alphas": np.array(alphas),
            "stds": np.array(stds),
            "errs": np.array(errs),
        }
    return plot_data


def make_exploratory_plots(plot_data, alpha_values):
    """Create exploratory plots with different scales and fits."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    m1_list = list(plot_data.keys())

    # Figure 1: Basic linear plot (for report)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        ax1.errorbar(
            d["alphas"], d["stds"], yerr=d["errs"],
            marker=markers[idx], color=colors[idx],
            linewidth=2, markersize=8, capsize=4,
            label=f"m₁ = {m1}"
        )
    ax1.set_xlabel("α (spread-sensitivity)", fontsize=12)
    ax1.set_ylabel("std(spread) [ticks]", fontsize=12)
    ax1.set_title(f"Spread Variability vs α (N={N_SIMS} sims/point)", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, max(alpha_values) + 0.5)
    ax1.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_std_vs_alpha.png", dpi=150)
    print("Saved spread_std_vs_alpha.png")

    # Figure 2: 2x2 grid with different scales
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Linear-Linear
    ax = axes[0, 0]
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        ax.errorbar(d["alphas"], d["stds"], yerr=d["errs"],
                    marker=markers[idx], color=colors[idx], linewidth=2,
                    markersize=6, capsize=3, label=f"m₁={m1}")
    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_title("Linear-Linear")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log-Linear (log y)
    ax = axes[0, 1]
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        mask = d["stds"] > 0
        ax.errorbar(d["alphas"][mask], d["stds"][mask], yerr=d["errs"][mask],
                    marker=markers[idx], color=colors[idx], linewidth=2,
                    markersize=6, capsize=3, label=f"m₁={m1}")
    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_yscale("log")
    ax.set_title("Linear-Log (log y)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Linear-Log (log x)
    ax = axes[1, 0]
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        mask = d["alphas"] > 0
        ax.errorbar(d["alphas"][mask], d["stds"][mask], yerr=d["errs"][mask],
                    marker=markers[idx], color=colors[idx], linewidth=2,
                    markersize=6, capsize=3, label=f"m₁={m1}")
    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_xscale("log")
    ax.set_title("Log-Linear (log x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log-Log (with floor to avoid outliers from numerical zeros)
    ax = axes[1, 1]
    std_floor = 1e-3  # Floor to avoid compression from numerical zeros
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        mask = (d["alphas"] > 0) & (d["stds"] > std_floor)
        ax.errorbar(d["alphas"][mask], d["stds"][mask], yerr=d["errs"][mask],
                    marker=markers[idx], color=colors[idx], linewidth=2,
                    markersize=6, capsize=3, label=f"m₁={m1}")
    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Log-Log (std > {std_floor})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_std_scales.png", dpi=150)
    print("Saved spread_std_scales.png")

    # Figure 3: Fits
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Power law fits
    ax = axes[0]
    print("\n" + "="*60)
    print("POWER LAW FITS: std(spread) = a * alpha^b")
    print("="*60)
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        mask = (d["alphas"] > 0) & (d["stds"] > 0)
        x = d["alphas"][mask]
        y = d["stds"][mask]

        ax.scatter(x, y, marker=markers[idx], color=colors[idx], s=60, label=f"m₁={m1}")

        if len(x) >= 3:
            try:
                popt, _ = curve_fit(power_law, x, y, p0=[0.5, -1], maxfev=5000)
                a, b = popt
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = power_law(x_fit, a, b)
                ax.plot(x_fit, y_fit, '--', color=colors[idx], alpha=0.7)
                print(f"m1={m1}: std = {a:.4f} * α^({b:.3f})")
            except Exception as e:
                print(f"m1={m1}: Power law fit failed: {e}")

    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Power Law Fits (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Exponential fits
    ax = axes[1]
    print("\n" + "="*60)
    print("EXPONENTIAL FITS: std(spread) = a * exp(-b * alpha)")
    print("="*60)
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        mask = d["stds"] > 0
        x = d["alphas"][mask]
        y = d["stds"][mask]

        ax.scatter(x, y, marker=markers[idx], color=colors[idx], s=60, label=f"m₁={m1}")

        if len(x) >= 3:
            try:
                popt, _ = curve_fit(exponential, x, y, p0=[0.5, 0.5], maxfev=5000)
                a, b = popt
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = exponential(x_fit, a, b)
                ax.plot(x_fit, y_fit, '--', color=colors[idx], alpha=0.7)
                print(f"m1={m1}: std = {a:.4f} * exp(-{b:.3f} * α)")
            except Exception as e:
                print(f"m1={m1}: Exponential fit failed: {e}")

    ax.set_xlabel("α")
    ax.set_ylabel("std(spread)")
    ax.set_yscale("log")
    ax.set_title("Exponential Fits (linear x, log y)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_std_fits.png", dpi=150)
    print("Saved spread_std_fits.png")

    # Figure 4: Collapsed plot - try rescaling
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Try alpha * m1 rescaling
    ax = axes[0]
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        mask = d["stds"] > 0
        x_rescaled = d["alphas"][mask] * m1
        ax.scatter(x_rescaled, d["stds"][mask], marker=markers[idx],
                   color=colors[idx], s=60, label=f"m₁={m1}")
    ax.set_xlabel("α × m₁")
    ax.set_ylabel("std(spread)")
    ax.set_title("Rescaled: α × m₁")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Try alpha / m1 rescaling
    ax = axes[1]
    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        mask = d["stds"] > 0
        x_rescaled = d["alphas"][mask] / m1
        ax.scatter(x_rescaled, d["stds"][mask], marker=markers[idx],
                   color=colors[idx], s=60, label=f"m₁={m1}")
    ax.set_xlabel("α / m₁")
    ax.set_ylabel("std(spread)")
    ax.set_title("Rescaled: α / m₁")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_std_rescaled.png", dpi=150)
    print("Saved spread_std_rescaled.png")


def discrimination_experiment(m1_values, alpha_values, plot_data):
    """
    Create plots specifically designed to discriminate power-law vs exponential.

    Key insight:
    - Power law: log(y) = log(a) + b*log(x)  -> linear in log-log
    - Exponential: log(y) = log(a) - b*x     -> linear in linear-log (semi-log)

    So we fit both and look at residuals.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    m1_list = list(plot_data.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    print("\n" + "="*70)
    print("DISCRIMINATION: Power-law vs Exponential")
    print("="*70)

    for idx, m1 in enumerate(m1_list):
        d = plot_data[m1]
        # Filter: positive alpha, positive std, std not too small (numerical zero)
        mask = (d["alphas"] > 0) & (d["stds"] > 1e-3)
        x = d["alphas"][mask]
        y = d["stds"][mask]

        if len(x) < 4:
            print(f"m1={m1}: Not enough data points for discrimination")
            continue

        # Fit power law in log-log space (linear regression on log-transformed data)
        log_x = np.log(x)
        log_y = np.log(y)
        pl_coeffs = np.polyfit(log_x, log_y, 1)
        pl_b, pl_log_a = pl_coeffs
        pl_a = np.exp(pl_log_a)
        pl_pred_log = pl_log_a + pl_b * log_x
        pl_residuals = log_y - pl_pred_log
        pl_ss_res = np.sum(pl_residuals**2)
        pl_ss_tot = np.sum((log_y - np.mean(log_y))**2)
        pl_r2 = 1 - pl_ss_res / pl_ss_tot

        # Fit exponential in linear-log space (linear regression on log(y) vs x)
        exp_coeffs = np.polyfit(x, log_y, 1)
        exp_neg_b, exp_log_a = exp_coeffs
        exp_a = np.exp(exp_log_a)
        exp_b = -exp_neg_b
        exp_pred_log = exp_log_a + exp_neg_b * x
        exp_residuals = log_y - exp_pred_log
        exp_ss_res = np.sum(exp_residuals**2)
        exp_ss_tot = np.sum((log_y - np.mean(log_y))**2)
        exp_r2 = 1 - exp_ss_res / exp_ss_tot

        print(f"\nm1={m1}:")
        print(f"  Power law:   std = {pl_a:.4f} * α^({pl_b:.3f}), R² = {pl_r2:.4f}")
        print(f"  Exponential: std = {exp_a:.4f} * exp(-{exp_b:.3f}*α), R² = {exp_r2:.4f}")

        if pl_r2 > exp_r2:
            print(f"  --> POWER LAW fits better (ΔR² = {pl_r2 - exp_r2:.4f})")
        else:
            print(f"  --> EXPONENTIAL fits better (ΔR² = {exp_r2 - pl_r2:.4f})")

        # Plot 1: Log-log with power law fit
        ax = axes[0, idx]
        ax.scatter(x, y, marker=markers[idx], color=colors[idx], s=80, zorder=5)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_pl = pl_a * np.power(x_fit, pl_b)
        y_exp = exp_a * np.exp(-exp_b * x_fit)
        ax.plot(x_fit, y_pl, 'k-', linewidth=2, label=f'Power: α^{pl_b:.2f}, R²={pl_r2:.3f}')
        ax.plot(x_fit, y_exp, 'r--', linewidth=2, label=f'Exp: e^(-{exp_b:.2f}α), R²={exp_r2:.3f}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("α")
        ax.set_ylabel("std(spread)")
        ax.set_title(f"m₁={m1}: Log-Log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Residuals
        ax = axes[1, idx]
        ax.scatter(x, pl_residuals, marker='o', color='black', s=50, label=f'Power law')
        ax.scatter(x, exp_residuals, marker='s', color='red', s=50, label=f'Exponential')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel("α")
        ax.set_ylabel("Residual in log(std)")
        ax.set_title(f"m₁={m1}: Residuals")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("depletion_scripts/spread_std_discrimination.png", dpi=150)
    print("\nSaved spread_std_discrimination.png")


def main(rerun=False):
    print("=" * 70)
    print("Spread std vs Alpha for fixed m1 values")
    print("=" * 70)

    # Check if we can load from pickle
    if RESULTS_FILE.exists() and not rerun:
        print(f"Loading results from {RESULTS_FILE}")
        with open(RESULTS_FILE, "rb") as f:
            results_data = pickle.load(f)
        all_results = results_data["all_results"]
        m1_values = results_data["m1_values"]
        alpha_values = np.array(results_data["alpha_values"])
        print(f"Loaded {len(all_results)} results")
        print(f"m1_values: {m1_values}")
        print(f"alpha_values: {alpha_values}")
    else:
        print(f"Running {N_SIMS} simulations per (alpha, m1) pair")
        m1_values = [50, 100, 150]
        alpha_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])

        # Run simulations
        all_results = run_simulations(m1_values, alpha_values)

        # Save raw results
        results_data = {
            "all_results": all_results,
            "m1_values": m1_values,
            "alpha_values": alpha_values,
            "params": PARAMS,
            "n_sims": N_SIMS,
        }
        with open(RESULTS_FILE, "wb") as f:
            pickle.dump(results_data, f)
        print(f"\nPickled results to {RESULTS_FILE}")

    # Process and summarize
    results_by_m1 = process_results(all_results, m1_values, alpha_values)
    print_summary(results_by_m1, alpha_values)

    # Extract plottable data
    plot_data = extract_plot_data(results_by_m1, alpha_values)

    # Make all plots
    make_exploratory_plots(plot_data, alpha_values)

    # Discrimination experiment
    discrimination_experiment(m1_values, alpha_values, plot_data)

    print("\nDone!")


if __name__ == "__main__":
    import sys
    rerun = "--rerun" in sys.argv
    main(rerun=rerun)
