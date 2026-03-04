"""
Experiment 1: Verify square-root impact law is preserved with alpha > 0.

The classical LLOB predicts price impact ~ sqrt(Q) where Q is cumulative volume.
We verify this scaling is preserved when spread-dependent deposition (alpha > 0)
is active.

This experiment:
1. Runs simulations with constant metaorder m0 at various alpha values
2. Measures price impact over time
3. Fits power law to check if sqrt scaling is preserved
"""

import multiprocessing as mp
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from llob import Simulation

# ============================================================================
# PARAMETERS
# Matching Matthieu's conventions: D=0.5, L=10, X=500 (large book)
# We use nu=0.1 to enable deposition/cancellation (required for alpha mechanism)
# ============================================================================

# --- TEST parameters (fast, finish in seconds) ---
PARAMS = {"model_type": "discrete", "duration": 100.0, "n_frames": 20, "n_grid": 200, "xmin": -100.0, "xmax": 100.0, "D": 0.5, "L": 10.0, "nu": 0.1}
M0_VALUES = [50, 100]
ALPHA_VALUES = [0.0, 1.0, 3.0]
N_SEEDS = 5

# --- PRODUCTION parameters ---
# PARAMS = {"model_type": "discrete", "duration": 1000.0, "n_frames": 100, "n_grid": 1000, "xmin": -500.0, "xmax": 500.0, "D": 0.5, "L": 10.0, "nu": 0.1}
# M0_VALUES = [50, 100, 200, 500]
# ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
# N_SEEDS = 50

EXPERIMENTS_PATH = Path(__file__).parent / "experiment1/"

# Output file
RESULTS_FILE = Path(EXPERIMENTS_PATH / "exp1_sqrtimpact_results.pkl")


def run_single_sim(args: tuple) -> dict:
    """Run a single simulation and return impact trajectory."""
    m0, alpha, seed = args

    lambd = PARAMS["L"] * np.sqrt(PARAMS["nu"] * PARAMS["D"])
    n_frames = PARAMS["n_frames"]
    dt = PARAMS["duration"] / n_frames

    np.random.seed(seed)

    sim = Simulation.from_params(
        **PARAMS,
        lambd=lambd,
        metaorder=[m0],  # Constant metaorder
        alpha=alpha,
        seed=seed,
    )

    try:
        sim.run()

        # Cumulative volume at each frame
        Q = m0 * dt * np.arange(1, n_frames + 1)

        return {
            "m0": m0,
            "alpha": alpha,
            "seed": seed,
            "success": True,
            "prices": sim.prices.copy(),
            "spreads": sim.spreads.copy(),
            "Q": Q,
            "time": sim.time_interval.copy(),
        }
    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {
                "m0": m0,
                "alpha": alpha,
                "seed": seed,
                "success": False,
            }
        raise


def run_experiment():
    """Run all simulations."""
    seeds = list(range(42, 42 + N_SEEDS))

    all_tasks = []
    for m0 in M0_VALUES:
        for alpha in ALPHA_VALUES:
            for seed in seeds:
                all_tasks.append((m0, alpha, seed))

    print(f"Running {len(all_tasks)} simulations")
    print(f"  m0 values: {M0_VALUES}")
    print(f"  alpha values: {ALPHA_VALUES}")
    print(f"  seeds per (m0, alpha): {N_SEEDS}")

    n_workers = min(mp.cpu_count(), 8)
    print(f"Using {n_workers} workers")

    with mp.Pool(n_workers) as pool:
        # imap_unordered allows workers to return results as they complete
        # which gives smoother progress updates and better parallelization
        results = list(tqdm(
            pool.imap_unordered(run_single_sim, all_tasks),
            total=len(all_tasks),
            desc="Simulations"
        ))

    return results


def process_results(results):
    """Process results into summary statistics."""
    data = {}

    for m0 in M0_VALUES:
        data[m0] = {}
        for alpha in ALPHA_VALUES:
            matching = [
                r
                for r in results
                if r["m0"] == m0 and r["alpha"] == alpha and r["success"]
            ]

            if not matching:
                continue

            # Stack all price trajectories
            prices = np.array([r["prices"] for r in matching])
            spreads = np.array([r["spreads"] for r in matching])
            Q = matching[0]["Q"]
            time = matching[0]["time"]

            data[m0][alpha] = {
                "prices_mean": np.mean(prices, axis=0),
                "prices_std": np.std(prices, axis=0),
                "spreads_mean": np.mean(spreads, axis=0),
                "spreads_std": np.std(spreads, axis=0),
                "Q": Q,
                "time": time,
                "n_success": len(matching),
            }

    return data


def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def fit_impact_exponent(data):
    """Fit power law to impact vs Q and return exponents."""
    fit_results = {}

    for m0 in data:
        fit_results[m0] = {}
        for alpha in data[m0]:
            d = data[m0][alpha]
            Q = d["Q"]
            impact = d["prices_mean"]

            # Skip first few points (transient) and fit
            start_idx = max(2, len(Q) // 5)
            Q_fit = Q[start_idx:]
            impact_fit = impact[start_idx:]

            # Only fit positive impacts
            mask = impact_fit > 0
            if np.sum(mask) < 3:
                continue

            try:
                popt, pcov = curve_fit(
                    power_law, Q_fit[mask], impact_fit[mask], p0=[1.0, 0.5], maxfev=5000
                )
                a, b = popt
                perr = np.sqrt(np.diag(pcov))

                fit_results[m0][alpha] = {
                    "a": a,
                    "b": b,  # This should be ~0.5 for sqrt law
                    "a_err": perr[0],
                    "b_err": perr[1],
                }
                print(f"m0={m0}, alpha={alpha}: impact ~ Q^{b:.3f} ± {perr[1]:.3f}")
            except Exception as e:
                print(f"m0={m0}, alpha={alpha}: fit failed ({e})")

    return fit_results


def make_plots(data, fit_results):
    """Create diagnostic plots."""
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(ALPHA_VALUES)))

    # Figure 1: Impact vs Q for each m0
    n_m0 = len(M0_VALUES)
    fig1, axes = plt.subplots(1, n_m0, figsize=(6 * n_m0, 5))
    if n_m0 == 1:
        axes = [axes]

    for idx, m0 in enumerate(M0_VALUES):
        ax = axes[idx]
        if m0 not in data:
            continue

        for j, alpha in enumerate(ALPHA_VALUES):
            if alpha not in data[m0]:
                continue
            d = data[m0][alpha]
            Q = d["Q"]
            impact = d["prices_mean"]
            impact_err = d["prices_std"] / np.sqrt(d["n_success"])

            label = f"α={alpha}"
            if m0 in fit_results and alpha in fit_results[m0]:
                b = fit_results[m0][alpha]["b"]
                label += f" (b={b:.2f})"

            ax.errorbar(
                Q,
                impact,
                yerr=impact_err,
                color=colors[j],
                marker="o",
                markersize=4,
                capsize=2,
                label=label,
                alpha=0.8,
            )

        # Add theoretical sqrt reference
        if m0 in data and ALPHA_VALUES[0] in data[m0]:
            Q_ref = data[m0][ALPHA_VALUES[0]]["Q"]
            L = PARAMS["L"]
            impact_th = np.sqrt(2 * Q_ref / L)
            ax.plot(Q_ref, impact_th, "k--", lw=2, label=r"$\sqrt{2Q/L}$ (theory)")

        ax.set_xlabel("Cumulative Volume Q")
        ax.set_ylabel("Price Impact")
        ax.set_title(f"m₀ = {m0}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / "exp1_impact_vs_Q.png", dpi=150)
    print("Saved exp1_impact_vs_Q.png")

    # Figure 2: Log-log to check power law
    fig2, axes = plt.subplots(1, n_m0, figsize=(6 * n_m0, 5))
    if n_m0 == 1:
        axes = [axes]

    for idx, m0 in enumerate(M0_VALUES):
        ax = axes[idx]
        if m0 not in data:
            continue

        for j, alpha in enumerate(ALPHA_VALUES):
            if alpha not in data[m0]:
                continue
            d = data[m0][alpha]
            Q = d["Q"]
            impact = d["prices_mean"]

            mask = (Q > 0) & (impact > 0)
            ax.scatter(Q[mask], impact[mask], color=colors[j], s=30, label=f"α={alpha}")

            # Add fit line
            if m0 in fit_results and alpha in fit_results[m0]:
                fr = fit_results[m0][alpha]
                Q_fit = np.linspace(Q[mask].min(), Q[mask].max(), 100)
                ax.plot(
                    Q_fit,
                    power_law(Q_fit, fr["a"], fr["b"]),
                    "--",
                    color=colors[j],
                    alpha=0.7,
                )

        # Reference line with slope 0.5
        if m0 in data and ALPHA_VALUES[0] in data[m0]:
            Q_ref = data[m0][ALPHA_VALUES[0]]["Q"]
            ax.plot(
                Q_ref, 0.5 * np.sqrt(Q_ref), "k:", lw=2, alpha=0.5, label="slope=0.5"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cumulative Volume Q")
        ax.set_ylabel("Price Impact")
        ax.set_title(f"m₀ = {m0} (log-log)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / "exp1_impact_loglog.png", dpi=150)
    print("Saved exp1_impact_loglog.png")

    # Figure 3: Exponent vs alpha
    fig3, ax = plt.subplots(figsize=(8, 5))

    for idx, m0 in enumerate(M0_VALUES):
        if m0 not in fit_results:
            continue

        alphas = []
        exponents = []
        errors = []
        for alpha in ALPHA_VALUES:
            if alpha in fit_results[m0]:
                alphas.append(alpha)
                exponents.append(fit_results[m0][alpha]["b"])
                errors.append(fit_results[m0][alpha]["b_err"])

        if alphas:
            ax.errorbar(
                alphas,
                exponents,
                yerr=errors,
                marker="o",
                markersize=8,
                capsize=4,
                label=f"m₀={m0}",
            )

    ax.axhline(0.5, color="k", ls="--", lw=2, label="sqrt law (b=0.5)")
    ax.set_xlabel("α (spread sensitivity)")
    ax.set_ylabel("Power law exponent b")
    ax.set_title("Impact scaling exponent vs α")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / "exp1_exponent_vs_alpha.png", dpi=150)
    print("Saved exp1_exponent_vs_alpha.png")


def main(rerun=False):
    print("=" * 70)
    print("Experiment 1: Square-root impact law with alpha > 0")
    print("=" * 70)

    # Ensure output directory exists
    EXPERIMENTS_PATH.mkdir(parents=True, exist_ok=True)

    if RESULTS_FILE.exists() and not rerun:
        print(f"Loading from {RESULTS_FILE}")
        with open(RESULTS_FILE, "rb") as f:
            saved = pickle.load(f)
        results = saved["results"]
        print(f"Loaded {len(results)} results")
    else:
        results = run_experiment()

        with open(RESULTS_FILE, "wb") as f:
            pickle.dump(
                {
                    "results": results,
                    "params": PARAMS,
                    "m0_values": M0_VALUES,
                    "alpha_values": ALPHA_VALUES,
                    "n_seeds": N_SEEDS,
                },
                f,
            )
        print(f"Saved to {RESULTS_FILE}")

    # Process and analyze
    data = process_results(results)
    fit_results = fit_impact_exponent(data)

    # Make plots
    make_plots(data, fit_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Impact scaling exponents")
    print("=" * 70)
    print("(b should be ~0.5 if sqrt law is preserved)")
    print()
    for m0 in fit_results:
        for alpha in fit_results[m0]:
            fr = fit_results[m0][alpha]
            print(
                f"m0={m0:3d}, alpha={alpha:.1f}: b = {fr['b']:.3f} ± {fr['b_err']:.3f}"
            )

    print("\nDone!")


if __name__ == "__main__":
    import sys

    rerun = "--rerun" in sys.argv
    main(rerun=rerun)
