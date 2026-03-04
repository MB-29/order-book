"""
Experiment 2: Density profile analysis in the moving (price) reference frame.

In the LLOB model, the order book density profile should have a specific shape
when viewed in the frame moving with the price. This experiment:

1. Runs simulations with various alpha values
2. Records the density profile at several time points
3. Shifts profiles to the moving frame (centered on mid-price)
4. Analyzes how alpha affects the density profile shape

Parameters match Matthieu's conventions: D=0.5, L=10, X=500
"""

import multiprocessing as mp
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from llob import Simulation

# ============================================================================
# PARAMETERS
# Matching Matthieu's conventions: D=0.5, L=10, X=500 (large book)
# ============================================================================

# --- TEST parameters (fast, finish in seconds) ---
# PARAMS = {"model_type": "discrete", "duration": 50.0, "n_frames": 20, "n_grid": 50, "xmin": -25.0, "xmax": 25.0, "D": 0.5, "L": 10.0, "nu": 0.1}
# M0, M1 = 0, 10
# ALPHA_VALUES = [0.0, 1.0, 5.0, 10.0]
# N_SEEDS = 3

# --- PRODUCTION parameters (Matthieu's settings: m1=50, H=0.7, N=200) ---
PARAMS = {
    "model_type": "discrete",
    "duration": 5000.0,
    "n_frames": 100,
    "n_grid": 800,
    "xmin": -400.0,
    "xmax": 400.0,
    "D": 0.5,
    "L": 10.0,
    "nu": 0.1,
}
M0, M1 = 0, 50
ALPHA_VALUES = [0.0, 1.0, 2.0, 5.0, 10.0]
N_SEEDS = 200

# Measurement frames (fraction of total frames)
MEASUREMENT_FRACTIONS = [0.25, 0.5, 0.75, 1.0]

# Output paths
EXPERIMENTS_PATH = Path(__file__).parent / "experiment2"
RESULTS_FILE = EXPERIMENTS_PATH / "exp2_density_results.pkl"


def run_single_with_profiles(args: tuple) -> dict:
    """
    Run simulation and record density profiles at multiple times.
    """
    alpha, seed, use_noise = args

    lambd = PARAMS["L"] * np.sqrt(PARAMS["nu"] * PARAMS["D"])
    n_frames = PARAMS["n_frames"]
    n_grid = PARAMS["n_grid"]
    xmin, xmax = PARAMS["xmin"], PARAMS["xmax"]
    duration = PARAMS["duration"]

    np.random.seed(seed)

    # Generate metaorder
    if use_noise:
        metaorder = M0 + M1 * np.random.randn(n_frames)
    else:
        metaorder = np.full(n_frames, M0, dtype=float)

    # Create simulation
    sim = Simulation.from_params(
        **PARAMS,
        lambd=lambd,
        metaorder=metaorder,
        alpha=alpha,
        seed=seed,
    )

    # Measurement frame indices
    measurement_frames = [int(f * (n_frames - 1)) for f in MEASUREMENT_FRACTIONS]

    dx = (xmax - xmin) / n_grid
    X = np.linspace(xmin, xmax, n_grid)
    dt = duration / n_frames

    profiles = {
        "ask": [],
        "bid": [],
        "mid_price": [],
        "spread": [],
        "frame_idx": [],
    }

    try:
        # Run frame by frame to capture intermediate states
        for n in range(n_frames):
            # Execute metaorder for this frame
            dq = metaorder[n] * dt

            sim.book.execute_metaorder(dq)
            for _ in range(sim.steps_per_frame):
                sim.book.stochastic_timestep()
                sim.book.update_price()
                sim.book.order_reaction()
                sim.book.update_price()

            # Record state
            sim.asks[n] = sim.book.best_ask
            sim.bids[n] = sim.book.best_bid
            sim.prices[n] = (sim.book.best_ask + sim.book.best_bid) / 2
            sim.spreads[n] = sim.book.best_ask - sim.book.best_bid

            # Record profiles at measurement times
            if n in measurement_frames:
                profiles["ask"].append(sim.book.get_ask_volumes().copy())
                profiles["bid"].append(sim.book.get_bid_volumes().copy())
                profiles["mid_price"].append(sim.prices[n])
                profiles["spread"].append(sim.spreads[n])
                profiles["frame_idx"].append(n)

        return {
            "alpha": alpha,
            "seed": seed,
            "use_noise": use_noise,
            "success": True,
            "prices": sim.prices.copy(),
            "spreads": sim.spreads.copy(),
            "profiles": profiles,
            "X": X,
            "dx": dx,
        }

    except ValueError as e:
        if "lacks" in str(e) and "liquidity" in str(e):
            return {
                "alpha": alpha,
                "seed": seed,
                "use_noise": use_noise,
                "success": False,
            }
        raise


def run_experiment(use_noise=False):
    """Run all simulations."""
    seeds = list(range(42, 42 + N_SEEDS))

    all_tasks = []
    for alpha in ALPHA_VALUES:
        for seed in seeds:
            all_tasks.append((alpha, seed, use_noise))

    noise_label = "noisy" if use_noise else "constant"
    print(f"Running {len(all_tasks)} simulations ({noise_label} metaorder)")
    print(f"  alpha values: {ALPHA_VALUES}")
    print(f"  seeds: {N_SEEDS}")

    n_workers = min(mp.cpu_count(), 8)
    print(f"Using {n_workers} workers")

    with mp.Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(run_single_with_profiles, all_tasks),
                total=len(all_tasks),
                desc="Simulations",
            )
        )

    return results


def process_results(results):
    """Process results into averaged density profiles."""
    data = {}

    for alpha in ALPHA_VALUES:
        matching = [r for r in results if r["alpha"] == alpha and r["success"]]

        if not matching:
            print(f"alpha={alpha}: no successful runs")
            continue

        X = matching[0]["X"]
        dx = matching[0]["dx"]
        n_measurements = len(matching[0]["profiles"]["frame_idx"])

        # For each measurement time, average profiles
        profile_data = []

        for m_idx in range(n_measurements):
            ask_vols = np.array([r["profiles"]["ask"][m_idx] for r in matching])
            bid_vols = np.array([r["profiles"]["bid"][m_idx] for r in matching])

            profile_data.append(
                {
                    "frame_idx": matching[0]["profiles"]["frame_idx"][m_idx],
                    "X": X,
                    "ask_mean": np.mean(ask_vols, axis=0),
                    "ask_std": np.std(ask_vols, axis=0),
                    "bid_mean": np.mean(bid_vols, axis=0),
                    "bid_std": np.std(bid_vols, axis=0),
                    "mid_prices": [r["profiles"]["mid_price"][m_idx] for r in matching],
                    "spreads": [r["profiles"]["spread"][m_idx] for r in matching],
                }
            )

        data[alpha] = {
            "profiles": profile_data,
            "n_success": len(matching),
            "dx": dx,
        }

    return data


def make_plots(data):
    """Create density profile plots."""
    successful_alphas = [a for a in ALPHA_VALUES if a in data]
    n_alpha = len(successful_alphas)
    if n_alpha == 0:
        print("No data to plot!")
        return

    # Figure 1: Final total density profiles (V-shape) for each alpha
    # Following Matthieu's style: plot ask_density + bid_density as total density
    fig1, axes = plt.subplots(1, n_alpha, figsize=(5 * n_alpha, 4))
    if n_alpha == 1:
        axes = [axes]

    for idx, alpha in enumerate(successful_alphas):
        ax = axes[idx]
        d = data[alpha]

        # Plot final profile (last measurement)
        final = d["profiles"][-1]
        X = final["X"]
        mid_price_avg = np.mean(final["mid_prices"])

        # Total density = ask + bid (V-shape), normalized
        total_density = final["ask_mean"] + final["bid_mean"]
        total_density_norm = total_density / (np.sum(total_density) * d["dx"] + 1e-10)

        # Plot in lab frame, mark where the price is
        ax.plot(X, total_density_norm, color="#465987", lw=2)
        ax.axvline(
            mid_price_avg,
            color="r",
            ls="--",
            lw=1.5,
            alpha=0.8,
            label=f"price={mid_price_avg:.1f}",
        )

        spread_avg = np.mean(final["spreads"])

        ax.set_xlabel("x (lab frame)")
        ax.set_ylabel("Normalized density")
        ax.set_title(f"α = {alpha} (spread={spread_avg:.2f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Final Density Profiles in Moving Frame")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / "exp2_final_profiles.png", dpi=150)
    print("Saved exp2_final_profiles.png")

    # Figure 2: Profile evolution over time (V-shape at different times)
    fig2, axes = plt.subplots(1, n_alpha, figsize=(6 * n_alpha, 5))
    if n_alpha == 1:
        axes = [axes]

    colors_time = plt.cm.viridis(np.linspace(0, 0.8, len(MEASUREMENT_FRACTIONS)))

    for col_idx, alpha in enumerate(successful_alphas):
        ax = axes[col_idx]
        d = data[alpha]

        for row_idx, profile in enumerate(d["profiles"]):
            X = profile["X"]
            mid_price_avg = np.mean(profile["mid_prices"])

            # Total density, normalized
            total_density = profile["ask_mean"] + profile["bid_mean"]
            total_density_norm = total_density / (
                np.sum(total_density) * d["dx"] + 1e-10
            )
            t_frac = MEASUREMENT_FRACTIONS[row_idx]

            ax.plot(
                X,
                total_density_norm,
                color=colors_time[row_idx],
                lw=2,
                alpha=0.8,
                label=f"t = {t_frac:.0%}T",
            )
            # Mark price at this time
            ax.axvline(
                mid_price_avg, color=colors_time[row_idx], ls="--", lw=1, alpha=0.4
            )

        ax.set_xlabel("x (lab frame)")
        ax.set_ylabel("Normalized density")
        ax.set_title(f"α = {alpha}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Density Profile Evolution")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / "exp2_profile_evolution.png", dpi=150)
    print("Saved exp2_profile_evolution.png")

    # Figure 3: Spread and mid-price statistics vs alpha
    fig3, axes = plt.subplots(1, 2, figsize=(10, 4))

    alphas_plot = []
    spreads_mean = []
    spreads_std = []
    mid_prices_mean = []
    mid_prices_std = []

    for alpha in ALPHA_VALUES:
        if alpha not in data:
            continue
        final = data[alpha]["profiles"][-1]
        alphas_plot.append(alpha)
        spreads_mean.append(np.mean(final["spreads"]))
        spreads_std.append(np.std(final["spreads"]))
        mid_prices_mean.append(np.mean(final["mid_prices"]))
        mid_prices_std.append(np.std(final["mid_prices"]))

    ax = axes[0]
    ax.errorbar(
        alphas_plot, spreads_mean, yerr=spreads_std, marker="o", markersize=8, capsize=4
    )
    ax.set_xlabel("α")
    ax.set_ylabel("Final Spread")
    ax.set_title("Spread vs α")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(
        alphas_plot,
        mid_prices_mean,
        yerr=mid_prices_std,
        marker="o",
        markersize=8,
        capsize=4,
    )
    ax.set_xlabel("α")
    ax.set_ylabel("Final Mid-Price")
    ax.set_title("Impact vs α")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / "exp2_spread_impact_vs_alpha.png", dpi=150)
    print("Saved exp2_spread_impact_vs_alpha.png")

    # Figure 4: All alphas overlaid on same plot (final profiles) - NORMALIZED
    fig4, ax = plt.subplots(figsize=(10, 6))

    colors_alpha = plt.cm.plasma(np.linspace(0.1, 0.9, len(successful_alphas)))

    for idx, alpha in enumerate(successful_alphas):
        d = data[alpha]
        final = d["profiles"][-1]
        X = final["X"]
        mid_price_avg = np.mean(final["mid_prices"])

        # Total density, normalized
        total_density = final["ask_mean"] + final["bid_mean"]
        total_density_norm = total_density / (np.sum(total_density) * d["dx"] + 1e-10)

        # Plot in lab frame
        ax.plot(
            X, total_density_norm, color=colors_alpha[idx], lw=2, label=f"α={alpha}"
        )
        # Mark the price location
        ax.axvline(mid_price_avg, color=colors_alpha[idx], ls="--", lw=1, alpha=0.5)

    ax.set_xlabel("x (lab frame)")
    ax.set_ylabel("Normalized density")
    ax.set_title("Normalized Density Profile Comparison Across α Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXPERIMENTS_PATH / "exp2_profile_comparison.png", dpi=150)
    print("Saved exp2_profile_comparison.png")


def main(rerun=False, use_noise=False):
    print("=" * 70)
    print("Experiment 2: Density Profile in Moving Frame")
    print("=" * 70)

    # Ensure output directory exists
    EXPERIMENTS_PATH.mkdir(parents=True, exist_ok=True)

    noise_suffix = "_noisy" if use_noise else ""
    results_file = EXPERIMENTS_PATH / f"exp2_density_results{noise_suffix}.pkl"

    if results_file.exists() and not rerun:
        print(f"Loading from {results_file}")
        with open(results_file, "rb") as f:
            saved = pickle.load(f)
        results = saved["results"]
        print(f"Loaded {len(results)} results")
    else:
        results = run_experiment(use_noise=use_noise)

        with open(results_file, "wb") as f:
            pickle.dump(
                {
                    "results": results,
                    "params": PARAMS,
                    "m0": M0,
                    "m1": M1 if use_noise else 0,
                    "alpha_values": ALPHA_VALUES,
                    "n_seeds": N_SEEDS,
                    "measurement_fractions": MEASUREMENT_FRACTIONS,
                },
                f,
            )
        print(f"Saved to {results_file}")

    # Process and analyze
    data = process_results(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for alpha in ALPHA_VALUES:
        if alpha not in data:
            print(f"alpha={alpha}: FAILED (all depleted)")
            continue
        final = data[alpha]["profiles"][-1]
        spread_mean = np.mean(final["spreads"])
        spread_std = np.std(final["spreads"])
        print(
            f"alpha={alpha}: spread = {spread_mean:.2f} ± {spread_std:.2f}, "
            f"n_success = {data[alpha]['n_success']}/{N_SEEDS}"
        )

    # Make plots
    make_plots(data)

    print("\nDone!")


if __name__ == "__main__":
    import sys

    rerun = "--rerun" in sys.argv
    use_noise = "--noise" in sys.argv
    main(rerun=rerun, use_noise=use_noise)
