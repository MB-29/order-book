"""
Run the perturbation experiment and save results.
"""

import argparse
import pickle

from config import (
    HURST, J, N_SAMPLES, REGIMES, RESULTS_DIR, SIM_PARAMS,
)

from llob import MonteCarlo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--n-samples", type=int, default=N_SAMPLES,
    help=f"Monte-Carlo sample count (default from config.py: {N_SAMPLES}).",
)
args = parser.parse_args()
n_samples = args.n_samples

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_results = {}
for name, rates in REGIMES.items():
    r0, r1 = rates["r0"], rates["r1"]
    m0, m1 = r0 * J, r1 * J
    print(f"\n=== {name}: m0/J = {r0}, m1/J = {r1} ===")
    mc = MonteCarlo.from_params(
        N_samples=n_samples,
        noise_args={"m0": m0, "m1": m1, "hurst": HURST},
        simulation_args=SIM_PARAMS,
    )
    mc.run()
    all_results[name] = mc.gather_results()
    all_results[name]["time"] = mc.simulation.time

output_path = RESULTS_DIR / "perturbation_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"\nResults saved to {output_path}")
