"""
Run the unbiased equilibrium experiment and save results.
"""

import argparse
import pickle

from llob import MonteCarlo
from scripts.equilibrium.config import (
    HURST, J, M1_VALUES, N_SAMPLES, RESULTS_DIR, SIM_PARAMS,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--n-samples", type=int, default=N_SAMPLES,
    help=f"Monte-Carlo sample count (default from config.py: {N_SAMPLES}).",
)
args = parser.parse_args()
n_samples = args.n_samples

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_results = {}
for m1 in M1_VALUES:
    print(f"\n=== m1/J = {m1 / J:.1f} (m1 = {m1:.1f}) ===")
    mc = MonteCarlo.from_params(
        N_samples=n_samples,
        noise_args={"m0": 0.0, "m1": m1, "hurst": HURST},
        simulation_args=SIM_PARAMS,
    )
    mc.run()
    all_results[m1] = mc.gather_results()
    all_results[m1]["time"] = mc.simulation.time
    all_results[m1]["X"] = mc.simulation.book.X

output_path = RESULTS_DIR / "equilibrium_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"\nResults saved to {output_path}")
