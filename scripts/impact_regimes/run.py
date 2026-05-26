"""
Run the impact-regime sweep and save all regime results into a single pickle.

A regime that raises a liquidity error is reported and skipped; subsequent
regimes still run. The error message is preserved in the pickle so the
plotting script can flag it.
"""

import argparse
import pickle
import traceback

from config import (
    HURST, J, N_SAMPLES, REGIMES, RESULTS_DIR, RESULTS_FILE, SIM_PARAMS,
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

all_results: dict = {}
failures: dict[str, str] = {}

for name, rates in REGIMES.items():
    r0, r1 = rates["r0"], rates["r1"]
    m0, m1 = r0 * J, r1 * J
    print(f"\n=== {name}: m0/J = {r0}, m1/J = {r1}  (m0={m0:g}, m1={m1:g}) ===")

    mc = MonteCarlo.from_params(
        N_samples=n_samples,
        noise_args={"m0": m0, "m1": m1, "hurst": HURST},
        simulation_args=SIM_PARAMS,
    )

    try:
        mc.run()
    except (ValueError, IndexError) as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"  !! LIQUIDITY ERROR: {msg}")
        traceback.print_exc(limit=2)
        failures[name] = msg
        continue

    result = mc.gather_results()
    result["time"] = mc.simulation.time
    result["r0"] = r0
    result["r1"] = r1
    all_results[name] = result

output = {"regimes": all_results, "failures": failures, "config": {"hurst": HURST, "J": J}}
with open(RESULTS_FILE, "wb") as f:
    pickle.dump(output, f)

print(f"\n=== Saved {len(all_results)}/{len(REGIMES)} regimes to {RESULTS_FILE} ===")
if failures:
    print("Failed regimes:")
    for name, msg in failures.items():
        print(f"  - {name}: {msg}")
