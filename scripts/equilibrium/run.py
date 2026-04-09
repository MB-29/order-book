"""
Run the unbiased equilibrium experiment and save results.
"""

import pickle

from llob.equilibrium import make_unbiased_mc
from scripts.equilibrium.config import (
    HURST, N_SAMPLES, M1_VALUES, J,
    SIM_PARAMS, PROFILE_INDICES, RESULTS_DIR,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_results = {}
for m1 in M1_VALUES:
    r1 = m1 / J
    print(f"\n=== m1/J = {r1:.1f} (m1 = {m1:.1f}) ===")
    mc = make_unbiased_mc(
        m1=m1,
        hurst=HURST,
        n_samples=N_SAMPLES,
        simulation_params=SIM_PARAMS,
        profile_indices=PROFILE_INDICES,
    )
    mc.run()
    all_results[m1] = mc.gather_results()
    all_results[m1]["time"] = mc.simulation.time_interval
    all_results[m1]["X"] = mc.simulation.book.X

output_path = RESULTS_DIR / "equilibrium_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"\nResults saved to {output_path}")
