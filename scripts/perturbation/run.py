"""
Run the perturbation experiment and save results.
"""

import pickle

from llob.perturbation import make_perturbation_mc
from config import (
    HURST, J, N_SAMPLES,
    REGIMES, SIM_PARAMS, RESULTS_DIR,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_results = {}
for name, rates in REGIMES.items():
    r0, r1 = rates["r0"], rates["r1"]
    m0, m1 = r0 * J, r1 * J
    print(f"\n=== {name}: m0/J = {r0}, m1/J = {r1} ===")
    mc = make_perturbation_mc(
        m0=m0,
        m1=m1,
        hurst=HURST,
        n_samples=N_SAMPLES,
        simulation_params=SIM_PARAMS,
    )
    mc.run()
    all_results[name] = mc.gather_results()
    all_results[name]["time"] = mc.simulation.time_interval

output_path = RESULTS_DIR / "perturbation_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"\nResults saved to {output_path}")
