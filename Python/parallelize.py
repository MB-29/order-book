import pandas as pd
import numpy as np

from monte_carlo import MonteCarlo

n_sigma = 5
n_hurst = 5
N_samples = 2
m0 = 10

args = {
    "T": 1000,
    "Nt": 100,
    "price_formula": "best_ask",
    "model_type": "continuous",
    "book_args": {
        "Nx": 10001,
        "dt": 10,
        "xmin": -10,
        "xmax": 150,
        "L": 1,
        "D": 0
    },
    "metaorder_args": {
        "metaorder": [
            10
        ],
        "m0": m0,
        "n_start": 0,
        "n_end": 100
    }
}

def parallelized_function(params):
    noise_args = params.to_dict()
    noise_args['m0'] = m0
    print(f'noise_args : {noise_args}')
    montecarlo = MonteCarlo(N_samples, noise_args, args)
    montecarlo.run()
    measures = montecarlo.gather_results()
    return measures


sigma_list = np.logspace(-2, 0, num=n_sigma)
hurst_list = np.linspace(0.55, 0.85, num=n_hurst)

index = pd.MultiIndex.from_product(
    [sigma_list, hurst_list], names=["sigma", "hurst"])

config_params = pd.DataFrame(index=index).reset_index()
results = parallelized_function(config_params.iloc[3, :])
# config_params = pd.Dataframe()
# job = ibase.parallelize(main_func=parallelized_function, config=config_params)
# ## This works at CFM, config_params is a pandas dataframe where each row is a set of parameters to run paralllelized_function
# ## The same philosophy should apply at Ladhyx
# ​
# results = job.results()
