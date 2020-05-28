import pandas as pd
import numpy as np

from monte_carlo import MonteCarlo
from simulation import standard_parameters

n_simulation = 2
n_sigma = 1
n_hurst = 1
N_samples = 2
participation_rate = 1e5
parameters = standard_parameters(participation_rate, 'discrete')
m0 = parameters['metaorder'][0]

def parallelized_function(params):
    noise_args = params.to_dict()
    # Add m0 here because it is constant
    noise_args['m0'] = m0
    print(f'noise_args : {noise_args}')
    montecarlo = MonteCarlo(N_samples, noise_args, parameters)
    montecarlo.run()
    measures = montecarlo.gather_results()
    return measures


sigma_list = np.logspace(-2, 0, num=n_sigma)
hurst_list = np.linspace(0.55, 0.85, num=n_hurst)

index = pd.MultiIndex.from_product(
    [sigma_list, hurst_list, range(n_simulation)], names=["sigma", "hurst", "simulation"])


config_params = pd.DataFrame(index=index).reset_index().drop('simulation', axis=1)
results = parallelized_function(config_params.iloc[1, :])
# job = ibase.parallelize(main_func=parallelized_function, config=config_params)
# ## This works at CFM, config_params is a pandas dataframe where each row is a set of parameters to run paralllelized_function
# ## The same philosophy should apply at Ladhyx
# ​
# results = job.results()
