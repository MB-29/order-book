#README 
## Firm network dynamics
Summary of the project at [the EconophysiX page](https://econophysix-confluence.atlassian.net/wiki/spaces/RES/pages/163184641/Networks)

### Description of the code

The code in this page contains scripts for simulating the network firm dynamics.

Some code can only run at CFM because of the ibase package that interfaces with the cluster. 

Ideally, to work with ibase, a simulation should be structured as a class, i.e.

```python
class Simulation(object):

	def __init__(params):
		...... 


	def do_stuff():
		....
```


So that simple simulation scripts can be written for some cluster in a **separate file**, i.e.

```python
import Simulation

def parallelized_function(params):
	sim = Simulation(params)
	sim.do_stuff()
	measures = sim.gather_measures()
	return measures
	
config_params = ...
job = ibase.parallelize(main_func = parallelized_function, config= config_params)
## This works at CFM, config_params is a pandas dataframe where each row is a set of parameters to run paralllelized_function
## The same philosophy should apply at Ladhyx

results = job.results()
``` 
and then the results can be analyzed. For an example check out the parallel_run notebook (**only added here as an example, notebooks should not be on git**). The result of that parallel run are stored in a .npy file (if the results are something different than a numpy array, for instance a ```dict``` then they can be stored with ```pickle```). 


Writing code in this format will allow for **separate** parallelization scripts to be written, whether they should run at Ladhyx or at CFM. 
