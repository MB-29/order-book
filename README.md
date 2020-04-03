# ReadMe 

## Linearized Latent Order book simulations
Parent page of all the LLOB projects at  [the EconophysiX page](https://econophysix-confluence.atlassian.net/wiki/spaces/RES/pages/43679790/LLOBs).

### Description of the code

**UPLOADING NOTEBOOKS HERE IS STRICTLY FORBIDDEN UNLESS PREVIOUSLY DISCUSSED**

The code in this page contains scripts for simulating the LLOB.

The code for all types of simulations should be stored in a `.py` file defining the objects, such as classes with:

```python
class Simulation(object):

	def __init__(params):
		...... 


    def do_stuff():
		....
        
def helper_function(...):
    ...
```


So that simple simulation test and execution scripts can be written in a **separate file**, i.e.

```python
from object_definitions import Simulation, helper_function

def test_function(*params):
	sim = Simulation(*params)
	sim.do_stuff()
	measures = sim.gather_measures()
	return measures

params = ....
test = test_function(*params)
```
and then the results can be analyzed.

Test results, as well as parameters, can be stored on `pkl` files with the Pickle module.

