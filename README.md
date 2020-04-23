# ReadMe 

## Linearized Latent Order book simulations
Parent page of all the LLOB projects at  [the EconophysiX page](https://econophysix-confluence.atlassian.net/wiki/spaces/RES/pages/43679790/LLOBs).

**UPLOADING NOTEBOOKS HERE IS STRICTLY FORBIDDEN UNLESS PREVIOUSLY DISCUSSED**
<!-- 
### Description of the code


The code in this page contains scripts for simulating the LLOB.

The code for all types of simulations should be stored in a `.py` file defining the objects, such as classes with:

```python
class Simulation(object):

    def __init__(params):
        ...

    def do_stuff():
        ...


def helper_function():
    ...
```


So that simple simulation test and execution scripts can be written in a **separate file**, i.e.

```python
from object_definitions import Simulation, helper_function


def test_function(*params):
    sim = Simulation(*params)
    sim.do_stuff()
    helper_function()
    measures = sim.gather_measures()
    return measures


params = ...
test = test_function(*params)
```
and then the results can be analyzed.

Test results, as well as parameters, can be stored on `pkl` files with the Pickle module.
 -->
# Locally Linear Order Book equation

This code aims at solving order density reaction-diffusion equation under the assumptions of infinite memory.

## Requirements
* Python 3
* Modules numpy and matplotlib

## Structure of the code

An order book is represented by an instance of class `OrderBook` from `order_book.py`.

A simulation is represented by an instance of class `Simulation` from `simulation.py`.

Numerical scheme functions for diffusion equation are imported from `diffusion_schemes.py`.

A script setting parameters, running and plotting a simulation is provided in `run_simulation.py`.

Plots can be performed with `simulation.plot_...` methods.

Display an animation of the order book and of the price evolution using `simulation.run(animation=True)`. 


## Run a simulation
Set parameters in `run_simulation.py` then run

```bash
cd Python
python run_simulation.py
```

## Output

![Density profile](demo/density.png)
![Price evolution](demo/price.png)
![Price evolution](demo/price_symlog.png)