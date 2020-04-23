# ReadMe 

## Linearized Latent Order book simulations
Parent page of all the LLOB projects at  [the EconophysiX page](https://econophysix-confluence.atlassian.net/wiki/spaces/RES/pages/43679790/LLOBs).

**UPLOADING NOTEBOOKS HERE IS STRICTLY FORBIDDEN UNLESS PREVIOUSLY DISCUSSED**

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