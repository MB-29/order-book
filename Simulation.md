# Run a simulation

## Requirements
* Python 3
* Modules numpy and matplotlib

## Structure of the code

An order book is represented by an instance of class `OrderBook` from `order_book.py`.
Numerical scheme functions for diffusion equation are imported from `diffusion_schemes.py`.
A script setting parameters, running and plotting a simulation is provided in `simulation.py`.


## Run a simulation
Set parameters in `run.py` then run

```bash
cd Python
python simulation.py
```
## Output

![Density profile](demo/density.png)
![Price evolution](demo/price.png)