# Locally Linear Order Book

An implementation of the Locally Linear Order Book model as introduced by Donier *et al.* in the paper [A fully consistent, minimal model for non-linear market impact ](https://https://arxiv.org/abs/1412.0141).

Our code aims at solving both the coarse-grained, *discrete* model, and the *continuous model* where the variable solved for is the algebraic order density.

## Structure of the code

Depending on the model, a locally linear order book is represented by an instance of either class `DiscreteBook` `discrete_book.py` or class `ContinuousBook` `continuous_book.py`.

A simulation can be created with class `Simulation` from `simulation.py` by inputting the model type and various simulation parameters.

Run the simulation with method `simulation.run()`. Display a `matplotlib` animation by setting argument `animation=True` and save it by setting `save=True`.

## Discrete Order Book

An instance of `DiscreteBook` has two `orders` attributes which are instances of class `LimitOrders` and represent ASK and BUY orders respectively.
Class `LinearDiscreteBook` inherits from `DiscreteBook` and may be initialized with latent liquidity constant *L* instead of order deposition and cancellation constants.

## Continuous Order Book

This part of the code aims at solving order density reaction-diffusion equation under the assumptions of infinite memory.

Numerical scheme functions for diffusion equation are imported from `diffusion_schemes.py`.

## Metaorder
A metaorder is defined by a dictionary of parameters 
```python
metaorder_args ={
    'metaorder' : [m0],
    'm0' : m0,
    'n_start' : n_start,
    'n_end' : n_end
}
```

## Run a simulation
Set parameters in `run.py` then run

```bash
cd Python
python run.py
```

## Output

![Discrete book animation](demo/histogram.gif)
![Price impact vs theory](demo/vs_theory.png)

## Requirements
* Python 3
* Modules numpy and matplotlib
