.. LLOB documentation master file, created by
   sphinx-quickstart on Fri Jun  5 14:22:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
LLOB
=====================================

Documentation for the LLOB repository
=====================================

The code is divided into different classes. The order book is modelled by an instance of one of the Order Book classes : :ref:`discrete` and :ref:`continuous`.

The evolution of the book, of its input and of its output over time is represented by an instance of the class :ref:`simulation`. It stores the values of various variables over time.

A :ref:`monte-carlo` computes the average of a certain number of :ref:`simulation` instances with random inputs, which are refered to as samples.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   api


Run and display a sample simulation
-----------------------------------

See the documentation of class :ref:`simulation` for more information about the parameters.

.. code-block:: python

   import matplotlib.pyplot as plt

   from simulation import Simulation

   # Simulation parameters
   parameters = {
    'xmin': -200,
    'xmax': 200,
    'Nx': 200,
    'nu': 0,
    'L': 10,
    'D': 0.5,
    'Nt': 100,
    'T': 5000,
    'model_type': 'discrete',
    'metaorder' : 10
   }
   
   # Run and display
   fig = plt.figure()
   simulation = Simulation(**parameters)
   simulation.run(simulation=True, fig=fig, save=True)
   plt.show()

	

.. centered:: **Output**

.. image:: _static/execution.gif 
   :align: center


Run a Monte Carlo simulation
----------------------------

.. code-block:: python

   #noise parameters
   N_samples = 1000
   noise_args = {
       'm0': 10,
       'm1': 20,
       'hurst': 0.75
   }

   #Run
   noisy_simulation = MonteCarlo(N_samples, noise_args, parameters)
   noisy_simulation.run()

   # Output dictionary
   output = noisy_simulation.gather_results()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
