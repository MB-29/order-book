.. _simulation:

****************
Simulation
**************** 


Model parameters
-----------------

One standard parameter setting is to take unitary time and space steps : chose ``xmax``, ``xmin`` and ``Nt`` to be integers and set

.. code-block:: python

   Nx = xmax - xmin
   T = n_steps * Nt
   
where ``n_steps`` is the number of evolution time steps (equal to 1 in the continuous model). 

.. code-block:: python

   parameters = simulation_args = {
    'model_type': 'discrete',

   # price interval
    'xmin': -100, 
    'xmax': 100,
    'Nx': 200,

   # time interval
    'Nt': 100, #number of time steps
    'T': 5000, #time horizon

   #dynamics constants
    'nu': 0,
    'L': 10,
    'D': 0.5,

    # meta-order
    'metaorder' : 10
   }

API
----

.. automodule:: llob.simulation
   :members:

