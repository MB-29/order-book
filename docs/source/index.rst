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

The evolution of the book and of the input and the output over time is represented by an instance of the class :ref:`simulation`.

A :ref:`monte-carlo` computes the average of a certain number of :ref:`simulation` instances with random inputs, which are refered to as samples.

Run a simulation
----------------

.. code-block:: python

   from simulation import Simulation

   simulation = Simulation(*parameters)
   simulation.run()

.. image:: _static/execution.gif 
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
