.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _odedata:

**************************************************
The Odedata Class and Dynamical Simulation Results
**************************************************

.. important::  In QuTiP 2, the results from all of the dynamics solvers are returned as Odedata objects.  This significantly simplifies the storage and saving of simulation data.  However, this change also results in the loss of backward compatibility with QuTiP version 1.x.  Therefore, please read this section to avoid running into any issues.

The Odedata Class
=================
Before embarking on simulating the dynamics of quantum systems, we will first look at the data structure used for returning the simulation results to the user.  This object is a :func:`qutip.Odedata` class that stores all the crucial data needed for analyzing and plotting the results of a simulation.  Like the :func:`qutip.Qobj` class, the ``Odedata`` class has a collection of properties for storing information.  However, in contrast to the Qobj class, this structure contains no methods, and is therefore nothing but a container object.  A generic Odedata object ``odedata`` contains the following properties for storing simulation data:

.. tabularcolumns:: | p{2cm} | p{3cm} |

+------------------------+-----------------------------------------------------------------------+
| Property               | Description                                                           |
+========================+=======================================================================+
| odedata.solver         | String indicating which solver was used to generate the data.         |
+------------------------+-----------------------------------------------------------------------+
| odedata.times          | List/array of times at which simulation data is calculated.           |
+------------------------+-----------------------------------------------------------------------+
| odedata.expect         | List/array of expectation values, if requested.                       |
+------------------------+-----------------------------------------------------------------------+
| odedata.states         | List/array of state vectors /density matrices calcuated at ``times``. |
+------------------------+-----------------------------------------------------------------------+
| odedata.num_expect     | The number of expectation value operators in the simulation.          |
+------------------------+-----------------------------------------------------------------------+
| odedata.num_collapse   | The number of collapse operators in the simulation.                   |
+------------------------+-----------------------------------------------------------------------+
| odedata.ntraj          | Number of monte-carlo trajectories run (if using mcsolve).            |
+------------------------+-----------------------------------------------------------------------+





Accessing Odedata Data
======================



Saving and Loading Odedata Objects
==================================
