.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _odedata:

**************************************************
The Odedata Class and Dynamical Simulation Results
**************************************************

.. important::  In QuTiP 2, the results from all of the dynamics solvers are returned as Odedata objects.  This significantly simplifies the storage and saving of simulation data.  However, this change also results in the loss of backward compatibility with QuTiP version 1.x.  Therefore, please read this section to avoid running into any issues.

The Odedata class
=================
Before embarking on simulating the dynamics of quantum systems, we will first look at the data structure used for returning the simulation results to the user.  This object is a :func:`qutip.Odedata` class 


Saving and Loading Odedata objects
==================================
