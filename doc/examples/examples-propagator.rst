.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

Using the propagator to find the steady state of a driven system
================================================================
  
In this example we consider a strongly driven two-level system where the driving field couples to the sigma-Z operator. The system is subject to repeated Landau-Zener-like transitions.

In the following code we evolve the system for a few driving periods and plot the results, to get an idea of how the state of the two-level system changes at the avoided-level crossing points (where the Sigma-Z coefficient in the Hamiltonian is zero). 

Next, we use the :func:`qutip.propagator` function to find the propagator for the system for one driving period, and then we use the :func:`qutip.propagator_steadystate` function to find the pseudo steady state density matrix that follows from infinitely many applications of the one-period propagotor.
    
.. include:: examples-propagator.py
    :literal:    

`Download example <http://qutip.googlecode.com/svn/doc/examples/examples-propagator.py>`_
      
.. figure:: examples-propagator.png
    :align: center
    


