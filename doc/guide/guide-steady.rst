.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _steady:

*************************************
Solving for Steady-State Solutions
*************************************

For open quantum systems with decay rates larger than the corresponding excitation rate, the system will tend toward a steady-state as :math:`t\rightarrow\infty`.  For many these systems, solving for the asymptotic state vector can be achieved using an iterative method faster than master equation or monte-carlo methods.  In QuTiP, the steady-state solution for a system Hamiltonian or Louivillian is given by :func:`qutip.steady.steadystate` or :func:`qutip.steady.steady`, respectively.  Both of these functions use an inverse power method with a random initial state.  (Details of these methods may be found in any iterative linear algebra text.)  In general, it is best to use the :func:`qutip.steady.steadystate` function with a given Hamiltonian and list of collapse operators.  This function will automatically build the Louivillian for you and then call the :func:`qutip.steady.steady` function. 

Usage
=====

A general call to the steady-state solver :func:`qutip.steady.steadystate` may be accomplished using the command::

>>> ss_ket=steadystate(H,c_op_list)

where ``H`` is a quantum object representing the system Hamiltonian, and ``c_op_list`` is a list of quantum objects for the system collapse operators.  The output, labeled as ``ss_ket``, is the steady-state solution for the system dynamics.  Note that the output here is a ket-vector and not a density matrix.  Although this method will produce the required solution in most situations, there are additional options that may be set if the algorithm does not converge properly.  These optional parameters may be used by calling the steady-state solver as::

>>> ss_ket=steadystate(H,c_op_list,maxiter,tol,method)

where ``maxiter`` is the maximum number of iterations performed by the solver, ``tol`` is the requested solution tolerance, and ``method`` indicates whether the linear equation solver uses a direct or iterative solution method.  More information on these options may be found in the :func:`qutip.steady.steadystate` section of the API.

This solver can also use a Louvillian constructed from a Hamiltonian and collapse operators as the input variable when called using the :func:`qutip.steady.steady` function::

>>> ss_ket=steady(L)

where ``L`` is the Louvillian.  This function also takes the previously mentioned optional parameters.  We do however recommend using the :func:`qutip.steady.steadystate` function as it will automatically build the system Louvillian and call :func:`qutip.steady.steady` for you.

Example
=======

A simple example of a system that reaches a steady-state is a harmonic oscillator coupled to a thermal environment.  Below we consider a harmonic oscillator, initially in a :math:`\left|10\right>` number state, and weakly coupled to a thermal environment characterized by an average particle expectation value of :math:`\left< n\right>=2`.  We calculate the evolution via master equation and monte-carlo methods, and see that they converge to the steady-state solution.  Here we choose to perform only a few Monte Carlo trajectories so we can distinguish this evolution from the master equation solution.
    
.. literalinclude:: scripts/ex_steady.py

.. _steady-figure: 
.. figure:: guide-steady.png
   :align: center
   :width: 6in

