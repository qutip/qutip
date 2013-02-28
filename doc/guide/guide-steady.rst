.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _steady:

*************************************
Solving for Steady-State Solutions
*************************************

.. _steady-intro:

Introduction
============

For open quantum systems with decay rates larger than the corresponding excitation rate, the system will tend toward a steady-state as :math:`t\rightarrow\infty` that satisfies the equation

:math:`\frac{\partial\rho_{ss}}{\partial t}=\mathcal{L}\rho_{ss}=0`.

For many these systems, solving for the asymptotic density matrix :math:`\rho_{ss}` can be achieved using an iterative method faster than master equation or Monte Carlo simulations.  In QuTiP, the steady-state solution for a system Hamiltonian or Liouvillian is given by :func:`qutip.steady.steadystate` or :func:`qutip.steady.steady`, respectively.  Both of these functions use a shifted inverse power method with a random initial state that finds the zero eigenvalue.  In general, it is best to use the :func:`qutip.steady.steadystate` function with a given Hamiltonian and list of collapse operators.  This function will automatically build the Louivillian for you and then call the :func:`qutip.steady.steady` function. 


.. _steady-usage:

Using the Steadystate Solver
=============================

A general call to the steady-state solver :func:`qutip.steady.steadystate` may be accomplished using the command::

>>> rho_ss = steadystate(H, c_op_list)

where ``H`` is a quantum object representing the system Hamiltonian, and ``c_op_list`` is a list of quantum objects for the system collapse operators.  The output, labeled as ``rho_ss``, is the steady-state solution for the system dynamics.  Although this method will produce the required solution in most situations, there are additional options that may be set if the algorithm does not converge properly.  These optional parameters may be used by calling the steady-state solver as::

>>> rho_ss = steadystate(H, c_op_list, maxiter, tol, method, use_umfpack, use_precond)

where ``maxiter`` is the maximum number of iterations performed by the solver, ``tol`` is the requested solution tolerance, and ``method`` indicates whether the linear equation solver uses a direct solver "solve" or an iterative stabilized bi-conjugate gradient "bicg" solution method.  In general, the direct solver is faster, but takes more memory than the iterative method.  The advantage of the iterative method is the memory efficiency of this routine, allowing for extremely large system sizes. The downside is that it takes much longer than the direct method, especially when the condition number of the Liouvillian matrix is large, as typically occurs.  To overcome this, the steady state solvers also include a preconditioner that attempts to normalize the condition number of the system.  This incomplete LU preconditioner is invoked by using the "use_precond=True" flag in combination with the iterative solver.  As a first try, it is recommended to begin with the direct solver before using the iterative ``bicg`` method.  More information on these options may be found in the :func:`qutip.steady.steadystate` section of the API.

This solver can also use a Liouvillian constructed from a Hamiltonian and collapse operators as the input variable when called using the :func:`qutip.steady.steady` function::

>>> rho_ss = steady(L)

where ``L`` is the Louvillian.  This function also takes the previously mentioned optional parameters.  We do however recommend using the :func:`qutip.steady.steadystate` function if you are using a standard Liouvillian as it will automatically build the system Liouvillian and call :func:`qutip.steady.steady` for you.

.. _steady-example:

Example: Harmonic Oscillator in Thermal Bath
============================================

A simple example of a system that reaches a steady state is a harmonic oscillator coupled to a thermal environment.  Below we consider a harmonic oscillator, initially in a :math:`\left|10\right>` number state, and weakly coupled to a thermal environment characterized by an average particle expectation value of :math:`\left<n\right>=2`.  We calculate the evolution via master equation and Monte Carlo methods, and see that they converge to the steady-state solution.  Here we choose to perform only a few Monte Carlo trajectories so we can distinguish this evolution from the master-equation solution.
    
.. literalinclude:: scripts/ex_steady.py

.. _steady-figure: 
.. figure:: guide-steady.png
   :align: center
   :width: 4in

