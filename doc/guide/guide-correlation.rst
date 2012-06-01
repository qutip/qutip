.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _correlation:

******************************
Two-time correlation functions
******************************

With the QuTiP time-evolution functions (for example :func:`qutip.odesolve.mesolve` and :func:`qutip.mcsolve`), a state vector or density matrix can be evolved from an initial state at :math:`t_0` to an arbitrary time :math:`t`, :math:`\rho(t)=V(t, t_0)\left\{\rho(t_0)\right\}`, where :math:`V(t, t_0)` is the propagator defined by the equation of motion. The resulting density matrix can then be used to evaluate the expectation values of arbitrary combinations of *same-time* operators.

To calculate *two-time* correlation functions on the form :math:`\left<A(t+\tau)B(t)\right>`, we can use the quantum regression theorem [see, e.g., Gardineer and Zoller, *Quantum Noise*, Springer, 2004] to write

.. math::

    \left<A(t+\tau)B(t)\right> = {\rm Tr}\left[A V(t+\tau, t)\left\{B\rho(t)\right\}\right]
                               = {\rm Tr}\left[A V(t+\tau, t)\left\{BV(t, 0)\left\{\rho(0)\right\}\right\}\right]

We therefore first calculate :math:`\rho(t)=V(t, 0)\left\{\rho(0)\right\}` using one of the QuTiP evolution solvers with :math:`\rho(0)` as initial state, and then again use the same solver to calculate :math:`V(t+\tau, t)\left\{B\rho(t)\right\}` using :math:`B\rho(t)` as initial state.

Note that if the intial state is the steady state, then :math:`\rho(t)=V(t, 0)\left\{\rho_{\rm ss}\right\}=\rho_{\rm ss}` and 

.. math::

    \left<A(t+\tau)B(t)\right> = {\rm Tr}\left[A V(t+\tau, t)\left\{B\rho_{\rm ss}\right\}\right] 
                               = {\rm Tr}\left[A V(\tau, 0)\left\{B\rho_{\rm ss}\right\}\right] = \left<A(\tau)B(0)\right>,
    
which is independent of :math:`t`, so that we only have one time coordinate :math:`\tau`.

In QuTiP, there are two functions that assists in the process of calculating two-time correlation functions, :func:`qutip.correlation.correlation` and :func:`qutip.correlation.correlation_ss` (for steadystate correlations). Both these functions can use one of the following evolution solvers: Master-equation, Exponential series and the Monte-Carlo. The choice of solver is defined by the optional last argument `solver`. The following table describes the usage of each function:

.. tabularcolumns:: | p{4cm} | L |

+----------------------------------------------+-----------------------------------------+
| Function                                     | Usage                                   |
+==============================================+=========================================+
| correlation_ss                               | Calculates the steady state correlation |
|                                              | :math:`\left<A(\tau)B(0)\right>`,       |
|                                              | using the either the master equation,   |
|                                              | the exponential series, or the          |
|                                              | Monte Carlo solver.                     |
+----------------------------------------------+-----------------------------------------+
| correlation                                  | Calculates the correlation function     |
|                                              | :math:`\left<A(t_1+t_2)B(t_1)\right>`,  |
|                                              | using the either the master eqaution,   |
|                                              | the exponential series, or the          |
|                                              | Monte Carlo solver.                     |
+----------------------------------------------+-----------------------------------------+

The most common use-case is to calculate correlation functions of the kind :math:`\left<A(\tau)B(0)\right>`, in which case we use the correlation function solvers that start from the steady state, e.g., the :func:`qutip.correlation.correlation_ss` function. These functions return a vector (in general complex) with the correlations between the operators as a function of the difference time. 

.. _correlation-steady:

Steadystate correlation function
================================

The following code demonstrates how to calculate the :math:`\left<x(t)x(0)\right>` correlation for a leaky cavity with three different relaxation rates.

.. plot:: guide/scripts/correlation_ex1.py
   :width: 4.0in
   :include-source:	

.. _correlation-nosteady:

Non-steadystate correlation function
====================================
    
More generally, we can also calculate correlation functions of the kind :math:`\left<A(t_1+t_2)B(t_1)\right>`, i.e., the correlation function of a system that is not in its steadystate. In QuTiP, we can evoluate such correlation functions using the function :func:`qutip.correlation.correlation`. The default behavior of this function is to return a matrix with the correlations as a function of the two time coordinates (:math:`t_1` and :math:`t_2`).

.. plot:: guide/scripts/correlation_ex2.py
   :width: 4.0in
   :include-source:

However, in some cases we might be interested in the correlation functions on the form :math:`\left<A(t_1+t_2)B(t_1)\right>`, but only as a function of time coordinate :math:`t_2`. In this case we can also use the :func:`qutip.correlation.correlation` function, if we pass the density matrix at time :math:`t_1` as second argument, and `None` as third argument. The :func:`qutip.correlation.correlation` function then returns a vector with the correlation values corresponding to the times in `taulist` (the fourth argument).

Example: first-order optical coherence function
-----------------------------------------------

This example demonstrates how to calculate a correlation function on the form :math:`\left<A(\tau)B(0)\right>` for a non-steady initial state. Consider an oscillator that is interacting with a thermal environment. If the oscillator initially is in a coherent state, it will gradually decay to a thermal (incoherent) state. The amount of coherence can be quantified using the first-order optical coherence function :math:`g^{(1)}(\tau) = \frac{\left<a^\dagger(\tau)a(0)\right>}{\sqrt{\left<a^\dagger(\tau)a(\tau)\right>\left<a^\dagger(0)a(0)\right>}}`. For a coherent state :math:`|g^{(1)}(\tau)| = 1`, and for a completely incoherent (thermal) state :math:`g^{(1)}(\tau) = 0`. The following code calculates and plots :math:`g^{(1)}(\tau)` as a function of :math:`\tau`.

.. plot:: guide/scripts/correlation_ex3.py
   :width: 4.0in
   :include-source:


