.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _correlation:

******************************
Two-time correlation functions
******************************

With the QuTiP time-evolution functions (for example :func:`qutip.mesolve` and :func:`qutip.mcsolve`), a state vector or density matrix can be evolved from an initial state at :math:`t_0` to an arbitrary time :math:`t`, :math:`\rho(t)=V(t, t_0)\left\{\rho(t_0)\right\}`, where :math:`V(t, t_0)` is the propagator defined by the equation of motion. The resulting density matrix can then be used to evaluate the expectation values of arbitrary combinations of *same-time* operators.

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

QuTiP provides a family of functions that assists in the process of calculating two-time correlation functions. The available functions and their usage is show in the table below. Each of these functions can use one of the following evolution solvers: Master-equation, Exponential series and the Monte-Carlo. The choice of solver is defined by the optional argument ``solver``. 

.. tabularcolumns:: | p{4cm} | L |

+----------------------------------------------+--------------------------------------------------+
| QuTiP function                               | Correlation function                             |
+==============================================+==================================================+
| :func:`qutip.correlation.correlation` or     | :math:`\left<A(t+\tau)B(t)\right>` or            |
| :func:`qutip.correlation.correlation_2op_2t` | :math:`\left<A(t)B(t+\tau)\right>`.              |
+----------------------------------------------+--------------------------------------------------+
| :func:`qutip.correlation.correlation_ss` or  | :math:`\left<A(\tau)B(0)\right>` or              |
| :func:`qutip.correlation.correlation_2op_1t` | :math:`\left<A(0)B(\tau)\right>`.                |
+----------------------------------------------+--------------------------------------------------+
| :func:`qutip.correlation.correlation_4op_1t` | :math:`\left<A(0)B(\tau)C(\tau)D(0)\right>`.     |
+----------------------------------------------+--------------------------------------------------+
| :func:`qutip.correlation.correlation_4op_2t` | :math:`\left<A(t)B(t+\tau)C(t+\tau)D(t)\right>`. |
+----------------------------------------------+--------------------------------------------------+

The most common use-case is to calculate correlation functions of the kind :math:`\left<A(\tau)B(0)\right>`, in which case we use the correlation function solvers that start from the steady state, e.g., the :func:`qutip.correlation.correlation_2op_1t` function. These correlation function sovlers return a vector or matrix (in general complex) with the correlations as a function of the delays times. 

.. _correlation-steady:

Steadystate correlation function
================================

The following code demonstrates how to calculate the :math:`\left<x(t)x(0)\right>` correlation for a leaky cavity with three different relaxation rates.

.. plot:: guide/scripts/correlation_ex1.py
   :width: 4.0in
   :include-source:	

.. _correlation-nosteady:


Emission spectrum
=================

Given a correlation function :math:`\left<A(\tau)B(0)\right>` we can define the corresponding power spectrum as

.. math::

    S(\omega) = \int_{-\infty}^{\infty} \left<A(\tau)B(0)\right> e^{-i\omega\tau} d\tau.

In QuTiP, we can calculate :math:`S(\omega)` using either :func:`qutip.correlation.spectrum_ss`, which first calculates the correlation function using the :func:`qutip.essolve.essolve` solver and then performs the Fourier transform semi-analytically, or we can use the function :func:`qutip.correlation.spectrum_correlation_fft` to numerically calculate the Fourier transform of a given correlation data using FFT.  

The following example demonstrates how these two functions can be used to obtain the emission power spectrum.

.. plot:: guide/scripts/spectrum_ex1.py
   :width: 4.0in
   :include-source:	

.. _correlation-spectrum:


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

For convenience, the steps for calculating the first-order coherence function have been collected in the function :func:`qutip.correlation.coherence_function_g1`.

Example: second-order optical coherence function
------------------------------------------------

The second-order optical coherence function, with time-delay :math:`\tau`, is defined as

.. math::

    \displaystyle g^{(2)}(\tau) = \frac{\langle a^\dagger(0)a^\dagger(\tau)a(\tau)a(0)\rangle}{\langle a^\dagger(0)a(0)\rangle^2}

For a coherent state :math:`g^{(2)}(\tau) = 1`, for a thermal state :math:`g^{(2)}(\tau=0) = 2` and it decreases as a function of time (bunched photons, they tend to appear together), and for a Fock state with :math:`n` photons :math:`g^{(2)}(\tau = 0) = n(n - 1)/n^2 < 1` and it increases with time (anti-bunched photons, more likely to arrive separated in time).  

To calculate this type of correlation function with QuTiP, we can use :func:`qutip.correlation.correlation_4op_1t`, which computes a correlation function on the form :math:`\left<A(0)B(\tau)C(\tau)D(0)\right>` (four operators, one delay-time vector).

The following code calculates and plots :math:`g^{(2)}(\tau)` as a function of :math:`\tau` for a coherent, thermal and fock state.

.. plot:: guide/scripts/correlation_ex4.py
   :width: 4.0in
   :include-source:

For convenience, the steps for calculating the second-order coherence function have been collected in the function :func:`qutip.correlation.coherence_function_g2`.


