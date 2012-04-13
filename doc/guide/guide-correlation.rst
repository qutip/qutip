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
| :func:`qutip.correlation.correlation_ss`     | Calculates the steady state correlation |
|                                              | :math:`\left<A(\tau)B(0)\right>`,       |
|                                              | using the either the master equation,   |
|                                              | the exponential series, or the          |
|                                              | Monte Carlo solver.                     |
+----------------------------------------------+-----------------------------------------+
| :func:`qutip.correlation.correlation`        | Calculates the correlation function     |
|                                              | :math:`\left<A(t_1+t_2)B(t_1)\right>`,  |
|                                              | using the either the master eqaution,   |
|                                              | the exponential series, or the          |
|                                              | Monte Carlo solver.                     |
+----------------------------------------------+-----------------------------------------+

The most common use-case is to calculate correlation functions of the kind :math:`\left<A(\tau)B(0)\right>`, in which case we use the correlation function solvers that start from the steady state, e.g., the :func:`qutip.correlation.correlation_ss` function. These functions return a vector (in general complex) with the correlations between the operators as a function of the difference time. 

.. _correlation-steady:

Steadystate correlation function
================================

The following code demonstrates how to calculate the :math:`\left<x(t)x(0)\right>` correlation for a leaky cavity with three different relaxation rates::

    >>> tlist = linspace(0,10.0,200)
    >>> a  = destroy(10)
    >>> x  = a.dag() + a
    >>> H  = a.dag()*a
    >>>  
    >>> corr1 = correlation_ss(H, tlist, [sqrt(0.5)*a], x, x)
    >>> corr2 = correlation_ss(H, tlist, [sqrt(1.0)*a], x, x)
    >>> corr3 = correlation_ss(H, tlist, [sqrt(2.0)*a], x, x)
    >>>  
    >>> from pylab import *
    >>> plot(tlist, real(corr1), tlist, real(corr2), tlist, real(corr3))
    >>> xlabel('Time')
    >>> ylabel('Correlation <x(t)x(0)>')

.. figure:: guide-correlation-1.png
    :align: center
    :width: 4in
	

.. _correlation-nosteady:

Non-steadystate correlation function
====================================
    
More generally, we can also calculate correlation functions of the kind :math:`\left<a(t_1+t_2)b(t_1)\right>`, i.e., the correlation function of a system that is not in its steadystate. In QuTiP, we can evoluate such correlation functions using, e.g., the function :func:`qutip.correlation.correlation`. This function returns a matrix with the correlations as a function of the two time coordinates::

    >>> tlist = linspace(0,10.0,200)
    >>> a  = destroy(10)
    >>> x  = a.dag() + a
    >>> H  = a.dag()*a
    >>> alpha = 2.5
    >>> corr = correlation(H, coherent_dm(10, alpha), tlist, tlist, [sqrt(0.25)*a], x, x)
    >>> 
    >>> from pylab import *
    >>> pcolor(corr)
    >>> xlabel('Time t2')
    >>> ylabel('Time t1')
    >>> title('Correlation <x(t1+t2)x(t1)>')
    >>> show()


.. figure:: guide-correlation-2.png
   :align:  center
   :width: 4in
   
   :math:`\alpha = 2.5`


.. figure:: guide-correlation-3.png
   :align:  center
   :width: 4in
   
   :math:`\alpha = 0`

Notice that in the figure above to the right, where :math:`\alpha = 0.0` and the system therefore initially is in its steadystate, that the correlations does not depend on the :math:`t_1` coordinate, and we could in this case have used the steadystate solver to only calculate the :math:`t_2` dependence. 


