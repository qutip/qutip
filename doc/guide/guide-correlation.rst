.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _correlation:

******************************
Two-time correlation functions
******************************

With the QuTiP time-evolution functions, for example :func:`qutip.odesolve.mesolve` and :func:`qutip.mcsolve`, the wave function or density matrix for a system can be propagated to an arbitrary time :math:`t`, given an initial state: :math:`\rho(t)=V(t, 0)\left\{\rho(0)\right\}`.
calculated, which can be used to evaluate the expectation values of arbitrary combinations of *same-time* operators. To calculate *two-time* correlation functions on the form :math:`\left<A(t+\tau)B(t)\right>`, we can use the quantum regression theorem to write

.. math::

    \left<A(t+\tau)B(t)\right> = {\rm Tr}_{\rm sys}\left[A V(t+\tau, t)\left\{B\rho(t)\right\}\right]

where :math:`V(t+\tau, t)` according to the Lindblad master equation from time :math:`t` to :math:`t+\tau`, so that :math:`\rho(t)=V(t, 0)\left\{\rho(0)\right\}`. To evalulate the correlation function we can therefore first calculate :math:`\rho(t)` using one of the QuTiP evolution solvers with :math:`\rho(0)` as initial state, and then again use the same solver to calculate :math:`V(t+\tau, t)\left\{B\rho(t)\right\}` using :math:`B\rho(t)` as initial state.

Here we demonstrate how to calculate two-time correlation functions in QuTiP. Using the quantum regression theorem, we can apply the equation of motion for the system itself also to calculate two-time correlation functions. In QuTiP, there are two functions that assists in this process: :func:`qutip.correlation.correlation` and :func:`qutip.correlation.correlation_ss` (for steadystate correlations). Both these functions can use one of the following evolution solvers: Master-equation, Exponential series and the Monte-Carlo. The choice of solver is defined by the optional last argument `solver`. The following table describes in detail the usage of each function:

.. tabularcolumns:: | p{5cm} | L |

+----------------------------------------------+-----------------------------------------+
| Function                                     | Usage                                   |
+==============================================+=========================================+
| :func:`qutip.correlation.correlation_ss`     | Calculates the steady state correlation |
|                                              | :math:`\left<a(0)b(\tau)\right>`,       |
|                                              | using the either the master eqaution,   |
|                                              | the exponential series, or the          |
|                                              | Monte Carlo solver.                     |
+----------------------------------------------+-----------------------------------------+
| :func:`qutip.correlation.correlation`        | Calculates the correlation function     |
|                                              | :math:`\left<a(t_1)b(t_1+t_2)\right>`,  |
|                                              | using the either the master eqaution,   |
|                                              | the exponential series, or the          |
|                                              | Monte Carlo solver.                     |
+----------------------------------------------+-----------------------------------------+

The most common use-case is to calculate correlation functions of the kind :math:`\left<a(0)b(t)\right>`, in which case we use the correlation function solvers that start from the steady state, e.g., the :func:`qutip.correlation.correlation_ss` function. These functions return a vector (in general complex) with the correlations between the operators as a function of the difference time. 

.. _correlation-steady:

Steadystate correlation function
================================

The following code demonstrates how to calculate the :math:`\left<x(0)x(t)\right>` correlation for a leaky cavity with three different relaxation rates::

    >>> tlist = linspace(0,10.0,200);
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
    >>> ylabel('Correlation <x(0)x(t)>')

.. figure:: guide-correlation-1.png
    :align: center
    :width: 4in
	

.. _correlation-nosteady:

Non-steadystate correlation function
====================================
    
More generally, we can also calculate correlation functions of the kind :math:`\left<a(t_1)b(t_1+t_2)\right>`, i.e., the correlation function of a system that is not in its steadystate. In QuTiP, we can evoluate such correlation functions using, e.g., the function :func:`qutip.correlation.correlation`. This function returns a matrix with the correlations as a function of the two time coordinates::

    >>> tlist = linspace(0,10.0,200);
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
    >>> title('Correlation <x(t1)x(t1+t2)>')
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


