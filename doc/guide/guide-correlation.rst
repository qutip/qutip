.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

.. _guide-correlation:

Solving Two-Time Correlation Functions
**************************************

Correlation functions
=====================

Here we demonstrate how to calculate two-time correlation functions in QuTiP. Using the quantum regression theorem, we can apply the equation of motion for the system itself also to calculate two-time correlation functions. In QuTiP, there are family functions that assists in this process: :func:`qutip.correlation_ode`, :func:`qutip.correlation_es`, :func:`qutip.correlation_mc`, and :func:`qutip.correlation_ss_ode`, :func:`qutip.correlation_ss_es`, :func:`qutip.correlation_ss_mc`. As the names suggest, these functions use the ODE, the exponential series, and the Monte-Carlo solvers, respectively, to evolve the correlation functions in time. The following table describes in detail the usage of each function:

+----------------------------------+---------------------------------------------+-----------------------------------------+
| Function                         | Input parameters                            | Usage                                   |
+==================================+=============================================+=========================================+
| :func:`qutip.correlation_ss_es`  | `H` - the Hamiltonian, `tlist` - list of    | Calculates the steady state correlation |
|                                  | times to evaluate the correlation function, | <a(0)b(tau)>, using the Exponential     |
|                                  | `c_op_list` - list of collapse operators,   | series solver.                          |
|                                  | `a_op` and `b_op` - the operators for which |                                         |
|                                  | to calculate correlations.                  |                                         |
+----------------------------------+---------------------------------------------+-----------------------------------------+
| :func:`qutip.correlation_ss_ode` | Same as above.                              | Calculates the steady state correlation |
|                                  |                                             | <a(0)b(tau)>, using the ODE solver.     |
+----------------------------------+---------------------------------------------+-----------------------------------------+
| :func:`qutip.correlation_ss_mc`  | Same as above.                              | Calculates the steady state correlation |
|                                  |                                             | <a(0)b(tau)>, using the Monte-Carlo     |
|                                  |                                             | evolution.                              |
+----------------------------------+---------------------------------------------+-----------------------------------------+
| :func:`qutip.correlation_es`     | `H` - the Hamiltonian, `rho0` - initial     | Calculates the correlation              |
|                                  | state of the system, `t1list` and `t2list`- | <a(t1)b(t1+t2)>, using the Exponential  |
|                                  | list of times to evaluate the correlation   | series solver.                          |
|                                  | function (t1 for operator `a` and t1+t2 for |                                         |
|                                  | operator `b`), `c_op_list`- list of collapse|                                         |
|                                  | operators, `a_op` and `b_op` - the operators|                                         |
|                                  | for which to calculate correlations.        |                                         |
+----------------------------------+---------------------------------------------+-----------------------------------------+
| :func:`qutip.correlation_ode`    | Same as above.                              | Calculates the  correlation             |
|                                  |                                             | <a(t1)b(t1+t2)>, using the ODE solver.  |
+----------------------------------+---------------------------------------------+-----------------------------------------+
| :func:`qutip.correlation_mc`     | Same as above.                              | Calculates the correlation              |
|                                  |                                             | <a(t1)b(t1+t2)>, using the Monte-Carlo  |
|                                  |                                             | evolution.                              |
+----------------------------------+---------------------------------------------+-----------------------------------------+

The most common use-case is to calculate correlation functions of the kind ``<a(0)b(t)>``, in which case we use the correlation function solvers that start from the steady state, i.e., the :func:`qutip.correlation_ss_xxx` functions. These functions return a vector (in general complex) with the correlations between the operators as a function of the difference time. 

The following code demonstrates how to calculate the ``<x(0)x(t)>`` correlation for a leaky cavity with three different relaxation rates::

    >>> tlist = linspace(0,10.0,200);
    >>> a  = destroy(10)
    >>> x  = a.dag() + a
    >>> H  = a.dag()*a
    >>> 
    >>> corr1 = correlation_ss_ode(H, tlist, [sqrt(0.5)*a], x, x)
    >>> corr2 = correlation_ss_ode(H, tlist, [sqrt(1.0)*a], x, x)
    >>> corr3 = correlation_ss_ode(H, tlist, [sqrt(2.0)*a], x, x)
    >>> 
    >>> from pylab import *
    >>> plot(tlist, real(corr1), tlist, real(corr2), tlist, real(corr3))
    >>> xlabel('Time')
    >>> ylabel('Correlation <x(0)x(t)>')

http://qutip.googlecode.com/svn/wiki/images/guide-correlation-1.png

More generally, we can also calculate correlation functions of the kind ``<a(t1)b(t1+t2)>``, i.e., the correlation function of a system that is not in its steadystate. In QuTiP, we can evoluate such correlation functions using the :func:`qutip.correlation_xxx`. These functions returns a matrix with the correlations as a function of the two time coordinates::

    >>> tlist = linspace(0,10.0,200);
    >>> a  = destroy(10)
    >>> x  = a.dag() + a
    >>> H  = a.dag()*a
    >>> alpha = 2.5
    >>> corr = correlation_ode(H, coherent_dm(10, alpha), tlist, tlist, [sqrt(0.25)*a], x, x)
    >>> 
    >>> from pylab import *
    >>> pcolor(corr)
    >>> xlabel('Time t2')
    >>> ylabel('Time t1')
    >>> title('Correlation <x(t1)x(t1+t2)>')
    >>> show()

+---------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
| .. figure:: http://qutip.googlecode.com/svn/wiki/images/guide-correlation-2.png | .. figure:: http://qutip.googlecode.com/svn/wiki/images/guide-correlation-3.png |
|    :align:  center                                                              |    :align:  center                                                              |
|                                                                                 |                                                                                 |
|    ``alpha = 2.5``                                                              |    ``alpha = 0.0``                                                              |
|                                                                                 |                                                                                 |
+---------------------------------------------------------------------------------+---------------------------------------------------------------------------------+


Notice that in the figure above to the right, where ``alpha = 0.0`` and the system therefore initially is in its steadystate, that the correlations does not depend on the ``t1`` coordinate, and we could in this case have used the steadystate solver to only calculate the ``t2`` dependence. 


