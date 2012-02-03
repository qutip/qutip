.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _guide-eseries:

An Overview of the Eseries Class
********************************

Exponential-series representation of time-dependent quantum objects
===================================================================

The eseries object in QuTiP is a representation of an exponential-series expansion of time-dependent quantum objects (a concept borrowed from the quantum optics toolbox). 

An exponential series is parameterized by its amplitude coefficients :math:`c_i` and rates :math:`r_i`, so that the series takes the form :math:`E(t) = \sum_i c_i e^{r_it}`. The coefficients are typically quantum objects (type Qobj: states, operators, etc.), so that the value of the eseries also is a quantum object, and the rates can be either real or complex numbers (describing decay rates and oscillation frequencies, respectively). Note that all amplitude coefficients in an exponential series must be of the same dimensions and composition. 

In QuTiP, an exponential series object is constructed by creating an instance of the class :class:`qutip.eseries`::
    
    >>> es1 = eseries(sigmax(), 1j)

where the first argument is the amplitude coefficient (here, the sigma-X operator), and the second argument is the rate. The eseries in this example represents the time-dependent operator :math:`\sigma_x e^{i t}`. 

To add more terms to an :class:`qutip.eseries` object we simply add objects using the ``+`` operator::

    >>> es2 = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)

The :class:`qutip.eseries` in this example represents the operator :math:`0.5 \sigma_x e^{i\omega t} + 0.5 \sigma_x e^{-i\omega t}`, which is the exponential series representation of :math:`\sigma_x \cos(\omega t)`. Alternatively, we can also specify a list of amplitudes and rates when the :class:`qutip.eseries` is created::

    >>> es2 = eseries([0.5 * sigmax(), 0.5 * sigmax()], [1j * omega, -1j * omega])

We can inspect the structure of an :class:`qutip.eseries` object by printing it to the standard output console::

    >>> print es2
    ESERIES object: 2 terms
    Hilbert space dimensions: [[2], [2]]
    Exponent #0 = 1j
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.   0.5]
     [ 0.5  0. ]]
    
    Exponent #1 = -1j
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.   0.5]
     [ 0.5  0. ]]

and we can evaluate it at time `t` by using the :func:`qutip.eseries.esval` function::

    >>> esval(es2, 0.0)                    # equivalent to es2.value(0.0)
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.+0.j  1.+0.j]
     [ 1.+0.j  0.+0.j]]
    
or for a list of times ``[0.0, 1.0 * pi, 2.0 * pi]``::

    >>> tlist = [0.0, 1.0 * pi, 2.0 * pi]
    >>> esval(es2, tlist)                  # equivalent to es2.value(tlist)
    [
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.+0.j  1.+0.j]
     [ 1.+0.j  0.+0.j]]
    , Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.+0.j -1.+0.j]
     [-1.+0.j  0.+0.j]]
    , Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.+0.j  1.+0.j]
     [ 1.+0.j  0.+0.j]]
    ]


To calculate the expectation value of an time-dependent operator represented by an :class:`qutip.eseries`, we use the :func:`qutip.expect` function. For example, consider the operator :math:`\sigma_x \cos(\omega t) + \sigma_z\sin(\omega t)`, and say we would like to know the expectation value of this operator for a spin in its excited state (``rho = fock_dm(2,1)`` produce this state)::

    >>> es3 = eseries([0.5*sigmaz(), 0.5*sigmaz()], [1j, -1j]) + eseries([-0.5j*sigmax(), 0.5j*sigmax()], [1j, -1j])
    >>> rho = fock_dm(2, 1)
    >>> es3_expect = expect(rho, es3)
    >>> print es3_expect
    ESERIES object: 2 terms
    Hilbert space dimensions: [[1, 1]]
    Exponent #0 = -1j
    (-0.5+0j)
    Exponent #1 = 1j
    (-0.5+0j)
    >>> print es3_expect.value([0.0, pi/2])
    [ -1.00000000e+00+0.j  -6.12303177e-17+0.j]
    
Note the expectation value of the :class:`qutip.eseries` object, ``expect(rho, es3)``, itself is an :class:`qutip.eseries`, but with amplitude coefficients that are C-numbers instead of quantum operators. To evaluate the C-number :class:`qutip.eseries` at the times `tlist` we use ``esval(es3_expect, tlist)``, or, equivalently, ``es3_expect.value(tlist)``.

Applications of exponential series
==================================

The exponential series formalism can be useful for the time-evolution of quantum systems. One approach to calculating the time evolution of a quantum system is to diagonalize its Hamiltonian (or Liouvillian, for dissipative systems) and to express the propagator (e.g., :math:`\exp(-iHt) \rho \exp(iHt)`) as an exponential series. 

The QuTiP function :func:`qutip.es2ode` and :func:`qutip.essolve` use this method to evolve quantum systems in time. The exponential series approach is particularly suitable for cases when the same system is to be evolved for many different initial states, since the diagonalization only needs to be performed once (as opposed to e.g. the ode solver that would need to be ran independently for each initial state).

As an example, consider a spin-1/2 with a Hamiltonian pointing in the :math:`\sigma_z` direction, and that is subject to noise causing relaxation. For a spin originally is in the up state, we can create an :class:`qutip.eseries` object describing its dynamics by using the :func:`qutip.es2ode` function::

    >>> psi0 = basis(2,1)
    >>> H = sigmaz()
    >>> L = liouvillian(H, [sqrt(1.0) * destroy(2)])
    >>> es = ode2es(L, psi0)

The :func:`qutip.eseries.ode2es` function diagonalizes the Liouvillian :math:`L` and creates an exponential series with the correct eigenfrequencies and amplitudes for the initial state :math:`\psi_0` (`psi0`).

We can examine the resulting :class:`qutip.eseries` object by printing a text representation::

    >>> print es
    ESERIES object: 2 terms
    Hilbert space dimensions: [[2], [2]]
    Exponent #0 = 0j
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 1.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j]]
    
    Exponent #1 = (-1+0j)
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[-1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j]]

or by evaluate it and arbitrary points in time (here at 0.0 and 1.0)::

    >>> print es.value([0.0, 1.0])
    [Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j]]
    , Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.63212056+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.36787944+0.j]]
    ]

and the expectation value of the exponential series can be calculated using the :func:`qutip.expect` function::

    >>> es_expect = expect(sigmaz(), es)

The result `es_expect` is now an exponential series with C-numbers as amplitudes, which easily can be evaluated at arbitrary times::

    >>> print es_expect.value([0.0, 1.0, 2.0, 3.0])
    [-1.00000000+0.j  0.26424112+0.j  0.72932943+0.j  0.90042586+0.j]
    >>> tlist = linspace(0.0, 10.0, 100)
    >>> sz_expect = es_expect.value(tlist)
    >>>
    >>> from pylab import *
    >>> plot(tlist, sz_expect)
    >>> xlabel("Time")
    >>> ylabel("Expectation value of sigma-Z")
    >>> show()


.. figure:: guide-eseries-1.png
    :align: center
    :width: 4in
    
    The expectation value of the :math:`\sigma_z` operator, calculated using the exponential-series expansion of the Liouvillian.
