.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _time:

**************************************************************
Solving Time-dependent Hamiltonians (unitary and non-unitary)
**************************************************************

In the previous examples of quantum evolution, we assumed that the systems under consideration were described by time-independent Hamiltonians.  However, many systems have explicit time-dependence in either the Hamiltonian, or the collpase operators describing coupling to the environment, and sometimes both components might depend on time.  The two main evolution solvers in QuTiP, :func:`qutip.mesolve` and :func:`qutip.mcsolve`, are capable of handling time-dependent Hamiltonians and collapse terms.  There are, in general, three different ways to input time-dependent problems in QutiP 2:

- String (Cython) based: The Hamiltonian and/or collapse operators are expressed as a list of [qobj,string] pairs, where the time-dependence is expresed in the string.  The resulting Hamiltonian is then compiled into c-code using Cython and executed.

- Function based: Hamiltonian / collapse operators expressed using [qobj,func] pairs, where the time-dependence is expressed in the Python function.

- Hamiltonian function (outdated): Hamiltonian is itself a Python function with time-dependence.  Collapse operators must be constant using this input format. 








 If a callback function is passed as first parameter to the solver function (instead of :class:`qutip.Qobj` Hamiltonian), then this function is called at each time step and is expected to return the :class:`qutip.Qobj` Hamiltonian for that point in time. The callback function takes two arguments: the time `t` and list additional Hamiltonian arguments ``args``. This list of additional arguments is the same object as is passed as the sixth parameter to the solver function (only used for time-dependent Hamiltonians).

For example, let's consider a two-level system with energy splitting 1.0, and subject to a time-dependent field that couples to the :math:`\sigma_x` operator with amplitude 0.1. Furthermore, to make the example a little bit more interesting, let's also assume that the two-level system is subject to relaxation, with relaxation rate 0.01. The following code calculates the dynamics of the system in the absence and in the presence of the time-dependent driving signal::





