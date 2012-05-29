.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _time:

*************************************
Solving Time-dependent Hamiltonians
*************************************

Methods for Writing Time-Dependent Hamiltonians
===============================================

In the previous examples of quantum evolution, we assumed that the systems under consideration were described by time-independent Hamiltonians.  However, many systems have explicit time-dependence in either the Hamiltonian, or the collpase operators describing coupling to the environment, and sometimes both components might depend on time.  The two main evolution solvers in QuTiP, :func:`qutip.mesolve` and :func:`qutip.mcsolve`, discussed in :ref:`master` and :ref:`monte` respectively, are capable of handling time-dependent Hamiltonians and collapse terms.  There are, in general, three different ways to implement time-dependent problems in QuTiP 2:


1. **Function based**: Hamiltonian / collapse operators expressed using [qobj,func] pairs, where the time-dependent coefficients of the Hamiltonian (or collapse operators) are expresed in the Python functions.


2. **String (Cython) based**: The Hamiltonian and/or collapse operators are expressed as a list of [qobj,string] pairs, where the time-dependent coefficients are represented as strings.  The resulting Hamiltonian is then compiled into c-code using Cython and executed.


3. **Hamiltonian function (outdated)**: The Hamiltonian is itself a Python function with time-dependence.  Collapse operators must be time-independent using this input format. 


Give the multiple choices of input style, the first question that arrises is which option to choose?  In short, the function based method (option #1) is the most general, allowing for essentially arbitrary coefficents expressed via user defined functions.  However, by automatically compiling your system into c-code, the second option (string based) tends to be more efficent and will run faster.  Of course, for small system sizes and evolution times, the difference will be minor.  Although this method does not support all time-dependent coefficients that one can think of, it does support essentially all problems that one would typically encounter.  If you can write you time-dependent coefficients using any of the following functions, or combinations thereof (including constants) then you may use this method::

   ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil'
   , 'copysign', 'cos', 'cosh', 'degrees', 'erf', 'erfc', 'exp', 'expm1'
   , 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma'
   , 'hypot', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 'log10', 'log1p'
   , 'modf', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc']

Finally option #3, expressing the Hamiltonian as a Python function, is the original method for time-dependence in QuTiP 1.x.  However, this method is somewhat less efficient then the previously mentioned methods, and does not allow for time-dependent collapse operators.

Function Based Time-Dependence
==============================
Here


String Format Method
=====================
Here


Function Based Hamiltonian
==========================

If a callback function is passed as first parameter to the solver function (instead of :class:`qutip.Qobj` Hamiltonian), then this function is called at each time step and is expected to return the :class:`qutip.Qobj` Hamiltonian for that point in time. The callback function takes two arguments: the time `t` and list additional Hamiltonian arguments ``args``. This list of additional arguments is the same object as is passed as the sixth parameter to the solver function (only used for time-dependent Hamiltonians).

For example, let's consider a two-level system with energy splitting 1.0, and subject to a time-dependent field that couples to the :math:`\sigma_x` operator with amplitude 0.1. Furthermore, to make the example a little bit more interesting, let's also assume that the two-level system is subject to relaxation, with relaxation rate 0.01. The following code calculates the dynamics of the system in the absence and in the presence of the time-dependent driving signal





