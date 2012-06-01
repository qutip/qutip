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
The most general way to write a time-dependent Hamiltonian or collapse operator is by using Python functions as the time-dependent coefficients.  To accomplish this, we need to write a python function that returns the time-dependent coefficient.  Additionally, we need to tell QuTiP that a given Hamiltonian or collapse operator should be associated with a given Python function.  To do this, one needs to specify operator-function pairs in list format: ``[Op,py_coeff]``, where ``Op`` is a given Hamiltonian or collapse operator and ``py_coeff`` is the name of the Python function representing the coefficient.  With this format, the form of the Hamiltonian for both ``mesolve`` and ``mcsolve`` is:

>>> H = [H0,[H1,py_coeff1],[H2,py_coeff2],...]

where ``H0`` is a time-INDEPENDENT Hamiltonian, while ``H1``,``H2``, are time-dependent.  Keep in mind that **there must always be at least one time-independent Hamiltonian term**.  The same format can be used for collapse operators:

>>> c_op_list = [[C0,py_coeff0],C1,[C2,py_coeff2],...]

Here we have demonstrated that the ordering of time-dependent and time-indepdendent terms does not matter.  In addition, any or all of the collapse operators may be time-depdendent.  

.. note:: While, in general, you can arrange time-dependent and independent terms in any order you like, it is best to place all time-independent terms first.

As an example, we will look at :ref:`exme41` that has a time-dependent Hamiltonian of the form :math:`H=H_{0}-f(t)H_{1}` where :math:`f(t)` is the time-dependent driving strength given as :math:`f(t)=9\exp\left[-\left( t/5 \right)^{2}\right]`.  The follow code sets up the problem::

    from qutip import *
    # Define atomic states. Use ordering from paper
    ustate = basis(3,0)
    excited = basis(3,1)
    ground = basis(3,2)
    
    # Set where to truncate Fock state for cavity
    N = 2
    
    # Create the atomic operators needed for the Hamiltonian
    sigma_ge = tensor(qeye(N), ground * excited.dag()) # |g><e|
    sigma_ue = tensor(qeye(N), ustate * excited.dag()) # |u><e|
    
    # Create the photon operator
    a = tensor(destroy(N), qeye(3))
    ada = tensor(num(N), qeye(3))
    
    # Define collapse operators
    c_op_list = []
    # Cavity decay rate
    kappa = 1.5
    c_op_list.append(sqrt(kappa) * a)
    
    # Atomic decay rate
    gamma = 6 #decay rate
    # Use Rb branching ratio of 5/9 e->u, 4/9 e->g
    c_op_list.append(sqrt(5*gamma/9) * sigma_ue)
    c_op_list.append(sqrt(4*gamma/9) * sigma_ge)
    
    # Define time vector
    t = linspace(-15,15,100)
	
    # Define initial state
    psi0 = tensor(basis(N,0), ustate)
    
    # Define states onto which to project
    state_GG = tensor(basis(N,1), ground)
    sigma_GG = state_GG * state_GG.dag()
    state_UU = tensor(basis(N,0), ustate)
    sigma_UU = state_UU * state_UU.dag()
    
    # Set up the time varying Hamiltonian
    g = 5 #coupling strength
    H0 = -g * (sigma_ge.dag() * a + a.dag() * sigma_ge) #time-INDEPENDENT term
    H1 = (sigma_ue.dag() + sigma_ue) #time-DEPENDENT term

Given that we have a single time-dependent Hamiltonian term, and constant collapse terms, we need to specify a single Python function for the coefficient :math:`f(t)`.  In this case, one can simply do::

	def H1_coeff(t, args):
	        return 9 * exp(-(t/5.)**2)

In this case, the return value dependents only on time.  However, when specifying Python functions for coefficients, **the function must have (t,args) as the input variables, in that order**.  Having specified our coefficient function, we can now specify the Hamiltonian in list format and call the solver (in this case ``mesolve``)::

    H=[H0,[H1,H1_coeff]]
    output = mesolve(H, psi0, t, c_op_list,[ada, sigma_UU, sigma_GG])

We can call the Monte-Carlo solver in the exact same way (if using the default ``ntraj=500``):

>>> output = mcsolve(H, psi0, t, c_op_list,[ada, sigma_UU, sigma_GG])

The output from the master equation solver is identical to that shown in the examples, the monte-carlo however will be noticably off, suggesting we should increase the number of trajectories for this example.  In addition, we can also consider the decay of a simple Harmonic oscillator with time-varying decay rate::

    from qutip import *
    kappa=0.5
    def col_coeff(t,args): #coefficient function
        return sqrt(kappa*exp(-t))
    N=10 #number of basis states
    a=destroy(N)
    H=a.dag()*a #simple HO
    psi0=basis(N,9) #initial state
    c_op_list=[[a,col_coeff]] #time-depdendent collapse term
    tlist=linspace(0,10,100)
    output=mesolve(H,psi0,tlist,c_op_list,[a.dag()*a])

A comparision of this time-dependent damping, with that of a constant decay term is presented below.

.. figure:: td-decay.png
   :width: 4in
   :align: center


Using the args variable
------------------------



String Format Method
=====================
Here


Function Based Hamiltonian
==========================

If a callback function is passed as first parameter to the solver function (instead of :class:`qutip.Qobj` Hamiltonian), then this function is called at each time step and is expected to return the :class:`qutip.Qobj` Hamiltonian for that point in time. The callback function takes two arguments: the time `t` and list additional Hamiltonian arguments ``args``. This list of additional arguments is the same object as is passed as the sixth parameter to the solver function (only used for time-dependent Hamiltonians).

For example, let's consider a two-level system with energy splitting 1.0, and subject to a time-dependent field that couples to the :math:`\sigma_x` operator with amplitude 0.1. Furthermore, to make the example a little bit more interesting, let's also assume that the two-level system is subject to relaxation, with relaxation rate 0.01. The following code calculates the dynamics of the system in the absence and in the presence of the time-dependent driving signal


Reusing Time-Dependent Hamiltonian Data
=======================================

.. note:: This section covers a specialized topic and may be skipped if you are new to QuTiP.



