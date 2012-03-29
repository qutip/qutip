.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _tensor:

******************************************
Using Tensor Products and Partial Traces
******************************************

.. _tensor-products:

Tensor products
===============

To describe the states of multipartite quantum systems, such as two coupled qubits, a qubit coupled to an oscillator, etc..., we need to expand the Hilbert space by taking the tensor product of the state vectors for each of the system components. Similarly, the operators acting on the state vectors in the combined Hilbert space (describing the coupled system) are formed by taking the tensor product of the individual operators.

In QuTiP the function :func:`qutip.tensor.tensor` is used to accomplish this task. The *tensor* function takes as its argument a collection::

>>> tensor(op1,op2,op3)

or a ``list``::

>>> tensor([op1,op2,op3])

of state vectors *or* operators and returns a composite quantum object for the combined Hilbert space.  The returned quantum objects type is the same as that of the input(s).

For example, the state vector describing two qubits in their ground states is formed by taking the tensor product of the two single-qubit ground state vectors::
    
    >>> tensor(basis(2,0), basis(2,0))
    Quantum object: dims = [[2, 2], [1, 1]], shape = [4, 1], type = ket
    Qobj data = 
    [[ 1.]
     [ 0.]
     [ 0.]
     [ 0.]]

or equivilently using the ``list`` format::

    >>> tensor([basis(2,0), basis(2,0)])
    Quantum object: dims = [[2, 2], [1, 1]], shape = [4, 1], type = ket
    Qobj data = 
    [[ 1.]
     [ 0.]
     [ 0.]
     [ 0.]]

This is straight forward to generalize to more qubits by adding more component state vectors in the argument list to the :func:`qutip.tensor.tensor` function, as illustrated in the following example::

    >>> tensor((basis(2,0)+basis(2,1)).unit(), (basis(2,0)+basis(2,1)).unit(), basis(2,0))
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = [8, 1], type = ket
    Qobj data = 
    [[ 0.5]
     [ 0. ]
     [ 0.5]
     [ 0. ]
     [ 0.5]
     [ 0. ]
     [ 0.5]
     [ 0. ]]

This state is slightly more complicated, describing two qubits in a superposition between the up and down states, while the third qubit remains in it's ground state.

To construct operators that act on an extended Hilbert space of a combined system, we similarly pass a list of operators for each component system to the :func:`qutip.tensor.tensor` function. For example, to form the operator that represents the simultaneous action of the sigma x operator on two qubits::

    >>> tensor(sigmax(), sigmax())
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
    Qobj data = 
    [[ 0.  0.  0.  1.]
     [ 0.  0.  1.  0.]
     [ 0.  1.  0.  0.]
     [ 1.  0.  0.  0.]]

To create operators in a combined Hilbert space that only act only on a single component, we take the tensor product of the operator acting on the subspace of interest, with the identity operators corresponding to the components that are to be unchanged. For example, the operator that represents sigma-z on the first qubit in a two-qubit system, while leaving the second qubit unaffected::

    >>> tensor(sigmaz(), qeye(2))
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
    Qobj data = 
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0. -1.  0.]
     [ 0.  0.  0. -1.]]
    

.. _tensor-product-example:

Example: Constructing composite Hamiltonians
============================================

The :func:`qutip.tensor.tensor` function is extensively used when constructing Hamiltonians for composite systems. Here we'll look at some simple examples.

.. _tensor-product-example-2qubits:

Two coupled qubits
------------------

First, let's consider a system of two coupled qubits. Assume that both qubit has equal energy splitting, and that the qubits are coupled through a sigmax-sigmax interaction with strength g = 0.05 (in units where the bare qubit energy splitting is unity). The Hamiltonian describing this system is::

    >>> H = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz()) + 0.05 * tensor(sigmax(), sigmax())
    >>> H
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
    Qobj data = 
    [[ 2.   0.   0.   0.05]
     [ 0.   0.   0.05 0. ]
     [ 0.   0.05 0.   0. ]
     [ 0.05 0.   0.  -2. ]]

.. _tensor-product-example-3qubits:

Three coupled qubits
--------------------

The two-qubit example is easily generalized to three coupled qubits::

    >>> H = tensor(sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz()) + 
    0.5 * tensor(sigmax(), sigmax(), qeye(2)) +  0.25 * tensor(qeye(2), sigmax(), sigmax())
    >>> H
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = [8, 8], type = oper, isHerm = True
    Qobj data = 
    [[ 3.    0.    0.    0.25  0.    0.    0.5   0.  ]
     [ 0.    1.    0.25  0.    0.    0.    0.    0.5 ]
     [ 0.    0.25  1.    0.    0.5   0.    0.    0.  ]
     [ 0.25  0.    0.   -1.    0.    0.5   0.    0.  ]
     [ 0.    0.    0.5   0.    1.    0.    0.    0.25]
     [ 0.    0.    0.    0.5   0.   -1.    0.25  0.  ]
     [ 0.5   0.    0.    0.    0.    0.25 -1.    0.  ]
     [ 0.    0.5   0.    0.    0.25  0.    0.   -3.  ]]
    

.. _tensor-product-example-jcmodel:

A two-level system coupled to a cavity: The Jaynes-Cummings model
-------------------------------------------------------------------

The simplest possible quantum mechanical description for light-matter interaction is encapsulated in the Jaynes-Cummings model, which describes the coupling between a two-level atom and a single-mode electomagnetic field (a cavity mode). Denoting the energy splitting of the atom and cavity omega_a and omega_c, respectively, and the atom-cavity interaction strength g, the Jaynes-Cumming Hamiltonian can be constructed as::

    >>> N = 10
    >>> omega_a = 1.0
    >>> omega_c = 1.25
    >>> g = 0.05
    >>> a = tensor(qeye(2), destroy(N))
    >>> sm = tensor(destroy(2), qeye(N))
    >>> sz = tensor(sigmaz(), qeye(N))
    >>> H = 0.5 * omega_a * sz + omega_c * a.dag() * a + g * (a.dag() * sm + a * sm.dag())

Here N is the number of Fock states included in the cavity mode. 

.. _tensor-ptrace:

Partial trace
=============

The partial trace is an operation that reduces the dimension of a Hilbert space by eliminating some degrees of freedom by averaging (tracing). In this sense it is therefore the converse of the tensor product. It is useful when one is interested in only a part of a coupled quantum system.  For open quantum systems, this typically involves tracing over the environment leaving only the system of interest.  In QuTiP the function :func:`qutip.ptrace.ptrace` is used to take partial traces. It takes two arguments: ``rho`` is the density matrix (or state vector) of the composite system, and ``sel`` is a ``list`` of integers that mark the component systems that should be **kept**.  All other components are traced over.

For example, the density matrix describing a single qubit obtained from a coupled two-qubit system is obtained via::

    >>> psi = tensor(basis(2,0), basis(2,1))
    >>> ptrace(psi, 0)
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 1.  0.]
     [ 0.  0.]]
    >>> ptrace(psi, 1)
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.  0.]
     [ 0.  1.]]

Note that the partial trace always results in a density matrix (mixed state), regardless of whether the composite system is a pure state (described by a state vector) or a mixed state (described by a density matrix)::

    >>> psi = tensor((basis(2,0)+basis(2,1)).unit(), basis(2,0))
    >>> psi
    Quantum object: dims = [[2, 2], [1, 1]], shape = [4, 1], type = ket
    Qobj data = 
    [[ 0.70710678]
     [ 0.        ]
     [ 0.70710678]
     [ 0.        ]]
    >>> ptrace(psi, 0)
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.5  0.5]
     [ 0.5  0.5]]
    >>> rho = tensor(ket2dm((basis(2,0)+basis(2,1)).unit()), fock_dm(2,0))
    >>> rho
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = True
    Qobj data = 
    [[ 0.5  0.   0.5  0. ]
     [ 0.   0.   0.   0. ]
     [ 0.5  0.   0.5  0. ]
     [ 0.   0.   0.   0. ]]
    >>> ptrace(rho, 0)
    Quantum object: dims = [[2], [2]], shape = [2, 2], type = oper, isHerm = True
    Qobj data = 
    [[ 0.5  0.5]
     [ 0.5  0.5]]

