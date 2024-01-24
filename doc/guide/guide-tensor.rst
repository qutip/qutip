.. _tensor:

******************************************
Using Tensor Products and Partial Traces
******************************************


.. _tensor-products:

Tensor products
===============

To describe the states of multipartite quantum systems - such as two coupled qubits, a qubit coupled to an oscillator, etc. - we need to expand the Hilbert space by taking the tensor product of the state vectors for each of the system components. Similarly, the operators acting on the state vectors in the combined Hilbert space (describing the coupled system) are formed by taking the tensor product of the individual operators.

In QuTiP the function :func:`~qutip.core.tensor.tensor` is used to accomplish this task. This function takes as argument a collection::

>>> tensor(op1, op2, op3) # doctest: +SKIP

or a ``list``::

>>> tensor([op1, op2, op3]) # doctest: +SKIP

of state vectors *or* operators and returns a composite quantum object for the combined Hilbert space. The function accepts an arbitrary number of states or operators as argument. The type returned quantum object is the same as that of the input(s).

For example, the state vector describing two qubits in their ground states is formed by taking the tensor product of the two single-qubit ground state vectors:

.. testcode:: [tensor]

    print(tensor(basis(2, 0), basis(2, 0)))

**Output**:

.. testoutput:: [tensor]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
    Qobj data =
    [[1.]
     [0.]
     [0.]
     [0.]]

or equivalently using the ``list`` format:

.. testcode:: [tensor]

    print(tensor([basis(2, 0), basis(2, 0)]))

**Output**:

.. testoutput:: [tensor]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
    Qobj data =
    [[1.]
     [0.]
     [0.]
     [0.]]

This is straightforward to generalize to more qubits by adding more component state vectors in the argument list to the :func:`~qutip.core.tensor.tensor` function, as illustrated in the following example:

.. testcode:: [tensor]

    print(tensor((basis(2, 0) + basis(2, 1)).unit(), (basis(2, 0) + basis(2, 1)).unit(), basis(2, 0)))

**Output**:

.. testoutput:: [tensor]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = (8, 1), type = ket
    Qobj data =
    [[0.5]
     [0. ]
     [0.5]
     [0. ]
     [0.5]
     [0. ]
     [0.5]
     [0. ]]


This state is slightly more complicated, describing two qubits in a superposition between the up and down states, while the third qubit is in its ground state.

To construct operators that act on an extended Hilbert space of a combined system, we similarly pass a list of operators for each component system to the :func:`~qutip.core.tensor.tensor` function. For example, to form the operator that represents the simultaneous action of the :math:`\sigma_x` operator on two qubits:

.. testcode:: [tensor]

    print(tensor(sigmax(), sigmax()))

**Output**:

.. testoutput:: [tensor]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [1. 0. 0. 0.]]

To create operators in a combined Hilbert space that only act on a single component, we take the tensor product of the operator acting on the subspace of interest, with the identity operators corresponding to the components that are to be unchanged. For example, the operator that represents :math:`\sigma_z` on the first qubit in a two-qubit system, while leaving the second qubit unaffected:

.. testcode:: [tensor]

    print(tensor(sigmaz(), identity(2)))

**Output**:

.. testoutput:: [tensor]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0. -1.  0.]
     [ 0.  0.  0. -1.]]


.. _tensor-product-example:

Example: Constructing composite Hamiltonians
============================================

The :func:`~qutip.core.tensor.tensor` function is extensively used when constructing Hamiltonians for composite systems. Here we'll look at some simple examples.

.. _tensor-product-example-2qubits:

Two coupled qubits
------------------

First, let's consider a system of two coupled qubits. Assume that both the qubits have equal energy splitting, and that the qubits are coupled through a :math:`\sigma_x\otimes\sigma_x` interaction with strength g = 0.05 (in units where the bare qubit energy splitting is unity). The Hamiltonian describing this system is:

.. testcode:: [tensor]

    H = tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz()) + 0.05 * tensor(sigmax(), sigmax())

    print(H)

**Output**:

.. testoutput:: [tensor]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[ 2.    0.    0.    0.05]
     [ 0.    0.    0.05  0.  ]
     [ 0.    0.05  0.    0.  ]
     [ 0.05  0.    0.   -2.  ]]

.. _tensor-product-example-3qubits:

Three coupled qubits
--------------------

The two-qubit example is easily generalized to three coupled qubits:

.. testcode:: [tensor]

    H = (tensor(sigmaz(), identity(2), identity(2)) + tensor(identity(2), sigmaz(), identity(2)) + tensor(identity(2), identity(2), sigmaz()) + 0.5 * tensor(sigmax(), sigmax(), identity(2)) + 0.25 * tensor(identity(2), sigmax(), sigmax()))

    print(H)

**Output**:

.. testoutput:: [tensor]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
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

The simplest possible quantum mechanical description for light-matter interaction is encapsulated in the Jaynes-Cummings model, which describes the coupling between a two-level atom and a single-mode electromagnetic field (a cavity mode). Denoting the energy splitting of the atom and cavity ``omega_a`` and ``omega_c``, respectively, and the atom-cavity interaction strength ``g``, the Jaynes-Cummings Hamiltonian can be constructed as:

.. plot::
    :context: reset

    N = 6

    omega_a = 1.0

    omega_c = 1.25

    g = 0.75

    a = tensor(identity(2), destroy(N))

    sm = tensor(destroy(2), identity(N))

    sz = tensor(sigmaz(), identity(N))

    H = 0.5 * omega_a * sz + omega_c * a.dag() * a + g * (a.dag() * sm + a * sm.dag())

    hinton(H, fig=plt.figure(figsize=(12, 12)))


Here ``N`` is the number of Fock states included in the cavity mode.

.. _tensor-ptrace:

Partial trace
=============

The partial trace is an operation that reduces the dimension of a Hilbert space by eliminating some degrees of freedom by averaging (tracing).
In this sense it is therefore the converse of the tensor product.
It is useful when one is interested in only a part of a coupled quantum system.
For open quantum systems, this typically involves tracing over the environment leaving only the system of interest.
In QuTiP the class method  :meth:`~qutip.core.qobj.Qobj.ptrace` is used to take partial traces. :meth:`~qutip.core.qobj.Qobj.ptrace` acts on the :class:`~qutip.core.qobj.Qobj` instance for which it is called, and it takes one argument ``sel``, which is a ``list`` of integers that mark the component systems that should be **kept**.
All other components are traced out.

For example, the density matrix describing a single qubit obtained from a coupled two-qubit system is obtained via:

.. doctest:: [tensor]
  :options: +NORMALIZE_WHITESPACE

  >>> psi = tensor(basis(2, 0), basis(2, 1))

  >>> psi.ptrace(0)
  Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
  Qobj data =
  [[1. 0.]
   [0. 0.]]

  >>> psi.ptrace(1)
  Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
  Qobj data =
  [[0. 0.]
   [0. 1.]]

Note that the partial trace always results in a density matrix (mixed state), regardless of whether the composite system is a pure state (described by a state vector) or a mixed state (described by a density matrix):

.. doctest:: [tensor]
  :options: +NORMALIZE_WHITESPACE

  >>> psi = tensor((basis(2, 0) + basis(2, 1)).unit(), basis(2, 0))

  >>> psi
  Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
  Qobj data =
  [[0.70710678]
   [0.        ]
   [0.70710678]
   [0.        ]]

  >>> psi.ptrace(0)
  Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
  Qobj data =
  [[0.5 0.5]
   [0.5 0.5]]

  >>> rho = tensor(ket2dm((basis(2, 0) + basis(2, 1)).unit()), fock_dm(2, 0))

  >>> rho
  Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[0.5 0.  0.5 0. ]
   [0.  0.  0.  0. ]
   [0.5 0.  0.5 0. ]
   [0.  0.  0.  0. ]]

  >>> rho.ptrace(0)
  Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
  Qobj data =
  [[0.5 0.5]
   [0.5 0.5]]

Superoperators and Tensor Manipulations
=======================================

As described in :ref:`states-super`, *superoperators* are operators
that act on Liouville space, the vectorspace of linear operators.
Superoperators can be represented
using the isomorphism
:math:`\mathrm{vec} : \mathcal{L}(\mathcal{H}) \to \mathcal{H} \otimes \mathcal{H}` [Hav03]_, [Wat13]_.
To represent superoperators acting on :math:`\mathcal{L}(\mathcal{H}_1 \otimes \mathcal{H}_2)` thus takes some tensor rearrangement to get the desired ordering
:math:`\mathcal{H}_1 \otimes \mathcal{H}_2 \otimes \mathcal{H}_1 \otimes \mathcal{H}_2`.

In particular, this means that :func:`.tensor` does not act as
one might expect on the results of :func:`.to_super`:

.. doctest:: [tensor]

  >>> A = qeye([2])

  >>> B = qeye([3])

  >>> to_super(tensor(A, B)).dims
  [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]

  >>> tensor(to_super(A), to_super(B)).dims
  [[[2], [2], [3], [3]], [[2], [2], [3], [3]]]

In the former case, the result correctly has four copies
of the compound index with dims ``[2, 3]``. In the latter
case, however, each of the Hilbert space indices is listed
independently and in the wrong order.

The :func:`.super_tensor` function performs the needed
rearrangement, providing the most direct analog to :func:`.tensor` on
the underlying Hilbert space. In particular, for any two ``type="oper"``
Qobjs ``A`` and ``B``, ``to_super(tensor(A, B)) == super_tensor(to_super(A), to_super(B))`` and
``operator_to_vector(tensor(A, B)) == super_tensor(operator_to_vector(A), operator_to_vector(B))``. Returning to the previous example:

.. doctest:: [tensor]

  >>> super_tensor(to_super(A), to_super(B)).dims
  [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]

The :func:`.composite` function automatically switches between
:func:`.tensor` and :func:`.super_tensor` based on the ``type``
of its arguments, such that ``composite(A, B)`` returns an appropriate Qobj to
represent the composition of two systems.

.. doctest:: [tensor]

  >>> composite(A, B).dims
  [[2, 3], [2, 3]]

  >>> composite(to_super(A), to_super(B)).dims
  [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]

QuTiP also allows more general tensor manipulations that are
useful for converting between superoperator representations [WBC11]_.
In particular, the :func:`~qutip.core.tensor.tensor_contract` function allows for
contracting one or more pairs of indices.
This can be used to find superoperators that represent partial trace maps.
Using this functionality, we can construct some quite exotic maps,
such as a map from :math:`3 \times 3` operators to :math:`2 \times 2`
operators:

.. doctest:: [tensor]

  >>> tensor_contract(composite(to_super(A), to_super(B)), (1, 3), (4, 6)).dims
  [[[2], [2]], [[3], [3]]]


..
    TODO: remake from notebook to tutorials
    .. _channel contraction tutorial: github/qutip/qutip-notebooks/blob/master/examples/superop-contract.ipynb
