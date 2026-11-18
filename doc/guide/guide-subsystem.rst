.. _guide-subsystem-apply:

******************************************
Applying Operators on Part of a System
******************************************

.. _subsystem-intro:

Introduction
============

A very common task in quantum simulation is to apply a small operator, such as
a gate, to a few subsystems of a much larger composite system.
For example, applying a CNOT to qubits 2 and 7 of a 10-qubit register.

The naive approach is to build the full :math:`2^{10} \times 2^{10}` operator
by taking the tensor product with identities, then multiply it with the state.
This works, but the cost grows quickly with the number of subsystems, and for
a single application most of that work is wasted on identities.

QuTiP provides several tools to perform such operations more efficiently.
They differ in what they accept (kets, density matrices, superoperators),
how they perform the computation (sparse, dense, Python or Cython) and how
mature the interface is.
This guide describes each of them, and gives some advice on choosing one.

All examples assume the following imports:

.. testcode::

    from qutip import (
        basis, tensor, qeye, sigmax, rand_ket, rand_dm, expand_operator, gates,
        einsum, local_matmul, subsystem_apply, to_super
    )

and a small system of four qubits, with a two-qubit gate acting on qubits 1
and 3 (counting from 0):

.. testcode::

    N = 4
    psi = rand_ket([2] * N)
    rho = rand_dm([2] * N)

    gate = gates.cnot()
    targets = [1, 3]

.. _subsystem-expand:

Expanding the operator: :func:`.expand_operator`
================================================

:func:`.expand_operator` builds the full-size operator by padding the small
operator with identities and permuting the subsystems so that it acts on the
requested ``targets``:

.. testcode::

    full_gate = expand_operator(gate, dims=psi.dims[0], targets=targets)
    psi_out = full_gate @ psi
    rho_out = full_gate @ rho @ full_gate.dag()

Things to note:

- ``dims`` is the flat list of the dimension of each subsystem of the large
  system, e.g. ``[2, 3, 4, 5]`` (not the dims of the operator).
- ``targets`` is an integer or a list of integers. It need not be sorted nor
  adjacent. The order of ``targets`` defines which subsystem each factor of
  the operator acts on: for an operator ``tensor(oper1, oper2)``,
  ``targets=[2, 0]`` puts ``oper1`` on subsystem 2 and ``oper2`` on
  subsystem 0.
- Subsystems may have different dimensions (qubits, oscillators, ...), as long
  as the dimensions of the operator match those of the targeted subsystems.
- The operator must be a square operator with pure (tensor product) dims;
  superoperators are not supported.
- The result is a :class:`.Qobj` of the full size, stored with the ``dtype``
  given (the default sparse type, CSR, if not specified).
- A :class:`.QobjEvo` is also accepted, so time-dependent operators can be
  expanded as well.

This is equivalent to the explicit construction using :func:`.tensor`:

.. testcode::

    # For targets [1, 3] on 4 qubits, this is what expand_operator builds:
    manual = (
        tensor(qeye(2), basis(2, 0).proj(), qeye(2), qeye(2))
        + tensor(qeye(2), basis(2, 1).proj(), qeye(2), sigmax())
    )
    assert manual == full_gate

Internally, the operator is tensored with the identities of the remaining
subsystems and the result is permuted so that it acts on the requested
``targets``.

**When to use it:** when the same operator is applied many times
(time evolution, repeated gates), when you need the operator itself as a
:class:`.Qobj` (e.g. in a Hamiltonian).

.. note::

    By default :func:`expand_operator` creates an operator in ``CSR`` format.
    This format is optimized for ``CSR @ Dense`` operations, but is quite slow for
    ``Dense @ CSR`` operations. Avoiding that operation, (for example, by using
    ``full_gate @ (full_gate @ rho).dag()`` instead) is recommended.

.. _subsystem-einsum:

Einstein summation: :func:`.einsum`
===================================

:func:`.einsum` is the :class:`.Qobj` counterpart of :func:`numpy.einsum`.
It contracts the indices of the operands directly, so the small
operator is never expanded to the full size of the system.

How subscripts map to a :class:`.Qobj`
--------------------------------------

Each subscript string describes one operand. It has one label per subsystem
of the row ``dims``, followed by one label per subsystem of the column ``dims``.

- A ket of four qubits has ``dims = [[2, 2, 2, 2], [1]]``,
  so it takes 5 labels: 4 for the qubits and 1 for the trivial columns.
- A density matrix of four qubits has ``dims = [[2, 2, 2, 2], [2, 2, 2, 2]]``,
  and takes 8 labels.
- A two-qubit gate has ``dims = [[2, 2], [2, 2]]`` and takes 4 labels.

Labels shared between operands (and absent from the output) are summed over.


Applying a gate to a ket
------------------------

To apply ``gate`` to qubits 1 and 3 of ``psi``, the gate's column labels are
the labels of the state's qubits 1 and 3 (``b`` and ``d``), and its row labels
(``x`` and ``y``) replace them in the output:

.. testcode::

    psi_out = einsum("xybd,abcde->axcye", gate, psi)
    assert psi_out == full_gate @ psi

Applying a gate to a density matrix
-----------------------------------

For :math:`\rho \to G \rho G^\dagger`, the three operands are contracted at
once. The labels ``f`` and ``h`` are the column indices of ``rho`` for qubits 1
and 3:

.. testcode::

    rho_out = einsum("xybd,abcdefgh,fhXY->axcyeXgY", gate, rho, gate.dag())
    assert rho_out == full_gate @ rho @ full_gate.dag()


Rules and limitations
---------------------

- Only multiplication and contraction are supported.
  Contracting a row index with a row index (or a column with a column)
  would require an implicit transpose, and raises a :class:`ValueError`.
  Apply ``.trans()`` or ``.dag()`` to the operand explicitly.
- In the output, all indices coming from row dimensions must come before
  the ones coming from column dimensions.
  Pure reordering of a single operand (no contraction) is not supported;
  use :meth:`.Qobj.permute` instead.
- Superoperators and operator-kets are not supported. Use
  :func:`.local_matmul` to apply a superoperator to a few modes of a density
  matrix.
- If all indices are contracted, a complex scalar is returned instead of
  a :class:`.Qobj`.
- The output dimensions are inferred from the subscripts. They can be
  provided with ``out_dims`` if needed.
- The computation is done on dense arrays using numpy.einsum.
  The sparsity of the operators is not exploited and the memory footprint
  is that of the dense state.

**When to use it:** applying a few-body operator to a state in a
dense-state workflow, as in quantum-circuit simulation (the function was
designed with ``qutip-qip`` in mind), or any contraction involving several
operators that is awkward to write as a chain of tensor products.


.. _subsystem-local-matmul:

Local product: :func:`.local_matmul`
====================================

:func:`.local_matmul` multiplies a small ``operator`` into the given ``modes``
of a larger ``state``, using a Cython implementation that keeps the operator
sparse and never builds the full-size operator. The state is handled as a
dense array.

.. testcode::

    psi_out = local_matmul(gate, psi, modes=targets)
    assert psi_out == full_gate @ psi

For a density matrix, :math:`G \rho G^\dagger` is obtained with two calls.
The ``dual=True`` option applies the operator from the right
(``state @ operator``):

.. testcode::

    rho_out = local_matmul(gate, rho, modes=targets)
    rho_out = local_matmul(gate.dag(), rho_out, modes=targets, dual=True)
    assert rho_out == full_gate @ rho @ full_gate.dag()

Superoperators acting on the modes of a density matrix are supported as well.
The pre- and post-multiplication parts of the superoperator are applied to the
same modes, and ``dual`` cannot be used with them:

.. testcode::

    rho_out = local_matmul(to_super(gate), rho, modes=targets)
    assert rho_out == full_gate @ rho @ full_gate.dag()

The single call with the superoperator is faster than two calls once the system
is large enough.


Things to note:

- ``modes`` is an integer or a list of integers. The number of modes must match
  the number of subsystems of the operator, and the operator dimensions must
  match those of the state on these modes. To apply the same single-qubit gate
  to several modes, tensor it first: ``local_matmul(tensor(X, X), state, [0, 1])``.
- The operator does not need to be square: the dimensions of the modes can
  change, and the output dims are updated accordingly.
- Any state with consistent dims is accepted: kets, density matrices
  (in operator form, not operator-ket), rectangular operators, etc.
- Operator and state must have pure dims. Spaces that are not a tensor
  product (e.g. ``ENR`` states or direct sums) raise
  :class:`NotImplementedError`.
- Incorrect mode indices or number of modes raise a :class:`ValueError`,
  and mismatched dimensions raise a :class:`TypeError`.
- Optimized for single or consecutive modes.
- Usually the fastest when the gate is applied only once.

**When to use it:** applying a (possibly large and sparse) multi-mode operator
to a dense state without building the full operator, for rectangular
operators, or when a superoperator acts on a few modes of a density matrix.
Compared to :func:`.expand_operator`, there is no construction step; compared to
:func:`.einsum`, the operator stays sparse.


.. _subsystem-legacy:

Legacy function: ``subsystem_apply``
=========================================

``subsystem_apply`` predates the other tools.
It has a few peculiarities that make it a poor fit for the "apply a gate to some qubits" task:

.. testcode::

    # Apply a single-qubit channel on qubits 0 and 2.
    sx = to_super(sigmax())
    rho_out = subsystem_apply(rho, sx, mask=[True, False, True, False])

- The ``mask`` selects *which* subsystems are affected, with one boolean per
  subsystem. The ``channel`` must act on a **single** subsystem, and is applied
  separately to each selected subsystem. It cannot apply two-qubit operators
  such as the cnot gate in other examples.
- ``channel`` can be an operator (unitary) or a superoperator.
- The input can be a ket or a density matrix, but the output is always a
  density matrix (a ket is converted with ``state.proj()``).
- The implementation works on dense blocks in Python loops,
  so it is slow for large systems.

Its strength is the ability to apply a *channel* to multiple selected subsystems
at once.

.. note::

    ``subsystem_apply`` is a legacy function, kept for backward compatibility.
    It is not part of the documented API and may be removed in a future
    release.


.. _subsystem-choosing:

Which one should I use?
=======================

.. list-table::
    :header-rows: 1
    :widths: 20 20 20 40

    * - Function
      - Input
      - Computation
      - Typical use
    * - :func:`.expand_operator`
      - operator, any ``targets``
      - sparse, builds the full operator
      - Building quantum system, repeated application
    * - :func:`.einsum`
      - Any non-super
      - dense, no full-size operator
      - Circuit-like simulation, custom contractions
    * - :func:`.local_matmul`
      - Any operator, non-super operator state
      - sparse operator, dense state, Cython
      - Fast target mode matmul.
    * - :func:`.subsystem_apply`
      - density matrix
      - dense blocks, Python
      - Single-subsystem channels given as superoperators (legacy)

As a rule of thumb:

- If you need the big operator itself, or reuse it many times: use :func:`.expand_operator`.
- If operating on multiple objects at once, or contraction is not limited to matrix multiplication: use
  :func:`.einsum`.
- If applying an operator to a dense state, or a superoperator on a few
  modes of a density matrix: use ``local_matmul``.
