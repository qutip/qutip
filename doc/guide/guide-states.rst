.. _states:

*************************************
Manipulating States and Operators
*************************************

.. _states-intro:

Introduction
=================

In the previous guide section :ref:`basics`, we saw how to create states and operators, using the functions built into QuTiP. In this portion of the guide, we will look at performing basic operations with states and operators.  For more detailed demonstrations on how to use and manipulate these objects, see the examples on the `tutorials <https://qutip.org/tutorials.html>`_ web page.


.. _states-vectors:

State Vectors (kets or bras)
==============================

Here we begin by creating a Fock :func:`.basis` vacuum state vector :math:`\left|0\right>` with in a Hilbert space with 5 number states, from 0 to 4:

.. testcode:: [states]

    vac = basis(5, 0)

    print(vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[1.]
     [0.]
     [0.]
     [0.]
     [0.]]




and then create a lowering operator :math:`\left(\hat{a}\right)` corresponding to 5 number states using the :func:`.destroy` function:

.. testcode:: [states]

    a = destroy(5)

    print(a)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = False
    Qobj data =
    [[0.         1.         0.         0.         0.        ]
     [0.         0.         1.41421356 0.         0.        ]
     [0.         0.         0.         1.73205081 0.        ]
     [0.         0.         0.         0.         2.        ]
     [0.         0.         0.         0.         0.        ]]


Now lets apply the destruction operator to our vacuum state ``vac``,


.. testcode:: [states]

    print(a * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
     [0.]
     [0.]
     [0.]
     [0.]]

We see that, as expected, the vacuum is transformed to the zero vector.  A more interesting example comes from using the adjoint of the lowering operator, the raising operator :math:`\hat{a}^\dagger`:

.. testcode:: [states]

    print(a.dag() * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
    [1.]
    [0.]
    [0.]
    [0.]]

The raising operator has in indeed raised the state `vec` from the vacuum to the :math:`\left| 1\right>` state.
Instead of using the dagger ``Qobj.dag()`` method to raise the state, we could have also used the built in :func:`.create` function to make a raising operator:

.. testcode:: [states]

    c = create(5)

    print(c * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
     [1.]
     [0.]
     [0.]
     [0.]]

which does the same thing.  We can raise the vacuum state more than once by successively apply the raising operator:

.. testcode:: [states]

    print(c * c * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.        ]
     [0.        ]
     [1.41421356]
     [0.        ]
     [0.        ]]

or just taking the square of the raising operator :math:`\left(\hat{a}^\dagger\right)^{2}`:

.. testcode:: [states]

    print(c ** 2 * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.        ]
     [0.        ]
     [1.41421356]
     [0.        ]
     [0.        ]]

Applying the raising operator twice gives the expected :math:`\sqrt{n + 1}` dependence.  We can use the product of :math:`c * a` to also apply the number operator to the state vector ``vac``:

.. testcode:: [states]

    print(c * a * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
     [0.]
     [0.]
     [0.]
     [0.]]

or on the :math:`\left| 1\right>` state:

.. testcode:: [states]

    print(c * a * (c * vac))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
     [1.]
     [0.]
     [0.]
     [0.]]

or the :math:`\left| 2\right>` state:

.. testcode:: [states]

    print(c * a * (c**2 * vac))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.        ]
     [0.        ]
     [2.82842712]
     [0.        ]
     [0.        ]]

Notice how in this last example, application of the number operator does not give the expected value :math:`n=2`, but rather :math:`2\sqrt{2}`.  This is because this last state is not normalized to unity as :math:`c\left| n\right> = \sqrt{n+1}\left| n+1\right>`.  Therefore, we should normalize our vector first:

.. testcode:: [states]

    print(c * a * (c**2 * vac).unit())

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
     [0.]
     [2.]
     [0.]
     [0.]]

Since we are giving a demonstration of using states and operators, we have done a lot more work than we should have.
For example, we do not need to operate on the vacuum state to generate a higher number Fock state.
Instead we can use the :func:`.basis` (or :func:`.fock`) function to directly obtain the required state:

.. testcode:: [states]

    ket = basis(5, 2)

    print(ket)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
     [0.]
     [1.]
     [0.]
     [0.]]

Notice how it is automatically normalized.  We can also use the built in :func:`.num` operator:

.. testcode:: [states]

    n = num(5)

    print(n)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 2. 0. 0.]
     [0. 0. 0. 3. 0.]
     [0. 0. 0. 0. 4.]]

Therefore, instead of ``c * a * (c ** 2 * vac).unit()`` we have:

.. testcode:: [states]

    print(n * ket)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.]
     [0.]
     [2.]
     [0.]
     [0.]]

We can also create superpositions of states:

.. testcode:: [states]

    ket = (basis(5, 0) + basis(5, 1)).unit()

    print(ket)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.70710678]
     [0.70710678]
     [0.        ]
     [0.        ]
     [0.        ]]

where we have used the :meth:`.Qobj.unit` method to again normalize the state. Operating with the number function again:

.. testcode:: [states]

    print(n * ket)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[0.        ]
     [0.70710678]
     [0.        ]
     [0.        ]
     [0.        ]]

We can also create coherent states and squeezed states by applying the :func:`.displace` and :func:`.squeeze` functions to the vacuum state:

.. testcode:: [states]

    vac = basis(5, 0)

    d = displace(5, 1j)

    s = squeeze(5, np.complex(0.25, 0.25))

    print(d * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[ 0.60655682+0.j        ]
     [ 0.        +0.60628133j]
     [-0.4303874 +0.j        ]
     [ 0.        -0.24104351j]
     [ 0.14552147+0.j        ]]

.. testcode:: [states]

    print(d * s * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [1]], shape = (5, 1), type = ket
    Qobj data =
    [[ 0.65893786+0.08139381j]
     [ 0.10779462+0.51579735j]
     [-0.37567217-0.01326853j]
     [-0.02688063-0.23828775j]
     [ 0.26352814+0.11512178j]]

Of course, displacing the vacuum gives a coherent state, which can also be generated using the built in :func:`.coherent` function.


.. _states-dm:

Density matrices
=================

One of the main purpose of QuTiP is to explore the dynamics of **open** quantum systems, where the most general state of a system is no longer a state vector, but rather a density matrix.  Since operations on density matrices operate identically to those of vectors, we will just briefly highlight creating and using these structures.

The simplest density matrix is created by forming the outer-product :math:`\left|\psi\right>\left<\psi\right|` of a ket vector:

.. testcode:: [states]

    ket = basis(5, 2)

    print(ket * ket.dag())

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]

A similar task can also be accomplished via the :func:`.fock_dm` or :func:`.ket2dm` functions:

.. testcode:: [states]

    print(fock_dm(5, 2))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]

.. testcode:: [states]

    print(ket2dm(ket))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]

If we want to create a density matrix with equal classical probability of being found in the :math:`\left|2\right>` or :math:`\left|4\right>` number states we can do the following:

.. testcode:: [states]

    print(0.5 * ket2dm(basis(5, 4)) + 0.5 * ket2dm(basis(5, 2)))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[0.  0.  0.  0.  0. ]
     [0.  0.  0.  0.  0. ]
     [0.  0.  0.5 0.  0. ]
     [0.  0.  0.  0.  0. ]
     [0.  0.  0.  0.  0.5]]

or use ``0.5 * fock_dm(5, 2) + 0.5 * fock_dm(5, 4)``.
There are also several other built-in functions for creating predefined density matrices, for example :func:`.coherent_dm` and :func:`.thermal_dm` which create coherent state and thermal state density matrices, respectively.


.. testcode:: [states]

    print(coherent_dm(5, 1.25))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[0.20980701 0.26141096 0.23509686 0.15572585 0.13390765]
     [0.26141096 0.32570738 0.29292109 0.19402805 0.16684347]
     [0.23509686 0.29292109 0.26343512 0.17449684 0.1500487 ]
     [0.15572585 0.19402805 0.17449684 0.11558499 0.09939079]
     [0.13390765 0.16684347 0.1500487  0.09939079 0.0854655 ]]

.. testcode:: [states]

    print(thermal_dm(5, 1.25))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[0.46927974 0.         0.         0.         0.        ]
     [0.         0.26071096 0.         0.         0.        ]
     [0.         0.         0.14483942 0.         0.        ]
     [0.         0.         0.         0.08046635 0.        ]
     [0.         0.         0.         0.         0.04470353]]

QuTiP also provides a set of distance metrics for determining how close two density matrix distributions are to each other.
Included are the trace distance :func:`.tracedist`, fidelity :func:`.fidelity`, Hilbert-Schmidt distance :func:`.hilbert_dist`, Bures distance :func:`.bures_dist`, Bures angle :func:`.bures_angle`, and quantum Hellinger distance :func:`.hellinger_dist`.

.. testcode:: [states]

    x = coherent_dm(5, 1.25)

    y = coherent_dm(5, np.complex(0, 1.25))  # <-- note the 'j'

    z = thermal_dm(5, 0.125)

    np.testing.assert_almost_equal(fidelity(x, x), 1)

    np.testing.assert_almost_equal(hellinger_dist(x, y), 1.3819080728932833)

We also know that for two pure states, the trace distance (T) and the fidelity (F) are related by :math:`T = \sqrt{1 - F^{2}}`, while the quantum Hellinger distance (QHE) between two pure states :math:`\left|\psi\right>` and :math:`\left|\phi\right>` is given by :math:`QHE = \sqrt{2 - 2\left|\left<\psi | \phi\right>\right|^2}`.

.. testcode:: [states]

    np.testing.assert_almost_equal(tracedist(y, x), np.sqrt(1 - fidelity(y, x) ** 2))

For a pure state and a mixed state, :math:`1 - F^{2} \le T` which can also be verified:

.. testcode:: [states]

    assert 1 - fidelity(x, z) ** 2 < tracedist(x, z)

.. _states-qubit:

Qubit (two-level) systems
=========================

Having spent a fair amount of time on basis states that represent harmonic oscillator states, we now move on to qubit, or two-level quantum systems (for example a spin-1/2). To create a state vector corresponding to a qubit system, we use the same :func:`.basis`, or :func:`.fock`, function with only two levels:


.. testcode:: [states]

    spin = basis(2, 0)

Now at this point one may ask how this state is different than that of a harmonic oscillator in the vacuum state truncated to two energy levels?

.. testcode:: [states]

    vac = basis(2, 0)

At this stage, there is no difference.  This should not be surprising as we called the exact same function twice.  The difference between the two comes from the action of the spin operators :func:`.sigmax`, :func:`.sigmay`, :func:`.sigmaz`, :func:`.sigmap`, and :func:`.sigmam` on these two-level states.  For example, if ``vac`` corresponds to the vacuum state of a harmonic oscillator, then, as we have already seen, we can use the raising operator to get the :math:`\left|1\right>` state:

.. testcode:: [states]

    print(vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[1.]
     [0.]]

.. testcode:: [states]

    c = create(2)

    print(c * vac)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[0.]
     [1.]]

For a spin system, the operator analogous to the raising operator is the sigma-plus operator :func:`.sigmap`.  Operating on the ``spin`` state gives:

.. testcode:: [states]

    print(spin)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[1.]
     [0.]]

.. testcode:: [states]

    print(sigmap() * spin)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[0.]
     [0.]]

Now we see the difference!  The :func:`.sigmap` operator acting on the ``spin`` state returns the zero vector.  Why is this?  To see what happened, let us use the :func:`.sigmaz` operator:

.. testcode:: [states]

    print(sigmaz())

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]

.. testcode:: [states]

    print(sigmaz() * spin)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[1.]
     [0.]]

.. testcode:: [states]

    spin2 = basis(2, 1)

    print(spin2)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[0.]
     [1.]]

.. testcode:: [states]

    print(sigmaz() * spin2)

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
  Qobj data =
  [[ 0.]
   [-1.]]

The answer is now apparent.  Since the QuTiP :func:`.sigmaz` function uses the standard z-basis representation of the sigma-z spin operator, the ``spin`` state corresponds to the :math:`\left|\uparrow\right>` state of a two-level spin system while ``spin2`` gives the :math:`\left|\downarrow\right>` state.  Therefore, in our previous example ``sigmap() * spin``, we raised the qubit state out of the truncated two-level Hilbert space resulting in the zero state.

While at first glance this convention might seem somewhat odd, it is in fact quite handy. For one, the spin operators remain in the conventional form. Second, when the spin system is in the :math:`\left|\uparrow\right>` state:

.. testcode:: [states]

    print(sigmaz() * spin)

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
  Qobj data =
  [[1.]
   [0.]]

the non-zero component is the zeroth-element of the underlying matrix (remember that python uses c-indexing, and matrices start with the zeroth element).  The :math:`\left|\downarrow\right>` state therefore has a non-zero entry in the first index position. This corresponds nicely with the quantum information definitions of qubit states, where the excited :math:`\left|\uparrow\right>` state is label as :math:`\left|0\right>`, and the :math:`\left|\downarrow\right>` state by :math:`\left|1\right>`.

If one wants to create spin operators for higher spin systems, then the :func:`.jmat` function comes in handy.

.. _quantum_gates:

Gates
=====

The pre-defined gates are shown in the table below:


.. cssclass:: table-striped

+------------------------------------------------+-------------------------------------------------------+
| Gate function                                  | Description                                           |
+================================================+=======================================================+
| :func:`~qutip.core.gates.rx`                   | Rotation around x axis                                |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.ry`                   | Rotation around y axis                                |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.rz`                   | Rotation around z axis                                |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.sqrtnot`              | Square root of not gate                               |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.sqrtnot`              | Square root of not gate                               |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.snot`                 | Hardmard gate                                         |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.phasegate`            | Phase shift gate                                      |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.qrot`                 | A qubit rotation under a Rabi pulse                   |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.cy_gate`              | Controlled y gate                                     |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.cz_gate`              | Controlled z gate                                     |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.s_gate`               | Single-qubit rotation                                 |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.t_gate`               | Square root of s gate                                 |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.cs_gate`              | Controlled s gate                                     |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.ct_gate`              | Controlled t gate                                     |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.cphase`               | Controlled phase gate                                 |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.cnot`                 | Controlled not gate                                   |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.csign`                | Same as cphase                                        |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.berkeley`             | Berkeley gate                                         |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.swapalpha`            | Swapalpha gate                                        |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.swap`                 | Swap the states of two qubits                         |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.iswap`                | Swap gate with additional phase for 01 and 10 states  |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.sqrtswap`             | Square root of the swap gate                          |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.sqrtiswap`            | Square root of the iswap gate                         |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.fredkin`              | Fredkin gate                                          |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.molmer_sorensen`      | Molmer Sorensen gate                                  |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.toffoli`              | Toffoli gate                                          |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.hadamard_transform`   | Hadamard gate                                         |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.qubit_clifford_group` | Generates the Clifford group on a single qubit        |
+------------------------------------------------+-------------------------------------------------------+
| :func:`~qutip.core.gates.globalphase`          | Global phase gate                                     |
+------------------------------------------------+-------------------------------------------------------+

To load this qutip module, first you have to import gates:

.. code-block:: Python

   from qutip import gates

For example to use the Hadamard Gate:

.. testcode:: [basics]

    H = gates.hadamard_transform()
    print(H)

**Output**:

.. testoutput:: [basics]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 0.70710678  0.70710678]
     [0.70710678 -0.70710678]]

.. _states-expect:

Expectation values
===================

Some of the most important information about quantum systems comes from calculating the expectation value of operators, both Hermitian and non-Hermitian, as the state or density matrix of the system varies in time.  Therefore, in this section we demonstrate the use of the :func:`.expect` function.  To begin:

.. testcode:: [states]

    vac = basis(5, 0)

    one = basis(5, 1)

    c = create(5)

    N = num(5)

    np.testing.assert_almost_equal(expect(N, vac), 0)

    np.testing.assert_almost_equal(expect(N, one), 1)

    coh = coherent_dm(5, 1.0j)

    np.testing.assert_almost_equal(expect(N, coh), 0.9970555745806597)

    cat = (basis(5, 4) + 1.0j * basis(5, 3)).unit()

    np.testing.assert_almost_equal(expect(c, cat), 0.9999999999999998j)


The :func:`.expect` function also accepts lists or arrays of state vectors or density matrices for the second input:

.. testcode:: [states]

    states = [(c**k * vac).unit() for k in range(5)]  # must normalize

    print(expect(N, states))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    [0. 1. 2. 3. 4.]

.. testcode:: [states]

    cat_list = [(basis(5, 4) + x * basis(5, 3)).unit() for x in [0, 1.0j, -1.0, -1.0j]]

    print(expect(c, cat_list))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    [ 0.+0.j  0.+1.j -1.+0.j  0.-1.j]

Notice how in this last example, all of the return values are complex numbers.  This is because the :func:`.expect` function looks to see whether the operator is Hermitian or not.  If the operator is Hermitian, then the output will always be real.  In the case of non-Hermitian operators, the return values may be complex.  Therefore, the :func:`.expect` function will return an array of complex values for non-Hermitian operators when the input is a list/array of states or density matrices.

Of course, the :func:`.expect` function works for spin states and operators:


.. testcode:: [states]

    up = basis(2, 0)

    down = basis(2, 1)

    np.testing.assert_almost_equal(expect(sigmaz(), up), 1)

    np.testing.assert_almost_equal(expect(sigmaz(), down), -1)


as well as the composite objects discussed in the next section :ref:`tensor`:

.. testcode:: [states]

    spin1 = basis(2, 0)

    spin2 = basis(2, 1)

    two_spins = tensor(spin1, spin2)

    sz1 = tensor(sigmaz(), qeye(2))

    sz2 = tensor(qeye(2), sigmaz())

    np.testing.assert_almost_equal(expect(sz1, two_spins), 1)

    np.testing.assert_almost_equal(expect(sz2, two_spins), -1)


.. _states-super:

Superoperators and Vectorized Operators
=======================================

In addition to state vectors and density operators, QuTiP allows for
representing maps that act linearly on density operators using the Kraus,
Liouville supermatrix and Choi matrix formalisms. This support is based on the
correspondence between linear operators acting on a Hilbert space, and vectors
in two copies of that Hilbert space,
:math:`\mathrm{vec} : \mathcal{L}(\mathcal{H}) \to \mathcal{H} \otimes \mathcal{H}`
[Hav03]_, [Wat13]_.

This isomorphism is implemented in QuTiP by the
:obj:`.operator_to_vector` and
:obj:`.vector_to_operator` functions:

.. testcode:: [states]

    psi = basis(2, 0)

    rho = ket2dm(psi)

    print(rho)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[1. 0.]
     [0. 0.]]

.. testcode:: [states]

    vec_rho = operator_to_vector(rho)

    print(vec_rho)

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[[2], [2]], [1]], shape = (4, 1), type = operator-ket
  Qobj data =
  [[1.]
   [0.]
   [0.]
   [0.]]

.. testcode:: [states]

    rho2 = vector_to_operator(vec_rho)

    np.testing.assert_almost_equal((rho - rho2).norm(), 0)

The :attr:`.Qobj.type` attribute indicates whether a quantum object is
a vector corresponding to an operator (``operator-ket``), or its Hermitian
conjugate (``operator-bra``).

Note that QuTiP uses the *column-stacking* convention for the isomorphism
between :math:`\mathcal{L}(\mathcal{H})` and :math:`\mathcal{H} \otimes \mathcal{H}`:

.. testcode:: [states]

    A = Qobj(np.arange(4).reshape((2, 2)))

    print(A)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = False
    Qobj data =
    [[0. 1.]
     [2. 3.]]

.. testcode:: [states]

    print(operator_to_vector(A))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [1]], shape = (4, 1), type = operator-ket
    Qobj data =
    [[0.]
     [2.]
     [1.]
     [3.]]

Since :math:`\mathcal{H} \otimes \mathcal{H}` is a vector space, linear maps
on this space can be represented as matrices, often called *superoperators*.
Using the :obj:`.Qobj`, the :obj:`.spre` and :obj:`.spost` functions, supermatrices
corresponding to left- and right-multiplication respectively can be quickly
constructed.

.. testcode:: [states]

    X = sigmax()

    S = spre(X) * spost(X.dag()) # Represents conjugation by X.

Note that this is done automatically by the :obj:`.to_super` function when given
``type='oper'`` input.

.. testcode:: [states]

    S2 = to_super(X)

    np.testing.assert_almost_equal((S - S2).norm(), 0)

Quantum objects representing superoperators are denoted by ``type='super'``:

.. testcode:: [states]

  print(S)

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True
  Qobj data =
  [[0. 0. 0. 1.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [1. 0. 0. 0.]]

Information about superoperators, such as whether they represent completely
positive maps, is exposed through the :attr:`.Qobj.iscp`, :attr:`.Qobj.istp`
and :attr:`.Qobj.iscptp` attributes:

.. testcode:: [states]

    print(S.iscp, S.istp, S.iscptp)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    True True True

In addition, dynamical generators on this extended space, often called
*Liouvillian superoperators*, can be created using the :func:`.liouvillian` function. Each of these takes a Hamiltonian along with
a list of collapse operators, and returns a ``type="super"`` object that can
be exponentiated to find the superoperator for that evolution.

.. testcode:: [states]

    H = 10 * sigmaz()

    c1 = destroy(2)

    L = liouvillian(H, [c1])

    print(L)

    S = (12 * L).expm()

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = False
    Qobj data =
    [[ 0.  +0.j  0.  +0.j  0.  +0.j  1.  +0.j]
     [ 0.  +0.j -0.5+20.j  0.  +0.j  0.  +0.j]
     [ 0.  +0.j  0.  +0.j -0.5-20.j  0.  +0.j]
     [ 0.  +0.j  0.  +0.j  0.  +0.j -1.  +0.j]]

For qubits, a particularly useful way to visualize superoperators is to plot them in the Pauli basis,
such that :math:`S_{\mu,\nu} = \langle\!\langle \sigma_{\mu} | S[\sigma_{\nu}] \rangle\!\rangle`. Because
the Pauli basis is Hermitian, :math:`S_{\mu,\nu}` is a real number for all Hermitian-preserving superoperators
:math:`S`,
allowing us to plot the elements of :math:`S` as a `Hinton diagram <https://matplotlib.org/examples/specialty_plots/hinton_demo.html>`_. In such diagrams, positive elements are indicated by white squares, and negative elements
by black squares. The size of each element is indicated by the size of the corresponding square. For instance,
let :math:`S[\rho] = \sigma_x \rho \sigma_x^{\dagger}`. Then :math:`S[\sigma_{\mu}] = \sigma_{\mu} \cdot \begin{cases} +1 & \mu = 0, x \\ -1 & \mu = y, z \end{cases}`. We can quickly see this by noting that the :math:`Y` and :math:`Z` elements
of the Hinton diagram for :math:`S` are negative:

.. plot::

    from qutip import *
    settings.colorblind_safe = True

    import matplotlib.pyplot as plt
    plt.rcParams['savefig.transparent'] = True

    X = sigmax()
    S = spre(X) * spost(X.dag())

    hinton(S)

Choi, Kraus, Stinespring and :math:`\chi` Representations
=========================================================

In addition to the superoperator representation of quantum maps, QuTiP
supports several other useful representations. First, the Choi matrix
:math:`J(\Lambda)` of a quantum map :math:`\Lambda` is useful for working with
ancilla-assisted process tomography (AAPT), and for reasoning about properties
of a map or channel. Up to normalization, the Choi matrix is defined by acting
:math:`\Lambda` on half of an entangled pair. In the column-stacking
convention,

.. math::

    J(\Lambda) = (\mathbb{1} \otimes \Lambda) [|\mathbb{1}\rangle\!\rangle \langle\!\langle \mathbb{1}|].

In QuTiP, :math:`J(\Lambda)` can be found by calling the :func:`.to_choi`
function on a ``type="super"`` :obj:`.Qobj`.

.. testcode:: [states]

    X = sigmax()

    S = sprepost(X, X)

    J = to_choi(S)

    print(J)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True, superrep = choi
    Qobj data =
    [[0. 0. 0. 0.]
     [0. 1. 1. 0.]
     [0. 1. 1. 0.]
     [0. 0. 0. 0.]]

.. testcode:: [states]

  print(to_choi(spre(qeye(2))))

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True, superrep = choi
  Qobj data =
  [[1. 0. 0. 1.]
   [0. 0. 0. 0.]
   [0. 0. 0. 0.]
   [1. 0. 0. 1.]]

If a :obj:`.Qobj` instance is already in the Choi :attr:`.Qobj.superrep`, then calling :func:`.to_choi`
does nothing:

.. testcode:: [states]

    print(to_choi(J))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True, superrep = choi
    Qobj data =
    [[0. 0. 0. 0.]
     [0. 1. 1. 0.]
     [0. 1. 1. 0.]
     [0. 0. 0. 0.]]

To get back to the superoperator representation, simply use the :func:`.to_super` function.
As with :func:`.to_choi`, :func:`.to_super` is idempotent:

.. testcode:: [states]

    print(to_super(J) - S)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True
    Qobj data =
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]

.. testcode:: [states]

    print(to_super(S))

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True
    Qobj data =
    [[0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [1. 0. 0. 0.]]

We can quickly obtain another useful representation from the Choi matrix by taking its eigendecomposition.
In particular, let :math:`\{A_i\}` be a set of operators such that
:math:`J(\Lambda) = \sum_i |A_i\rangle\!\rangle \langle\!\langle A_i|`.
We can write :math:`J(\Lambda)` in this way
for any hermicity-preserving map; that is, for any map :math:`\Lambda` such that :math:`J(\Lambda) = J^\dagger(\Lambda)`.
These operators then form the Kraus representation of :math:`\Lambda`. In particular, for any input :math:`\rho`,

.. math::

    \Lambda(\rho) = \sum_i A_i \rho A_i^\dagger.

Notice using the column-stacking identity that :math:`(C^\mathrm{T} \otimes A) |B\rangle\!\rangle = |ABC\rangle\!\rangle`,
we have that

.. math::

      \sum_i (\mathbb{1} \otimes A_i) (\mathbb{1} \otimes A_i)^\dagger |\mathbb{1}\rangle\!\rangle \langle\!\langle\mathbb{1}|
    = \sum_i |A_i\rangle\!\rangle \langle\!\langle A_i| = J(\Lambda).

The Kraus representation of a hermicity-preserving map can be found in QuTiP
using the :func:`.to_kraus` function.

.. testcode:: [states]

    del sum # np.sum overwrote sum and caused a bug.


.. testcode:: [states]

    I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

.. testcode:: [states]

    S = sum([sprepost(P, P) for P in (I, X, Y, Z)]) / 4
    print(S)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True
    Qobj data =
    [[0.5 0.  0.  0.5]
     [0.  0.  0.  0. ]
     [0.  0.  0.  0. ]
     [0.5 0.  0.  0.5]]

.. testcode:: [states]

    J = to_choi(S)
    print(J)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True, superrep = choi
    Qobj data =
    [[0.5 0.  0.  0. ]
     [0.  0.5 0.  0. ]
     [0.  0.  0.5 0. ]
     [0.  0.  0.  0.5]]

.. testcode:: [states]

    print(J.eigenstates()[1])

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    [Quantum object: dims = [[[2], [2]], [1, 1]], shape = (4, 1), type = operator-ket
    Qobj data =
    [[1.]
     [0.]
     [0.]
     [0.]]
     Quantum object: dims = [[[2], [2]], [1, 1]], shape = (4, 1), type = operator-ket
    Qobj data =
    [[0.]
     [1.]
     [0.]
     [0.]]
     Quantum object: dims = [[[2], [2]], [1, 1]], shape = (4, 1), type = operator-ket
    Qobj data =
    [[0.]
     [0.]
     [1.]
     [0.]]
     Quantum object: dims = [[[2], [2]], [1, 1]], shape = (4, 1), type = operator-ket
    Qobj data =
    [[0.]
     [0.]
     [0.]
     [1.]]]

.. testcode:: [states]

    K = to_kraus(S)
    print(K)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    [Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0.70710678 0.        ]
     [0.         0.        ]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = False
    Qobj data =
    [[0.         0.        ]
     [0.70710678 0.        ]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = False
    Qobj data =
    [[0.         0.70710678]
     [0.         0.        ]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0.         0.        ]
     [0.         0.70710678]]]

As with the other representation conversion functions, :func:`.to_kraus`
checks the :attr:`.Qobj.superrep` attribute of its input, and chooses an appropriate
conversion method. Thus, in the above example, we can also call :func:`.to_kraus`
on ``J``.

.. testcode:: [states]

    KJ = to_kraus(J)
    print(KJ)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    [Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0.70710678 0.        ]
     [0.         0.        ]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = False
    Qobj data =
    [[0.         0.        ]
     [0.70710678 0.        ]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = False
    Qobj data =
    [[0.         0.70710678]
     [0.         0.        ]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0.         0.        ]
     [0.         0.70710678]]]

.. testcode:: [states]

    for A, AJ in zip(K, KJ):
      print(A - AJ)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0. 0.]
     [0. 0.]]
    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0. 0.]
     [0. 0.]]
    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0. 0.]
     [0. 0.]]
    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0. 0.]
     [0. 0.]]

The Stinespring representation is closely related to the Kraus representation,
and consists of a pair of operators :math:`A` and :math:`B` such that for
all operators :math:`X` acting on :math:`\mathcal{H}`,

.. math::

    \Lambda(X) = \operatorname{Tr}_2(A X B^\dagger),

where the partial trace is over a new index that corresponds to the
index in the Kraus summation. Conversion to Stinespring
is handled by the :func:`.to_stinespring`
function.

.. testcode:: [states]

    a = create(2).dag()

    S_ad = sprepost(a * a.dag(), a * a.dag()) + sprepost(a, a.dag())
    S = 0.9 * sprepost(I, I) + 0.1 * S_ad

    print(S)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = False
    Qobj data =
    [[1.  0.  0.  0.1]
     [0.  0.9 0.  0. ]
     [0.  0.  0.9 0. ]
     [0.  0.  0.  0.9]]

.. testcode:: [states]

    A, B = to_stinespring(S)
    print(A)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 3], [2]], shape = (6, 2), type = oper, isherm = False
    Qobj data =
    [[-0.98845443  0.        ]
     [ 0.          0.31622777]
     [ 0.15151842  0.        ]
     [ 0.         -0.93506452]
     [ 0.          0.        ]
     [ 0.         -0.16016975]]

.. testcode:: [states]

    print(B)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[2, 3], [2]], shape = (6, 2), type = oper, isherm = False
    Qobj data =
    [[-0.98845443  0.        ]
     [ 0.          0.31622777]
     [ 0.15151842  0.        ]
     [ 0.         -0.93506452]
     [ 0.          0.        ]
     [ 0.         -0.16016975]]

Notice that a new index has been added, such that :math:`A` and :math:`B`
have dimensions ``[[2, 3], [2]]``, with the length-3 index representing the
fact that the Choi matrix is rank-3 (alternatively, that the map has three
Kraus operators).

.. testcode:: [states]

    to_kraus(S)
    print(to_choi(S).eigenenergies())

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    [0.         0.04861218 0.1        1.85138782]

Finally, the last superoperator representation supported by QuTiP is
the :math:`\chi`-matrix representation,

.. math::

    \Lambda(\rho) = \sum_{\alpha,\beta} \chi_{\alpha,\beta} B_{\alpha} \rho B_{\beta}^\dagger,

where :math:`\{B_\alpha\}` is a basis for the space of matrices acting
on :math:`\mathcal{H}`. In QuTiP, this basis is taken to be the Pauli
basis :math:`B_\alpha = \sigma_\alpha / \sqrt{2}`. Conversion to the
:math:`\chi` formalism is handled by the :func:`.to_chi`
function.

.. testcode:: [states]

    chi = to_chi(S)
    print(chi)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    Quantum object: dims = [[[2], [2]], [[2], [2]]], shape = (4, 4), type = super, isherm = True, superrep = chi
    Qobj data =
    [[3.7+0.j  0. +0.j  0. +0.j  0.1+0.j ]
     [0. +0.j  0.1+0.j  0. +0.1j 0. +0.j ]
     [0. +0.j  0. -0.1j 0.1+0.j  0. +0.j ]
     [0.1+0.j  0. +0.j  0. +0.j  0.1+0.j ]]


One convenient property of the :math:`\chi` matrix is that the average
gate fidelity with the identity map can be read off directly from
the :math:`\chi_{00}` element:

.. testcode:: [states]

    np.testing.assert_almost_equal(average_gate_fidelity(S), 0.9499999999999998)

    print(chi[0, 0] / 4)

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    (0.925+0j)

Here, the factor of 4 comes from the dimension of the underlying
Hilbert space :math:`\mathcal{H}`. As with the superoperator
and Choi representations, the :math:`\chi` representation is
denoted by the :attr:`.Qobj.superrep`, such that :func:`.to_super`,
:func:`.to_choi`, :func:`.to_kraus`,
:func:`.to_stinespring` and :func:`.to_chi`
all convert from the :math:`\chi` representation appropriately.

Properties of Quantum Maps
==========================

In addition to converting between the different representations of quantum maps,
QuTiP also provides attributes to make it easy to check if a map is completely
positive, trace preserving and/or hermicity preserving. Each of these attributes
uses :attr:`.Qobj.superrep` to automatically perform any needed conversions.

In particular, a quantum map is said to be positive (but not necessarily completely
positive) if it maps all positive operators to positive operators. For instance, the
transpose map :math:`\Lambda(\rho) = \rho^{\mathrm{T}}` is a positive map. We run into
problems, however, if we tensor :math:`\Lambda` with the identity to get a partial
transpose map.

.. testcode:: [states]

    rho = ket2dm(bell_state())
    rho_out = partial_transpose(rho, [0, 1])
    print(rho_out.eigenenergies())

**Output**:

.. testoutput:: [states]
    :options: +NORMALIZE_WHITESPACE

    [-0.5  0.5  0.5  0.5]

Notice that even though we started with a positive map, we got an operator out
with negative eigenvalues. Complete positivity addresses this by requiring that
a map returns positive operators for all positive operators, and does so even
under tensoring with another map. The Choi matrix is very useful here, as it
can be shown that a map is completely positive if and only if its Choi matrix
is positive [Wat13]_. QuTiP implements this check with the :attr:`.Qobj.iscp`
attribute. As an example, notice that the snippet above already calculates
the Choi matrix of the transpose map by acting it on half of an entangled
pair. We simply need to manually set the ``dims`` and ``superrep`` attributes to reflect the
structure of the underlying Hilbert space and the chosen representation.

.. testcode:: [states]

    J = rho_out
    J.dims = [[[2], [2]], [[2], [2]]]
    J.superrep = 'choi'
    print(J.iscp)

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  False

This confirms that the transpose map is not completely positive. On the other hand,
the transpose map does satisfy a weaker condition, namely that it is hermicity preserving.
That is, :math:`\Lambda(\rho) = (\Lambda(\rho))^\dagger` for all :math:`\rho` such that
:math:`\rho = \rho^\dagger`. To see this, we note that :math:`(\rho^{\mathrm{T}})^\dagger
= \rho^*`, the complex conjugate of :math:`\rho`. By assumption, :math:`\rho = \rho^\dagger
= (\rho^*)^{\mathrm{T}}`, though, such that :math:`\Lambda(\rho) = \Lambda(\rho^\dagger) = \rho^*`.
We can confirm this by checking the :attr:`.Qobj.ishp` attribute:

.. testcode:: [states]

    print(J.ishp)

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  True

Next, we note that the transpose map does preserve the trace of its inputs, such that
:math:`\operatorname{Tr}(\Lambda[\rho]) = \operatorname{Tr}(\rho)` for all :math:`\rho`.
This can be confirmed by the :attr:`.Qobj.istp` attribute:

.. testcode:: [states]

    print(J.istp)

**Output**:

.. testoutput:: [states]
  :options: +NORMALIZE_WHITESPACE

  False

Finally, a map is called a quantum channel if it always maps valid states to valid
states. Formally, a map is a channel if it is both completely positive and trace preserving.
Thus, QuTiP provides a single attribute to quickly check that this is true.

.. doctest:: [states]

    >>> print(J.iscptp)
    False

    >>> print(to_super(qeye(2)).iscptp)
    True
