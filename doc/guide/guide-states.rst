.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson


.. _states:

*************************************
Manipulating States and Operators
*************************************

.. ipython::
   :suppress:

   In [1]: from qutip import *

.. _states-intro:

Introduction
=================

In the previous guide section :ref:`basics`, we saw how to create states and operators, using the functions built into QuTiP. In this portion of the guide, we will look at performing basic operations with states and operators.  For more detailed demonstrations on how to use and manipulate these objects, see the examples on the `tutorials <http://qutip.org/tutorials.html>`_ web page.


.. _states-vectors:

State Vectors (kets or bras)
==============================

Here we begin by creating a Fock :func:`qutip.states.basis` vacuum state vector :math:`\left|0\right>` with in a Hilbert space with 5 number states, from 0 to 4:

.. ipython::
 
    In [1]: vac = basis(5, 0)
    
    In [2]: vac


and then create a lowering operator :math:`\left(\hat{a}\right)` corresponding to 5 number states using the :func:`qutip.operators.destroy` function:

.. ipython::

    In [1]: a = destroy(5)
    
    In [2]: a


Now lets apply the destruction operator to our vacuum state ``vac``,


.. ipython::

    In [1]: a * vac


We see that, as expected, the vacuum is transformed to the zero vector.  A more interesting example comes from using the adjoint of the lowering operator, the raising operator :math:`\hat{a}^\dagger`:

.. ipython::

    In [1]: a.dag() * vac


The raising operator has in indeed raised the state `vec` from the vacuum to the :math:`\left| 1\right>` state.  Instead of using the dagger ``Qobj.dag()`` method to raise the state, we could have also used the built in :func:`qutip.operators.create` function to make a raising operator:

.. ipython::

    In [1]: c = create(5)
    
    In [2]: c * vac


which does the same thing.  We can raise the vacuum state more than once by successively apply the raising operator:

.. ipython::

    In [1]: c * c * vac


or just taking the square of the raising operator :math:`\left(\hat{a}^\dagger\right)^{2}`:

.. ipython::

    In [1]: c ** 2 * vac


Applying the raising operator twice gives the expected :math:`\sqrt{n + 1}` dependence.  We can use the product of :math:`c * a` to also apply the number operator to the state vector ``vac``:

.. ipython::

    In [1]: c * a * vac


or on the :math:`\left| 1\right>` state:

.. ipython::

    In [1]: c * a * (c * vac)


or the :math:`\left| 2\right>` state:

.. ipython::

    In [1]: c * a * (c**2 * vac)


Notice how in this last example, application of the number operator does not give the expected value :math:`n=2`, but rather :math:`2\sqrt{2}`.  This is because this last state is not normalized to unity as :math:`c\left| n\right> = \sqrt{n+1}\left| n+1\right>`.  Therefore, we should normalize our vector first:

.. ipython::

    In [1]: c * a * (c**2 * vac).unit()


Since we are giving a demonstration of using states and operators, we have done a lot more work than we should have.  For example, we do not need to operate on the vacuum state to generate a higher number Fock state.  Instead we can use the :func:`qutip.states.basis` (or :func:`qutip.states.fock`) function to directly obtain the required state:

.. ipython::

    In [1]: ket = basis(5, 2)
   
    In [2]: print(ket)


Notice how it is automatically normalized.  We can also use the built in :func:`qutip.operators.num` operator:

.. ipython::

    In [1]: n = num(5)
   
    In [2]: print(n)


Therefore, instead of ``c * a * (c ** 2 * vac).unit()`` we have:

.. ipython::

    In [1]: n * ket


We can also create superpositions of states:

.. ipython::

    In [1]: ket = (basis(5, 0) + basis(5, 1)).unit()
   
    In [2]: print(ket)


where we have used the :func:`qutip.Qobj.unit` method to again normalize the state. Operating with the number function again:

.. ipython::

    In [1]: n * ket


We can also create coherent states and squeezed states by applying the :func:`qutip.operators.displace` and :func:`qutip.operators.squeeze` functions to the vacuum state:

.. ipython::

    In [1]: vac = basis(5, 0)
  
    In [2]: d = displace(5, 1j)
   
    In [3]: s = squeeze(5, 0.25 + 0.25j)
   
    In [4]: d * vac


.. ipython::

    In [1]: d * s * vac


Of course, displacing the vacuum gives a coherent state, which can also be generated using the built in :func:`qutip.states.coherent` function.


.. _states-dm:

Density matrices
=================

One of the main purpose of QuTiP is to explore the dynamics of **open** quantum systems, where the most general state of a system is not longer a state vector, but rather a density matrix.  Since operations on density matrices operate identically to those of vectors, we will just briefly highlight creating and using these structures.

The simplest density matrix is created by forming the outer-product :math:`\left|\psi\right>\left<\psi\right|` of a ket vector:

.. ipython::

    In [1]: ket = basis(5, 2)
   
    In [2]: ket * ket.dag()

A similar task can also be accomplished via the :func:`qutip.states.fock_dm` or :func:`qutip.states.ket2dm` functions:

.. ipython::

    In [1]: fock_dm(5, 2)


.. ipython::

    In [1]: ket2dm(ket)


If we want to create a density matrix with equal classical probability of being found in the :math:`\left|2\right>` or :math:`\left|4\right>` number states we can do the following:

.. ipython::

    In [1]: 0.5 * ket2dm(basis(5, 4)) + 0.5 * ket2dm(basis(5, 2))


or use ``0.5 * fock_dm(5, 2) + 0.5 * fock_dm(5, 4)``. There are also several other built-in functions for creating predefined density matrices, for example :func:`qutip.states.coherent_dm` and :func:`qutip.states.thermal_dm` which create coherent state and thermal state density matrices, respectively.


.. ipython::

    In [1]: coherent_dm(5, 1.25)


.. ipython::

    In [1]: thermal_dm(5, 1.25)


QuTiP also provides a set of distance metrics for determining how close two density matrix distributions are to each other. Included are the trace distance :func:`qutip.metrics.tracedist`, fidelity :func:`qutip.metrics.fidelity`, Hilbert-Schmidt distance :func:`qutip.metrics.hilbert_dist`, Bures distance :func:`qutip.metrics.bures_dist`, and Bures angle :func:`qutip.metrics.bures_angle`.

.. ipython::

    In [1]: x = coherent_dm(5, 1.25)
    
    In [2]: y = coherent_dm(5, 1.25j)  # <-- note the 'j'
    
    In [3]: z = thermal_dm(5, 0.125)
    
    In [4]: fidelity(x, x)
    
    In [5]: tracedist(y, y)


We also know that for two pure states, the trace distance (T) and the fidelity (F) are related by :math:`T = \sqrt{1 - F^{2}}`.

.. ipython::

    In [1]: tracedist(y, x)

.. ipython::

    In [1]: np.sqrt(1 - fidelity(y, x) ** 2)


For a pure state and a mixed state, :math:`1 - F^{2} \le T` which can also be verified:

.. ipython::

    In [1]: 1 - fidelity(x, z) ** 2

.. ipython::

    In [1]: tracedist(x, z)


.. _states-qubit:

Qubit (two-level) systems
=========================

Having spent a fair amount of time on basis states that represent harmonic oscillator states, we now move on to qubit, or two-level quantum systems (for example a spin-1/2). To create a state vector corresponding to a qubit system, we use the same :func:`qutip.states.basis`, or :func:`qutip.states.fock`, function with only two levels:


.. ipython::

    In [1]: spin = basis(2, 0)

Now at this point one may ask how this state is different than that of a harmonic oscillator in the vacuum state truncated to two energy levels?

.. ipython::
    
    In [1]: vac = basis(2, 0)

At this stage, there is no difference.  This should not be surprising as we called the exact same function twice.  The difference between the two comes from the action of the spin operators :func:`qutip.operators.sigmax`, :func:`qutip.operators.sigmay`, :func:`qutip.operators.sigmaz`, :func:`qutip.operators.sigmap`, and :func:`qutip.operators.sigmam` on these two-level states.  For example, if ``vac`` corresponds to the vacuum state of a harmonic oscillator, then, as we have already seen, we can use the raising operator to get the :math:`\left|1\right>` state:

.. ipython::
    
    In [1]: vac

.. ipython::
    
    In [1]: c = create(2)
    
    In [2]: c * vac


For a spin system, the operator analogous to the raising operator is the sigma-plus operator :func:`qutip.operators.sigmap`.  Operating on the ``spin`` state gives:

.. ipython::
    
    In [1]: spin
    
    In [2]: sigmap() * spin

Now we see the difference!  The :func:`qutip.operators.sigmap` operator acting on the ``spin`` state returns the zero vector.  Why is this?  To see what happened, let us use the :func:`qutip.operators.sigmaz` operator:

.. ipython::
    
    In [1]: sigmaz()
    
    In [2]: sigmaz() * spin
    
    In [3]: spin2 = basis(2, 1)
    
    In [4]: spin2
    
    In [5]: sigmaz() * spin2


The answer is now apparent.  Since the QuTiP :func:`qutip.operators.sigmaz` function uses the standard z-basis representation of the sigma-z spin operator, the ``spin`` state corresponds to the :math:`\left|\uparrow\right>` state of a two-level spin system while ``spin2`` gives the :math:`\left|\downarrow\right>` state.  Therefore, in our previous example ``sigmap() * spin``, we raised the qubit state out of the truncated two-level Hilbert space resulting in the zero state.  

While at first glance this convention might seem somewhat odd, it is in fact quite handy. For one, the spin operators remain in the conventional form. Second, when the spin system is in the :math:`\left|\uparrow\right>` state:

.. ipython::
    
    In [1]: sigmaz() * spin

the non-zero component is the zeroth-element of the underlying matrix (remember that python uses c-indexing, and matrices start with the zeroth element).  The :math:`\left|\downarrow\right>` state therefore has a non-zero entry in the first index position. This corresponds nicely with the quantum information definitions of qubit states, where the excited :math:`\left|\uparrow\right>` state is label as :math:`\left|0\right>`, and the :math:`\left|\downarrow\right>` state by :math:`\left|1\right>`.

If one wants to create spin operators for higher spin systems, then the :func:`qutip.operators.jmat` function comes in handy. 

.. _states-expect:

Expectation values
===================

Some of the most important information about quantum systems comes from calculating the expectation value of operators, both Hermitian and non-Hermitian, as the state or density matrix of the system varies in time.  Therefore, in this section we demonstrate the use of the :func:`qutip.expect` function.  To begin:

.. ipython::
    
    In [1]: vac = basis(5, 0)
    
    In [2]: one = basis(5, 1)
    
    In [3]: c = create(5)
    
    In [4]: N = num(5)
    
    In [5]: expect(N, vac)
    
    In [6]: expect(N, one)


.. ipython::
    
    In [1]: coh = coherent_dm(5, 1.0j)
    
    In [2]: expect(N, coh)

.. ipython::
    
    In [1]: cat = (basis(5, 4) + 1.0j * basis(5, 3)).unit()
    
    In [2]: expect(c, cat)

The :func:`qutip.expect` function also accepts lists or arrays of state vectors or density matrices for the second input:

.. ipython::
    
    In [1]: states = [(c**k * vac).unit() for k in range(5)]  # must normalize

    In [2]: expect(N, states)

.. ipython::
    
    In [1]: cat_list = [(basis(5, 4) + x * basis(5, 3)).unit()
       ...:             for x in [0, 1.0j, -1.0, -1.0j]]
    
    In [2]: expect(c, cat_list)

Notice how in this last example, all of the return values are complex numbers.  This is because the :func:`qutip.expect` function looks to see whether the operator is Hermitian or not.  If the operator is Hermitian, than the output will always be real.  In the case of non-Hermitian operators, the return values may be complex.  Therefore, the :func:`qutip.expect` function will return an array of complex values for non-Hermitian operators when the input is a list/array of states or density matrices.

Of course, the :func:`qutip.expect` function works for spin states and operators:


.. ipython::
    
    In [1]: up = basis(2, 0)
    
    In [2]: down = basis(2, 1)
    
    In [3]: expect(sigmaz(), up)
    
    In [4]: expect(sigmaz(), down)


as well as the composite objects discussed in the next section :ref:`tensor`:

.. ipython::
    
    In [1]: spin1 = basis(2, 0)
    
    In [2]: spin2 = basis(2, 1)
    
    In [3]: two_spins = tensor(spin1, spin2)
    
    In [4]: sz1 = tensor(sigmaz(), qeye(2))
    
    In [5]: sz2 = tensor(qeye(2), sigmaz())
    
    In [6]: expect(sz1, two_spins)
    
    In [7]: expect(sz2, two_spins)

.. _states-super:

Superoperators and Vectorized Operators
=======================================

In addition to state vectors and density operators, QuTiP allows for
representing maps that act linearly on density operators using the Kraus,
Liouville supermatrix and Choi matrix formalisms. This support is based on the
correspondance between linear operators acting on a Hilbert space, and vectors
in two copies of that Hilbert space,
:math:`\mathrm{vec} : \mathcal{L}(\mathcal{H}) \to \mathcal{H} \otimes \mathcal{H}`
[Hav03]_, [Wat13]_.

This isomorphism is implemented in QuTiP by the
:obj:`~qutip.superoperator.operator_to_vector` and 
:obj:`~qutip.superoperator.vector_to_operator` functions:

.. ipython::

    In [1]: psi = basis(2, 0)
    
    In [2]: rho = ket2dm(psi)
    
    In [3]: rho
    
    In [4]: vec_rho = operator_to_vector(rho)

    In [5]: vec_rho

    In [6]: rho2 = vector_to_operator(vec_rho)
    
    In [7]: (rho - rho2).norm()
    
The :attr:`~qutip.Qobj.type` attribute indicates whether a quantum object is
a vector corresponding to an operator (``operator-ket``), or its Hermitian
conjugate (``operator-bra``).

Note that QuTiP uses the *column-stacking* convention for the isomorphism
between :math:`\mathcal{L}(\mathcal{H})` and :math:`\mathcal{H} \otimes \mathcal{H}`:

.. ipython::

    In [1]: import numpy as np
    
    In [2]: A = Qobj(np.arange(4).reshape((2, 2)))
    
    In [3]: A
     
    In [4]: operator_to_vector(A)

Since :math:`\mathcal{H} \otimes \mathcal{H}` is a vector space, linear maps
on this space can be represented as matrices, often called *superoperators*.
Using the :obj:`~qutip.Qobj`, the :obj:`~qutip.superoperator.spre` and :obj:`~qutip.superoperator.spost` functions, supermatrices
corresponding to left- and right-multiplication respectively can be quickly
constructed.

.. ipython::

    In [1]: X = sigmax()
    
    In [2]: S = spre(X) * spost(X.dag()) # Represents conjugation by X.
    
Note that this is done automatically by the :obj:`~qutip.superop_reps.to_super` function when given
``type='oper'`` input.

.. ipython::

    In [1]: S2 = to_super(X)
    
    In [2]: (S - S2).norm()
    
Quantum objects representing superoperators are denoted by ``type='super'``:

.. ipython::

    In [1]: S

Information about superoperators, such as whether they represent completely
positive maps, is exposed through the :attr:`~qutip.Qobj.iscp`, :attr:`~qutip.Qobj.istp`
and :attr:`~qutip.Qobj.iscptp` attributes:

.. ipython::

    In [1]: S.iscp, S.istp, S.iscptp
    True True True
    
In addition, dynamical generators on this extended space, often called
*Liouvillian superoperators*, can be created using the :func:`~qutip.superoperator.liouvillian` function. Each of these takes a Hamilonian along with
a list of collapse operators, and returns a ``type="super"`` object that can
be exponentiated to find the superoperator for that evolution.

.. ipython::

    In [1]: H = 10 * sigmaz()

    In [2]: c1 = destroy(2)

    In [3]: L = liouvillian(H, [c1])

    In [4]: L
     
    In [5]: S = (12 * L).expm()

For qubits, a particularly useful way to visualize superoperators is to plot them in the Pauli basis,
such that :math:`S_{\mu,\nu} = \langle\!\langle \sigma_{\mu} | S[\sigma_{\nu}] \rangle\!\rangle`. Because
the Pauli basis is Hermitian, :math:`S_{\mu,\nu}` is a real number for all Hermitian-preserving superoperators
:math:`S`,
allowing us to plot the elements of :math:`S` as a `Hinton diagram <http://matplotlib.org/examples/specialty_plots/hinton_demo.html>`_. In such diagrams, positive elements are indicated by white squares, and negative elements
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

In QuTiP, :math:`J(\Lambda)` can be found by calling the :func:`~qutip.superop_reps.to_choi`
function on a ``type="super"`` :ref:`Qobj`.

.. ipython::
    
    In [1]: X = sigmax()
    
    In [2]: S = sprepost(X, X)

    In [3]: J = to_choi(S)

    In [4]: print(J)

    In [5]: print(to_choi(spre(qeye(2))))

If a :ref:`Qobj` instance is already in the Choi :attr:`~Qobj.superrep`, then calling :func:`~qutip.superop_reps.to_choi`
does nothing:

.. ipython::
    
    In [1]: print(to_choi(J))

To get back to the superoperator representation, simply use the :func:`~qutip.superop_reps.to_super` function.
As with :func:`~qutip.superop_reps.to_choi`, :func:`~qutip.superop_reps.to_super` is idempotent:

.. ipython::
    
    In [1]: print(to_super(J) - S)

    In [2]: print(to_super(S))

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
using the :func:`~qutip.superop_reps.to_kraus` function.

.. ipython::

    In [1]: I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

    In [2]: S = sum(sprepost(P, P) for P in (I, X, Y, Z)) / 4
       ...: print(S)

    In [3]: J = to_choi(S)
       ...: print(J)

    In [4]: print(J.eigenstates()[1])

    In [5]: K = to_kraus(S)
       ...: print(K)

As with the other representation conversion functions, :func:`~qutip.superop_reps.to_kraus`
checks the :attr:`~Qobj.superrep` attribute of its input, and chooses an appropriate
conversion method. Thus, in the above example, we can also call :func:`~qutip.superop_reps.to_kraus`
on ``J``.

.. ipython::

    In [1]: KJ = to_kraus(J)
       ...: print(KJ)

    In [2]: for A, AJ in zip(K, KJ):
       ...:     print(A - AJ)

The Stinespring representation is closely related to the Kraus representation,
and consists of a pair of operators :math:`A` and :math:`B` such that for
all operators :math:`X` acting on :math:`\mathcal{H}`,

.. math::

    \Lambda(X) = \operatorname{Tr}_2(A X B^\dagger),

where the partial trace is over a new index that corresponds to the
index in the Kraus summation. Conversion to Stinespring
is handled by the :func:`~qutip.superop_reps.to_stinespring`
function.

.. ipython::

    In [1]: a = create(2).dag()

    In [2]: S_ad = sprepost(a * a.dag(), a * a.dag()) + sprepost(a, a.dag())
       ...: S = 0.9 * sprepost(I, I) + 0.1 * S_ad
       ...: print(S)

    In [3]: A, B = to_stinespring(S)
       ...: print(A)
       ...: print(B)

Notice that a new index has been added, such that :math:`A` and :math:`B`
have dimensions ``[[2, 3], [2]]``, with the length-3 index representing the
fact that the Choi matrix is rank-3 (alternatively, that the map has three
Kraus operators).

.. ipython::

    In [1]: to_kraus(S)

    In [2]: print(to_choi(S).eigenenergies())

Finally, the last superoperator representation supported by QuTiP is
the :math:`\chi`-matrix representation,

.. math::

    \Lambda(\rho) = \sum_{\alpha,\beta} \chi_{\alpha,\beta} B_{\alpha} \rho B_{\beta}^\dagger,

where :math:`\{B_\alpha\}` is a basis for the space of matrices acting
on :math:`\mathcal{H}`. In QuTiP, this basis is taken to be the Pauli
basis :math:`B_\alpha = \sigma_\alpha / \sqrt{2}`. Conversion to the
:math:`\chi` formalism is handled by the :func:`~qutip.superop_reps.to_chi`
function.

.. ipython::

    In [1]: chi = to_chi(S)
       ...: print(chi)

One convenient property of the :math:`\chi` matrix is that the average
gate fidelity with the identity map can be read off directly from
the :math:`\chi_{00}` element:

.. ipython::

    In [1]: print(average_gate_fidelity(S))

    In [2]: print(chi[0, 0] / 4)

Here, the factor of 4 comes from the dimension of the underlying
Hilbert space :math:`\mathcal{H}`. As with the superoperator
and Choi representations, the :math:`\chi` representation is
denoted by the :attr:`~Qobj.superrep`, such that :func:`~qutip.superop_reps.to_super`,
:func:`~qutip.superop_reps.to_choi`, :func:`~qutip.superop_reps.to_kraus`,
:func:`~qutip.superop_reps.to_stinespring` and :func:`~qutip.superop_reps.to_chi`
all convert from the :math:`\chi` representation appropriately.

Properties of Quantum Maps
==========================

In addition to converting between the different representations of quantum maps,
QuTiP also provides attributes to make it easy to check if a map is completely
positive, trace preserving and/or hermicity preserving. Each of these attributes
uses :attr:`~Qobj.superrep` to automatically perform any needed conversions.

In particular, a quantum map is said to be positive (but not necessarily completely
positive) if it maps all positive operators to positive operators. For instance, the
transpose map :math:`\Lambda(\rho) = \rho^{\mathrm{T}}` is a positive map. We run into
problems, however, if we tensor :math:`\Lambda` with the identity to get a partial
transpose map.

.. ipython::

    In [1]: rho = ket2dm(bell_state())

    In [2]: rho_out = partial_transpose(rho, [0, 1])
       ...: print(rho_out.eigenenergies())

Notice that even though we started with a positive map, we got an operator out
with negative eigenvalues. Complete positivity addresses this by requiring that
a map returns positive operators for all positive operators, and does so even
under tensoring with another map. The Choi matrix is very useful here, as it
can be shown that a map is completely positive if and only if its Choi matrix
is positive [Wat13]_. QuTiP implements this check with the :attr:`~Qobj.iscp`
attribute. As an example, notice that the snippet above already calculates
the Choi matrix of the transpose map by acting it on half of an entangled
pair. We simply need to manually set the ``dims`` and ``superrep`` attributes to reflect the
structure of the underlying Hilbert space and the chosen representation.

.. ipython::

    In [1]: J = rho_out

    In [2]: J.dims = [[[2], [2]], [[2], [2]]]
       ...: J.superrep = 'choi'

    In [3]: print(J.iscp)

This confirms that the transpose map is not completely positive. On the other hand,
the transpose map does satisfy a weaker condition, namely that it is hermicity preserving.
That is, :math:`\Lambda(\rho) = (\Lambda(\rho))^\dagger` for all :math:`\rho` such that
:math:`\rho = \rho^\dagger`. To see this, we note that :math:`(\rho^{\mathrm{T}})^\dagger
= \rho^*`, the complex conjugate of :math:`\rho`. By assumption, :math:`\rho = \rho^\dagger
= (\rho^*)^{\mathrm{T}}`, though, such that :math:`\Lambda(\rho) = \Lambda(\rho^\dagger) = \rho^*`.
We can confirm this by checking the :attr:`~Qobj.ishp` attribute:

.. ipython::

    In [1]: print(J.ishp)

Next, we note that the transpose map does preserve the trace of its inputs, such that
:math:`\operatorname{Tr}(\Lambda[\rho]) = \operatorname{Tr}(\rho)` for all :math:`\rho`.
This can be confirmed by the :attr:`~Qobj.istp` attribute:

.. ipython::

    In [1]: print(J.ishp)

Finally, a map is called a quantum channel if it always maps valid states to valid
states. Formally, a map is a channel if it is both completely positive and trace preserving.
Thus, QuTiP provides a single attribute to quickly check that this is true.

.. ipython::

    In [1]: print(J.iscptp)

    In [2]: print(to_super(qeye(2)).iscptp)

