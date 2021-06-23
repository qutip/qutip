.. _random:

********************************************
Generating Random Quantum States & Operators
********************************************

.. testsetup:: [random]

   from qutip import rand_herm, rand_dm, rand_super_bcsz, rand_dm_ginibre

QuTiP includes a collection of random state, unitary and channel generators for simulations, Monte Carlo evaluation, theorem evaluation, and code testing.
Each of these objects can be sampled from one of several different distributions including the default distributions
used by QuTiP versions prior to 3.2.0.

For example, a random Hermitian operator can be sampled by calling `rand_herm` function:

.. doctest:: [random]
    :hide:

    >>> np.random.seed(42)

.. doctest:: [random]

  >>> rand_herm(5) # doctest: +NORMALIZE_WHITESPACE
  Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
  Qobj data =
  [[-0.25091976+0.j          0.        +0.j          0.        +0.j
    -0.21793701+0.47037633j -0.23212846-0.61607187j]
   [ 0.        +0.j         -0.88383278+0.j          0.836086  -0.23956218j
    -0.09464275+0.45370863j -0.15243356+0.65392096j]
   [ 0.        +0.j          0.836086  +0.23956218j  0.66488528+0.j
    -0.26290446+0.64984451j -0.52603038-0.07991553j]
   [-0.21793701-0.47037633j -0.09464275-0.45370863j -0.26290446-0.64984451j
    -0.13610996+0.j         -0.34240902-0.2879303j ]
   [-0.23212846+0.61607187j -0.15243356-0.65392096j -0.52603038+0.07991553j
    -0.34240902+0.2879303j   0.        +0.j        ]]



.. tabularcolumns:: | p{2cm} | p{3cm} | c |

.. cssclass:: table-striped

+-------------------------------+--------------------------------------------+------------------------------------------+
| Random Variable Type          | Sampling Functions                         | Dimensions                               |
+===============================+============================================+==========================================+
| State vector (``ket``)        | `rand_ket`, `rand_ket_haar`                | :math:`N \times 1`                       |
+-------------------------------+--------------------------------------------+------------------------------------------+
| Hermitian operator (``oper``) | `rand_herm`                                | :math:`N \times 1`                       |
+-------------------------------+--------------------------------------------+------------------------------------------+
| Density operator (``oper``)   | `rand_dm`, `rand_dm_hs`, `rand_dm_ginibre` | :math:`N \times N`                       |
+-------------------------------+--------------------------------------------+------------------------------------------+
| Unitary operator (``oper``)   | `rand_unitary`, `rand_unitary_haar`        | :math:`N \times N`                       |
+-------------------------------+--------------------------------------------+------------------------------------------+
| CPTP channel (``super``)      | `rand_super`, `rand_super_bcsz`            | :math:`(N \times N) \times (N \times N)` |
+-------------------------------+--------------------------------------------+------------------------------------------+

In all cases, these functions can be called with a single parameter :math:`N` that specifies the dimension of the relevant Hilbert space. The optional
``dims`` keyword argument allows for the dimensions of a random state, unitary or channel to be broken down into subsystems.

.. doctest:: [random]

    >>> rand_super_bcsz(7).dims
    [[[7], [7]], [[7], [7]]]
    >>> rand_super_bcsz(6, dims=[[[2, 3], [2, 3]], [[2, 3], [2, 3]]]).dims
    [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]

Several of the distributions supported by QuTiP support additional parameters as well, namely *density* and *rank*. In particular,
the `rand_herm` and `rand_dm` functions return quantum objects such that a fraction of the elements are identically equal to zero.
The ratio of nonzero elements is passed as the ``density`` keyword argument. By contrast, the `rand_dm_ginibre` and
`rand_super_bcsz` take as an argument the rank of the generated object, such that passing ``rank=1`` returns a random
pure state or unitary channel, respectively. Passing ``rank=None`` specifies that the generated object should be
full-rank for the given dimension.

For example,

.. doctest:: [random]
    :hide:

    >>> np.random.seed(42)

.. doctest:: [random]

   >>> rand_dm(5, density=0.5)
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
   Qobj data =
   [[ 0.05157906+0.j          0.04491736+0.01043329j  0.06966148+0.00344713j
      0.        +0.j          0.04031493-0.01886791j]
    [ 0.04491736-0.01043329j  0.33632352+0.j         -0.08046093+0.02954712j
      0.0037455 +0.03940256j -0.05679126-0.01322392j]
    [ 0.06966148-0.00344713j -0.08046093-0.02954712j  0.2938209 +0.j
      0.0029377 +0.04463531j  0.05318743-0.02817689j]
    [ 0.        +0.j          0.0037455 -0.03940256j  0.0029377 -0.04463531j
      0.22553181+0.j          0.01657495+0.06963845j]
    [ 0.04031493+0.01886791j -0.05679126+0.01322392j  0.05318743+0.02817689j
      0.01657495-0.06963845j  0.09274471+0.j        ]]

   >>> rand_dm_ginibre(5, rank=2)
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
   Qobj data =
   [[ 0.07318288+2.60675616e-19j  0.10426866-6.63115850e-03j
     -0.05377455-2.66949369e-02j -0.01623153+7.66824687e-02j
     -0.12255602+6.11342416e-02j]
    [ 0.10426866+6.63115850e-03j  0.30603789+1.44335373e-18j
     -0.03129486-4.16194216e-03j -0.09832531+1.74110000e-01j
     -0.27176358-4.84608761e-02j]
    [-0.05377455+2.66949369e-02j -0.03129486+4.16194216e-03j
      0.07055265-8.76912454e-19j -0.0183289 -2.72720794e-02j
      0.01196277-1.01037189e-01j]
    [-0.01623153-7.66824687e-02j -0.09832531-1.74110000e-01j
     -0.0183289 +2.72720794e-02j  0.14168414-1.51340961e-19j
      0.07847628+2.07735199e-01j]
    [-0.12255602-6.11342416e-02j -0.27176358+4.84608761e-02j
      0.01196277+1.01037189e-01j  0.07847628-2.07735199e-01j
      0.40854244-6.75775934e-19j]]




See the API documentation: :ref:`functions-rand` for details.

.. warning::

    When using the ``density`` keyword argument, setting the density too low may result in not enough diagonal elements to satisfy trace
    constraints.

Random objects with a given eigen spectrum
==========================================

It is also possible to generate random Hamiltonian (``rand_herm``) and densitiy matrices (``rand_dm``) with a given eigen spectrum.  This is done by passing an array of eigenvalues as the first argument to either function.  For example,

.. doctest:: [random]
    :hide:

    >>> np.random.seed(42)

.. doctest:: [random]

   >>> eigs = np.arange(5)

   >>> H = rand_herm(eigs, density=0.5)

   >>> H # doctest: +NORMALIZE_WHITESPACE
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
   Qobj data =
   [[ 2.51387054-5.55111512e-17j  0.81161447+2.02283642e-01j
      0.        +0.00000000e+00j  0.875     +3.35634092e-01j
      0.81161447+2.02283642e-01j]
    [ 0.81161447-2.02283642e-01j  1.375     +0.00000000e+00j
      0.        +0.00000000e+00j -0.76700198+5.53011066e-01j
      0.375     +0.00000000e+00j]
    [ 0.        +0.00000000e+00j  0.        +0.00000000e+00j
      2.        +0.00000000e+00j  0.        +0.00000000e+00j
      0.        +0.00000000e+00j]
    [ 0.875     -3.35634092e-01j -0.76700198-5.53011066e-01j
      0.        +0.00000000e+00j  2.73612946+0.00000000e+00j
     -0.76700198-5.53011066e-01j]
    [ 0.81161447-2.02283642e-01j  0.375     +0.00000000e+00j
      0.        +0.00000000e+00j -0.76700198+5.53011066e-01j
      1.375     +0.00000000e+00j]]


   >>> H.eigenenergies() # doctest: +NORMALIZE_WHITESPACE
   array([7.70647994e-17, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
       4.00000000e+00])


In order  to generate a random object with a given spectrum QuTiP applies a series of random complex Jacobi rotations.  This technique requires many steps to build the desired quantum object, and is thus suitable only for objects with Hilbert dimensionality :math:`\lesssim 1000`.



Composite random objects
========================

In many cases, one is interested in generating random quantum objects that correspond to composite systems generated using the :func:`qutip.tensor.tensor` function.  Specifying the tensor structure of a quantum object is done using the `dims` keyword argument in the same fashion as one would do for a :class:`qutip.Qobj` object:

.. doctest:: [random]
    :hide:

    >>> np.random.seed(42)

.. doctest:: [random]

   >>> rand_dm(4, 0.5, dims=[[2,2], [2,2]]) # doctest: +NORMALIZE_WHITESPACE
   Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
   Qobj data =
   [[ 0.13622928+0.j          0.        +0.j          0.01180807-0.01739166j
      0.        +0.j        ]
    [ 0.        +0.j          0.14600238+0.j          0.10335328+0.21790786j
     -0.00426027-0.02193627j]
    [ 0.01180807+0.01739166j  0.10335328-0.21790786j  0.57566072+0.j
     -0.0670631 +0.04124094j]
    [ 0.        +0.j         -0.00426027+0.02193627j -0.0670631 -0.04124094j
      0.14210761+0.j        ]]
