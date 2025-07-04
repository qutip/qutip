.. _random:

********************************************
Generating Random Quantum States & Operators
********************************************

.. testsetup:: [random]

   from qutip import rand_herm, rand_dm, rand_super_bcsz, rand_dm_ginibre

QuTiP includes a collection of random state, unitary and channel generators for simulations, Monte Carlo evaluation, theorem evaluation, and code testing.
Each of these objects can be sampled from one of several different distributions.

For example, a random Hermitian operator can be sampled by calling :func:`.rand_herm` function:

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

+-------------------------------+-----------------------------------------------+------------------------------------------+
| Random Variable Type          | Sampling Functions                            | Dimensions                               |
+===============================+===============================================+==========================================+
| State vector (``ket``)        | :func:`.rand_ket`                             | :math:`N \times 1`                       |
+-------------------------------+-----------------------------------------------+------------------------------------------+
| Hermitian operator (``oper``) | :func:`.rand_herm`                            | :math:`N \times N`                       |
+-------------------------------+-----------------------------------------------+------------------------------------------+
| Density operator (``oper``)   | :func:`.rand_dm`                              | :math:`N \times N`                       |
+-------------------------------+-----------------------------------------------+------------------------------------------+
| Unitary operator (``oper``)   | :func:`.rand_unitary`                         | :math:`N \times N`                       |
+-------------------------------+-----------------------------------------------+------------------------------------------+
| stochastic matrix (``oper``)  | :func:`.rand_stochastic`                      | :math:`N \times N`                       |
+-------------------------------+-----------------------------------------------+------------------------------------------+
| CPTP channel (``super``)      | :func:`.rand_super`, :func:`.rand_super_bcsz` | :math:`(N \times N) \times (N \times N)` |
+-------------------------------+-----------------------------------------------+------------------------------------------+
| CPTP map (list of ``oper``)   | :func:`.rand_kraus_map`                       | :math:`N \times N` (N**2 operators)      |
+-------------------------------+-----------------------------------------------+------------------------------------------+

In all cases, these functions can be called with a single parameter :math:`dimensions` that can be the size of the relevant Hilbert space or the dimensions of a random state, unitary or channel.



.. doctest:: [random]

    >>> rand_super_bcsz(7).dims
    [[[7], [7]], [[7], [7]]]
    >>> rand_super_bcsz([[2, 3], [2, 3]]).dims
    [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]

Several of the random :class:`.Qobj` function in QuTiP support additional parameters as well, namely *density* and *distribution*.
:func:`.rand_dm`, :func:`.rand_herm`, :func:`.rand_unitary` and :func:`.rand_ket` can be created using multiple method controlled by *distribution*.
The :func:`.rand_ket`, :func:`.rand_herm` and :func:`.rand_unitary` functions can return quantum objects such that a fraction of the elements are identically equal to zero.
The ratio of nonzero elements is passed as the ``density`` keyword argument.
By contrast, `rand_super_bcsz` take as an argument the rank of the generated object, such that passing ``rank=1`` returns a random pure state or unitary channel, respectively.
Passing ``rank=None`` specifies that the generated object should be full-rank for the given dimension.
`rand_dm` can support *density* or *rank* depending on the chosen distribution.

For example,

.. doctest:: [random]
    :hide:

    >>> np.random.seed(42)

.. doctest:: [random]

   >>> rand_dm(5, density=0.5, distribution="herm")
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
   Qobj data =
   [[ 0.298+0.j   ,  0.   +0.j   , -0.095+0.1j  ,  0.   +0.j   ,-0.105+0.122j],
    [ 0.   +0.j   ,  0.088+0.j   ,  0.   +0.j   , -0.018-0.001j, 0.   +0.j   ],
    [-0.095-0.1j  ,  0.   +0.j   ,  0.328+0.j   ,  0.   +0.j   ,-0.077-0.033j],
    [ 0.   +0.j   , -0.018+0.001j,  0.   +0.j   ,  0.084+0.j   , 0.   +0.j   ],
    [-0.105-0.122j,  0.   +0.j   , -0.077+0.033j,  0.   +0.j   , 0.201+0.j   ]]

   >>> rand_dm_ginibre(5, rank=2)
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
   Qobj data =
   [[ 0.307+0.j   , -0.258+0.039j, -0.039+0.184j,  0.041-0.054j, 0.016+0.045j],
    [-0.258-0.039j,  0.239+0.j   ,  0.075-0.15j , -0.053+0.008j,-0.057-0.078j],
    [-0.039-0.184j,  0.075+0.15j ,  0.136+0.j   , -0.05 -0.052j,-0.028-0.058j],
    [ 0.041+0.054j, -0.053-0.008j, -0.05 +0.052j,  0.083+0.j   , 0.101-0.056j],
    [ 0.016-0.045j, -0.057+0.078j, -0.028+0.058j,  0.101+0.056j, 0.236+0.j   ]]


See the API documentation: :ref:`api-rand` for details.

.. warning::

    When using the ``density`` keyword argument, setting the density too low may result in not enough diagonal elements to satisfy trace
    constraints.

Random objects with a given eigen spectrum
==========================================

It is also possible to generate random Hamiltonian (:func:`.rand_herm`) and densitiy matrices (:func:`.rand_dm`) with a given eigen spectrum.
This is done by passing an array to eigenvalues argument to either function and choosing the "eigen" distribution.
For example,

.. doctest:: [random]
    :hide:

    >>> np.random.seed(42)

.. doctest:: [random]

   >>> eigs = np.arange(5)

   >>> H = rand_herm(5, density=0.5, eigenvalues=eigs, distribution="eigen")

   >>> H # doctest: +NORMALIZE_WHITESPACE
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
   Qobj data =
   [[ 0.5  +0.j  ,  0.228+0.27j,  0.   +0.j  ,  0.   +0.j  ,-0.228-0.27j],
    [ 0.228-0.27j,  1.75 +0.j  ,  0.456+0.54j,  0.   +0.j  , 1.25 +0.j  ],
    [ 0.   +0.j  ,  0.456-0.54j,  3.   +0.j  ,  0.   +0.j  , 0.456-0.54j],
    [ 0.   +0.j  ,  0.   +0.j  ,  0.   +0.j  ,  3.   +0.j  , 0.   +0.j  ],
    [-0.228+0.27j,  1.25 +0.j  ,  0.456+0.54j,  0.   +0.j  , 1.75 +0.j  ]]


   >>> H.eigenenergies() # doctest: +NORMALIZE_WHITESPACE
   array([7.70647994e-17, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
       4.00000000e+00])


In order  to generate a random object with a given spectrum QuTiP applies a series of random complex Jacobi rotations.
This technique requires many steps to build the desired quantum object, and is thus suitable only for objects with Hilbert dimensionality :math:`\lesssim 1000`.



Composite random objects
========================

In many cases, one is interested in generating random quantum objects that correspond to composite systems generated using the :func:`.tensor` function.
Specifying the tensor structure of a quantum object is done passing a list for the first argument.
The resulting quantum objects size will be the product of the elements in the list and the resulting :class:`.Qobj` dimensions will be ``[dims, dims]``:

.. doctest:: [random]
    :hide:

    >>> np.random.seed(42)

.. doctest:: [random]

   >>> rand_unitary([2, 2], density=0.5) # doctest: +NORMALIZE_WHITESPACE
   Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
   Qobj data =
   [[ 0.887+0.061j,  0.   +0.j   ,  0.   +0.j   , -0.191-0.416j],
    [ 0.   +0.j   ,  0.604+0.116j, -0.32 -0.721j,  0.   +0.j   ],
    [ 0.   +0.j   ,  0.768+0.178j,  0.227+0.572j,  0.   +0.j   ],
    [ 0.412-0.2j  ,  0.   +0.j   ,  0.   +0.j   ,  0.724+0.516j]]


Controlling the random number generator
=======================================

Qutip uses numpy random number generator to create random quantum objects.
To control the random number, a seed as an `int` or `numpy.random.SeedSequence` or a `numpy.random.Generator` can be passed to the `seed` keyword argument:

.. doctest:: [random]

    >>> rng = np.random.default_rng(12345)
    >>> rand_ket(2, seed=rng) # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [1]], shape=(2, 1), type='ket'
    Qobj data =
    [[-0.697+0.618j],
     [-0.326-0.163j]]


Internal matrix format
======================

The internal storage type of the generated random quantum objects can be set with the *dtype* keyword.

.. doctest:: [random]

    >>> rand_ket(2, dtype="dense").data
    Dense(shape=(2, 1), fortran=True)

    >>> rand_ket(2, dtype="CSR").data
    CSR(shape=(2, 1), nnz=2)

..
  TODO: add a link to a page explaining data-types.
