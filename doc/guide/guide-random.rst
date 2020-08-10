.. QuTiP
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

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
  :skipif: True

  >>> rand_herm(5)
  Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
  Qobj data =
  [[ 0.92220399+0.j         -0.00177495-0.26175434j -0.84572304-0.06459622j
    -0.57046199+0.46994043j  0.64580663+0.44468901j]
   [-0.00177495+0.26175434j -0.55665533+0.j         -0.67850336-0.70193262j
     0.44073064+0.11465923j  0.        +0.j        ]
   [-0.84572304+0.06459622j -0.67850336+0.70193262j  0.        +0.j
     0.        +0.j          0.29735536+0.14848001j]
   [-0.57046199-0.46994043j  0.44073064-0.11465923j  0.        +0.j
     0.80589034+0.j          0.63044993-0.74810352j]
   [ 0.64580663-0.44468901j  0.        +0.j          0.29735536-0.14848001j
     0.63044993+0.74810352j  0.69628846+0.j        ]]



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
  :skipif: True

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
  :skipif: True

   >>> rand_dm(5, density=0.5)
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[ 0.21853371+0.j          0.        +0.j          0.        +0.j
      -0.1110277 +0.07168105j -0.00798517+0.01564542j]
     [ 0.        +0.j          0.10960418+0.j         -0.06966734-0.04140564j
       0.01322498-0.00281043j  0.02413738-0.02181861j]
     [ 0.        +0.j         -0.06966734+0.04140564j  0.06386367+0.j
      -0.00734444+0.00678244j -0.01163395+0.0376669j ]
     [-0.1110277 -0.07168105j  0.01322498+0.00281043j -0.00734444-0.00678244j
       0.42714306+0.j          0.02079556-0.0120616j ]
     [-0.00798517-0.01564542j  0.02413738+0.02181861j -0.01163395-0.0376669j
       0.02079556+0.0120616j   0.18085537+0.j        ]]

   >>> rand_dm_ginibre(5, rank=2)
   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[ 0.32670633+3.47958003e-18j  0.04080996-2.41217584e-01j
       0.13936723+1.65441254e-01j -0.06369351+8.34214409e-02j
       0.01621671+9.81149214e-02j]
     [ 0.04080996+2.41217584e-01j  0.24040644+1.62919540e-18j
      -0.18487672+1.04474499e-01j -0.09667389-1.00881810e-01j
      -0.03533823+1.90356635e-02j]
     [ 0.13936723-1.65441254e-01j -0.18487672-1.04474499e-01j
       0.26184618-3.29923918e-18j  0.07451586+1.48819867e-01j
       0.00920194+5.26217454e-02j]
     [-0.06369351-8.34214409e-02j -0.09667389+1.00881810e-01j
       0.07451586-1.48819867e-01j  0.11879212-9.40619819e-19j
       0.01109481+1.86027569e-02j]
     [ 0.01621671-9.81149214e-02j -0.03533823-1.90356635e-02j
       0.00920194-5.26217454e-02j  0.01109481-1.86027569e-02j
       0.05224893-8.68916424e-19j]]



See the API documentation: :ref:`functions-rand` for details.

.. warning::

    When using the ``density`` keyword argument, setting the density too low may result in not enough diagonal elements to satisfy trace
    constraints.

Random objects with a given eigen spectrum
==========================================

.. note::

    New in QuTiP 3.2

It is also possible to generate random Hamiltonian (``rand_herm``) and densitiy matrices (``rand_dm``) with a given eigen spectrum.  This is done by passing an array of eigenvalues as the first argument to either function.  For example,


.. doctest:: [random]
  :skipif: True

   eigs = np.arange(5)

   H = rand_herm(eigs, density=0.5)

   >>> H

   Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True
    Qobj data =
    [[ 1.        +0.00000000e+00j  0.30440592-3.96657330e-01j
       0.30440592-3.96657330e-01j  0.70620446+3.57107365e-02j
       0.        +0.00000000e+00j]
     [ 0.30440592+3.96657330e-01j  1.5       -5.55111512e-17j
       0.5       -5.55111512e-17j -0.40161578-5.81983472e-01j
       0.        +0.00000000e+00j]
     [ 0.30440592+3.96657330e-01j  0.5       -5.55111512e-17j
       1.5       -5.55111512e-17j -0.40161578-5.81983472e-01j
       0.        +0.00000000e+00j]
     [ 0.70620446-3.57107365e-02j -0.40161578+5.81983472e-01j
      -0.40161578+5.81983472e-01j  2.        +0.00000000e+00j
       0.        +0.00000000e+00j]
     [ 0.        +0.00000000e+00j  0.        +0.00000000e+00j
       0.        +0.00000000e+00j  0.        +0.00000000e+00j
       4.        +0.00000000e+00j]]


   >>> H.eigenenergies()
   array([5.55111512e-16, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
       4.00000000e+00])


In order  to generate a random object with a given spectrum QuTiP applies a series of random complex Jacobi rotations.  This technique requires many steps to build the desired quantum object, and is thus suitable only for objects with Hilbert dimensionality :math:`\lesssim 1000`.



Composite random objects
========================

In many cases, one is interested in generating random quantum objects that correspond to composite systems generated using the :func:`qutip.tensor.tensor` function.  Specifying the tensor structure of a quantum object is done using the `dims` keyword argument in the same fashion as one would do for a :class:`qutip.Qobj` object:

.. doctest:: [random]
  :skipif: True

   >>> rand_dm(4, 0.5, dims=[[2,2], [2,2]])
   Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[ 0.36179013+0.j         -0.00915101-0.07615654j  0.03834756-0.07267967j
       0.06856633+0.09612863j]
     [-0.00915101+0.07615654j  0.18964291+0.j          0.12239802+0.08465464j
       0.        +0.j        ]
     [ 0.03834756+0.07267967j  0.12239802-0.08465464j  0.23616123+0.j
       0.04744011-0.09439229j]
     [ 0.06856633-0.09612863j  0.        +0.j          0.04744011+0.09439229j
       0.21240573+0.j        ]]
