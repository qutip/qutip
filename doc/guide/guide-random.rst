.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _random:

********************************************
Generating Random Quantum States & Operators
********************************************

.. ipython::
   :suppress:

   In [1]: from qutip import *

QuTiP includes a collection of random state, unitary and channel generators for simulations, Monte Carlo evaluation, theorem evaluation, and code testing.
Each of these objects can be sampled from one of several different distributions including the default distributions
used by QuTiP versions prior to 3.2.0.

For example, a random Hermitian operator can be sampled by calling `rand_herm` function:

.. ipython::

	In [2]: rand_herm(5)

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

In all cases, these functions can be called with a single parameter :math:`N` that dimension of the relevant Hilbert space. The optional
``dims`` keyword argument allows for the dimensions of a random state, unitary or channel to be broken down into subsystems.

.. ipython::
	
	In [3]: print rand_super_bcsz(7).dims

	In [4]: print rand_super_bcsz(6, dims=[[[2, 3], [2, 3]], [[2, 3], [2, 3]]]).dims

Several of the distributions supported by QuTiP support additional parameters as well, namely *density* and *rank*. In particular,
the `rand_herm` and `rand_dm` functions return quantum objects such that a fraction of the elements are identically equal to zero.
The ratio of nonzero elements is passed as the ``density`` keyword argument. By contrast, the `rand_dm_ginibre` and
`rand_super_bcsz` take as an argument the rank of the generated object, such that passing ``rank=1`` returns a random
pure state or unitary channel, respectively. Passing ``rank=None`` specifies that the generated object should be
full-rank for the given dimension.

For example,

.. ipython::

   In [5]: rand_dm(5, density=0.5)

   In [6]: rand_dm_ginibre(5, rank=2)
	
See the API documentation: :ref:`functions-rand` for details.

.. warning::  

    When using the ``density`` keyword argument, setting the density too low may result in not enough diagonal elements to satisfy trace
    constraints.

Random objects with a given eigen spectrum
==========================================

.. note::

    New in QuTiP 3.2

It is also possible to generate random Hamiltonian (``rand_herm``) and densitiy matrices (``rand_dm``) with a given eigen spectrum.  This is done by passing an array of eigenvalues as the first argument to either function.  For example,


.. ipython::
    
   In [7]: eigs = np.arange(5)    
   
   In [8]: H = rand_herm(eigs, density=0.5)
   
   In [9]: H
   
   In [10]: H.eigenenergies()


In order to generate a random object with a given spectrum QuTiP applies a series of random complex Jacobi rotations.  This technique requires many steps to build the desired quantum object, and is thus suitable only for objects with Hilbert dimensionality :math:`\lesssim 1000`.



Composite random objects
========================

In many cases, one is interested in generating random quantum objects that correspond to composite systems generated using the :func:`qutip.tensor.tensor` function.  Specifying the tensor structure of a quantum object is done using the `dims` keyword argument in the same fashion as one would do for a :class:`qutip.Qobj` object:

.. ipython::

   In [1]: rand_dm(4, 0.5, dims=[[2,2], [2,2]])

