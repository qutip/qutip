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

.. tabularcolumns:: | p{2cm} | p{3cm} |

.. cssclass:: table-striped

===========================   ========================================== ========================================
Random Variable Type          Sampling Functions                         Dimensions
===========================   ========================================== ========================================
State vector (``ket``)        `rand_ket`, `rand_ket_haar`                :math:`N \times 1`
Hermitian operator (``oper``) `rand_herm`                                :math:`N \times 1`
Density operator (``oper``)   `rand_dm`, `rand_dm_hs`, `rand_dm_ginibre` :math:`N \times N`
Unitary operator (``oper``)   `rand_unitary`, `rand_unitary_haar`        :math:`N \times N`
CPTP channel (``super``)      `rand_super`, `rand_super_bcsz`            :math:`(N \times N) \times (N \times N)`
===========================   ========================================== ========================================

In all cases, these functions can be called with a single parameter :math:`N` that dimension of the relevant Hilbert space. The optional
``dims`` keyword argument allows for the dimensions of a random state, unitary or channel to be broken down into subsystems.

.. ipython::
	
	In [3]: print rand_super_bcsz(7).dims

	In [4]: print rand_super_bcsz(6, dims=[[[2, 3], [2, 3]], [[2, 3], [2, 3]]]).dims
	
See the API documentation: :ref:`functions-rand` for details.


In this previous example, we see that the generated Hermitian operator contains a fraction of elements that are identically equal to zero.  The number of nonzero elements is called the `density` and can be controlled by calling any of the random state/operator generators with a second argument between 0 and 1.  By default, the density for the operators is `0.75` where as ket vectors are completely dense (`1`).  For example:

.. ipython::

   In [1]: rand_dm(5, 0.5)


has roughly half nonzero elements, or equivalently a density of 0.5.

.. warning::  

    In the case of a density matrix, setting the density too low will result in     not enough diagonal elements to satisfy :math:`Tr(\rho)=1`.

Composite random objects
========================

In many cases, one is interested in generating random quantum objects that correspond to composite systems generated using the :func:`qutip.tensor.tensor` function.  Specifying the tensor structure of a quantum object is done using the `dims` keyword argument in the same fashion as one would do for a :class:`qutip.Qobj` object:


.. ipython::

   In [1]: rand_dm(4, 0.5, dims=[[2,2], [2,2]])


