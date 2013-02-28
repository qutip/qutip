.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _random:

********************************************
Generating Random Quantum States & Operators
********************************************

.. ipython::
   :suppress:

   In [1]: from qutip import *

QuTiP includes a collection of random state generators for simulations, theorem evaluation, and code testing:

.. tabularcolumns:: | p{2cm} | p{3cm} |

+-------------------------------+---------------------------------+
| Function                      | Description                     |
+===============================+=================================+
| `rand_ket`                    | Random ket-vector               |
+-------------------------------+---------------------------------+
| `rand_dm`                     | Random density matrix           |
+-------------------------------+---------------------------------+
| `rand_herm`                   | Random Hermitian matrix         |
+-------------------------------+---------------------------------+
| `rand_unitary`                | Random Unitary matrix           |
+-------------------------------+---------------------------------+

See the API documentation: :ref:`functions-rand` for details.

In all cases, these functions can be called with a single parameter :math:`N` that indicates a :math:`NxN` matrix (`rand_dm`, `rand_herm`, `rand_unitary`), or a :math:`Nx1` vector (`rand_ket`), should be generated.  For example:

.. ipython::

   In [1]: rand_ket(5)

or

.. ipython::

   In [1]: rand_herm(5)

In this previous example, we see that the generated Hermitian operator contains a fraction of elements that are identically equal to zero.  The number of nonzero elements is called the `density` and can be controlled by calling any of the random state/operator generators with a second argument between 0 and 1.  By default, the density for the operators is `0.75` where as ket vectors are completely dense (`1`).  For example:

.. ipython::

   In [1]: rand_dm(5, 0.5)


has rougly half nonzero elements, or equivalently a density of 0.5.

.. important::  In the case of a density matrix, setting the density too low will result in not enough diagonal elements to satisfy :math:`Tr(\rho)=1`.

Composite random objects
========================

In many cases, one is interested in generating random quantum objects that correspond to composite systems generated using the :func:`qutip.tensor.tensor` function.  Specifying the tensor structure of a quantum object is done using the `dims` keyword argument in the same fashion as one would do for a :class:`qutip.Qobj` object:


.. ipython::

   In [1]: rand_dm(4, 0.5, dims=[[2,2], [2,2]])


