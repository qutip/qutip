.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _random:

********************************************
Generating Random Quantum States & Operators
********************************************

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

In all cases, these functions can be called with a single parameter :math:`N` that indicates a :math:`NxN` matrix (`rand_dm`, `rand_herm`, `rand_unitary`), or a :math:`Nx1` vector (`rand_ket`), should be generated.  For example::

	>>> rand_ket(5)
	Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
	Qobj data =
	[[-0.69257444+0.07716822j]
	 [-0.48903635+0.00547509j]
	 [-0.04434023-0.04301746j]
	 [-0.27839376+0.00130696j]
	 [-0.35384822+0.26204821j]]

or::

   	>>> rand_herm(5)
	Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
	Qobj data =
	[[ 0.00000000+0.j         -0.06185503-0.20132642j -0.14510832-0.02572751j
	  -0.59958603-0.13581992j -0.11296987-0.19364997j]
	 [-0.06185503+0.20132642j  0.00000000+0.j         -0.67138090+0.38714512j
	  -0.14847184+0.13814627j -0.34759237-0.09025718j]
	 [-0.14510832+0.02572751j -0.67138090-0.38714512j  0.00000000+0.j
	  -0.45128021+0.16790562j  0.00000000+0.j        ]
	 [-0.59958603+0.13581992j -0.14847184-0.13814627j -0.45128021-0.16790562j
	  -0.33788000+0.j          0.00000000+0.j        ]
	 [-0.11296987+0.19364997j -0.34759237+0.09025718j  0.00000000+0.j
	   0.00000000+0.j          0.00000000+0.j        ]]

In this previous example, we see that the generated Hermitian operator contains a faction of elements that are identically equal to zero.  The number of nonzero elements is called the `density` and can be controlled by calling any of the random state/operator generators with a second argument between 0 and 1.  By default, the density for the operators is `0.75` where as ket vectors are completely dense (`1`).  For example: ::

	>>> rand_dm(5,0.5)
	Quantum object: dims = [[5], [5]], shape = [5, 5], type = oper, isHerm = True
	Qobj data =
	[[ 0.21266414+0.j          0.02943826+0.04209834j  0.16825067-0.08486911j
	   0.05243364-0.08772405j  0.00000000+0.j        ]
	 [ 0.02943826-0.04209834j  0.78733586+0.j          0.02000972+0.05611193j
	   0.19337602+0.08780273j  0.00000000+0.j        ]
	 [ 0.16825067+0.08486911j  0.02000972-0.05611193j  0.00000000+0.j
	   0.00000000+0.j          0.00000000+0.j        ]
	 [ 0.05243364+0.08772405j  0.19337602-0.08780273j  0.00000000+0.j
	   0.00000000+0.j          0.00000000+0.j        ]
	 [ 0.00000000+0.j          0.00000000+0.j          0.00000000+0.j
	   0.00000000+0.j          0.00000000+0.j        ]]

has 12 nonzero elements, or equivilently a density of 0.5.

.. important::  In the case of a density matrix, setting the density too low will result in not enough diagonal elements to satisfy :math:`Tr(\rho)=1`.

Composite random objects
========================

In many cases, one is interested in generating random quantum objects that correspond to composite systems generated using the :func:`qutip.tensor.tensor` function.  Specifying the tensor structure of a quantum object is done using the `dims` keyword argument in the same fashion as one would do for a :class:`qutip.Qobj` object::

	>>> rand_unitary(4,dims=[[2,2],[2,2]])
	Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = False
	Qobj data =
	[[ 0.63224397+0.03961913j -0.48934943+0.5741797j  -0.11171375+0.12333134j
	  -0.01673677-0.03977414j]
	 [-0.02160731+0.75410763j  0.51216688+0.40252671j -0.06994174-0.0380775j
	   0.00990143-0.00850615j]
	 [ 0.02132803+0.16503241j -0.07581063-0.02438219j  0.77718792+0.3962325j
	  -0.35597928+0.27968763j]
	 [-0.04216640+0.00917044j  0.00197097-0.0129038j  -0.16907045+0.41995429j
	   0.61014364+0.64864923j]]