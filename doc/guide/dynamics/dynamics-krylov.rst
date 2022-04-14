.. _krylov:

*******************************************
Krylov Solver
*******************************************

.. _krylov-intro:

Introduction
=============

The Krylov-subspace method is a standard method to approximate quantum dynamics.  Let :math:`\left|\psi\right\rangle` be a state in a :math:`D`-dimensional complex Hilbert space that evolves under a time-independent Hamiltonian :math:`H`. Then, the :math:`N`-dimensional Krylov subspace associated with that state and Hamiltonian is given by

.. math::
	:label: krylovsubspace

	\mathcal{K}_{N}=\operatorname{span}\left\{|\psi\rangle, H|\psi\rangle, \ldots, H^{N-1}|\psi\rangle\right\},

where the dimension :math:`N<D` is a parameter of choice. In order to construct an orthornormal basis :math:`B_N` for :math:`\mathcal{K}_{N}`, the simplest algorithmiss the well-known Lanczos algorithm, which provides a sort of Gram-Schmidt procedure that harnesses the fact that orthonormalization needs to be imposed only with respect to the last two vectors in the basis. Written in this basism the time-evolved state can be approximated as

.. math::
	:label: lanczoskrylov

	|\psi(t)\rangle=e^{-iHt}|\psi\rangle\approx\mathbb{P}_{N}e^{-iHt}\mathbb{P}_{N}|\psi\rangle=\mathbb{V}_{N}^{\dagger}e^{-iT_{N}t}\mathbb{V}_{N}|\psi\rangle\equiv\left|\psi_{N}(t)\right\rangle,

where  :math:`T_{N}=\mathbb{V}_{N} H \mathbb{V}_{N}^{\dagger}` is the Hamiltonian reduced to the Krylov subspace (which takes a tridiagonal matrix form), and :math:`\mathbb{V}_{N}^{\dagger}` is the matrix containing the vectors of the Krylov basis as columns.

With the above approximation, the time-evolution is calculated only with a smaller square matrix of the desired size. Therefore, the Krylov-method provides huge speed-ups in computation of short-time evolutions when the dimension of the Hamiltonian is very large, point at which exact calculations on the complete subspace are practically impossible.

One of the biggest problems with this type of method is the control of the error. After a short time, the error starts to grow exponentially. However, this can be easily corrected by restarting the subspace when the error reaches a certain threshold. Therefore, a series of :math:`M` Krylov-subspace time evolutions provides accurate solutions for a time-evolution.

.. _krylov-qutip:

Krylov Solver in QuTiP
====================

In QuTiP, Krylov-subspace evolution is implemented as the function :func:`qutip.krylovsolve`. Arguments are nearly the same as :func:`qutip.mesolve`
function for master-equation evolution, except that the initial state must be a ket vector, as oppose to a density matrix, and the additional parameter ``krylov_dim`` that defines the maximum allowed Krylov-subspace dimension. The maximum number of allowed Lanczos partitions can also be determined using the :func:`qutip.solver.options.nsteps` parameter, which defaults to '10000'. 

Let's solve a simple example using the algorithm in QuTiP to get familiar with the method.

.. _krylov-sparse:

Sparse and Dense Hamiltonians
-----------------------------------

If the Hamiltonian of interest is known to be sparse, :func:`qutip.krylovsolve` also comes equipped with the possibility to store its internal data in a sparse optimized format using scipy. This allows for significant speed-ups, let's showcase it:


.. code:: python
    :context: krylov-sparse-dense-comparison

	from qutip import rand_ket, rand_herm, krylovsolve
	from time import time
	import numpy as np

	def time_krylov(psi0, H, tlist, sparse):
	  start = time()
	  krylovsolve(H, psi0, tlist, krylov_dim=20, sparse=sparse)
	  end = time()
	  return end - start

	dim = 1000
	n_random_samples = 20

	# first index for type of H and second index for sparse = True or False (dense)
	t_ss_list, t_sd_list, t_ds_list, t_dd_list = [], [], [], []
	tlist = np.linspace(0, 1, 200)

	for n in range(n_random_samples):
	  psi0 = rand_ket(dim)
	  H_sparse = rand_herm(dim, density=0.1, seed=0)
	  H_dense = rand_herm(dim, density=0.9, seed=0)

	  t_ss_list.append(time_krylov(psi0, H_sparse, tlist, sparse=True))
	  t_sd_list.append(time_krylov(psi0, H_sparse, tlist, sparse=False))
	  t_ds_list.append(time_krylov(psi0, H_dense, tlist, sparse=True))
	  t_dd_list.append(time_krylov(psi0, H_dense, tlist, sparse=False))

	t_ss_average = np.mean(t_ss_list)
	t_sd_average = np.mean(t_sd_list)
	t_ds_average = np.mean(t_ds_list)
	t_dd_average = np.mean(t_dd_list)

	print(f"Average time of solution for a sparse H is {t_ss_average} for sparse=True and {t_sd_average} for sparse=False")
	print(f"Average time of solution for a dense H is {t_ds_average} for sparse=True and {t_dd_average} for sparse=False")
