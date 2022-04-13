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

Although this approximation may fail quickly depending on the properties of the Hamiltonian, a series of :math:`M` Krylov-subspace time evolutions provides accurate solutions for "not so small" final times. 


.. _krylov-qutip:

Krylov Solver in QuTiP
====================

In QuTiP, Krylov-subspace evolution is implemented as the function :func:`qutip.krylovsolve`. Arguments are nearly the same as :func:`qutip.mesolve`
function for master-equation evolution, except that the initial state must be a ket vector, as oppose to a density matrix, and the additional parameter ``krylov_dim`` that defines the maximum allowed Krylov-subspace dimension. The maximum number of allowed Lanczos partitions can also be determined using the :func:`qutip.solver.options.nsteps` parameter, which defaults to '10000'. 
