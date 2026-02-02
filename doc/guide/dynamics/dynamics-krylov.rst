.. _krylov:

*******************************************
Krylov Solver
*******************************************

.. _krylov-intro:

Introduction
=============

The Krylov-subspace method is a standard method to approximate quantum dynamics.
Let :math:`\left|\psi\right\rangle` be a state in a :math:`D`-dimensional
complex Hilbert space that evolves under a time-independent Hamiltonian :math:`H`.
Then, the :math:`N`-dimensional Krylov subspace associated with that state and
Hamiltonian is given by

.. math::
	:label: krylovsubspace

	\mathcal{K}_{N}=\operatorname{span}\left\{|\psi\rangle, H|\psi\rangle, \ldots, H^{N-1}|\psi\rangle\right\},

where the dimension :math:`N<D` is a parameter of choice. To construct an
orthonormal basis :math:`B_N` for :math:`\mathcal{K}_{N}`, the simplest algorithm
is the well-known Lanczos algorithm, which provides a sort of Gram-Schmidt procedure
that harnesses the fact that orthonormalization needs to be imposed only for the last
two vectors in the basis. Written in this basis the time-evolved state can be approximated as

.. math::
	:label: lanczoskrylov

	|\psi(t)\rangle=e^{-iHt}|\psi\rangle\approx\mathbb{P}_{N}e^{-iHt}\mathbb{P}_{N}|\psi\rangle=\mathbb{V}_{N}^{\dagger}e^{-iT_{N}t}\mathbb{V}_{N}|\psi\rangle\equiv\left|\psi_{N}(t)\right\rangle,

where  :math:`T_{N}=\mathbb{V}_{N} H \mathbb{V}_{N}^{\dagger}` is the Hamiltonian
reduced to the Krylov subspace (which takes a tridiagonal matrix form), and
:math:`\mathbb{V}_{N}^{\dagger}` is the matrix containing the vectors of the
Krylov basis as columns.

With the above approximation, the time-evolution is calculated only with a
smaller square matrix of the desired size. Therefore, the Krylov method provides
huge speed-ups in computation of short-time evolutions when the dimension of the
Hamiltonian is very large, a point at which exact calculations on the complete
subspace are practically impossible.

One of the biggest problems with this type of method is the control of the error.
After a short time, the error starts to grow exponentially. However, this can be
easily corrected by restarting the subspace when the error reaches a certain
threshold. Therefore, a series of :math:`M` Krylov-subspace time evolutions
provides accurate solutions for the complete time evolution. Within this scheme,
the magic of Krylov resides not only in its ability to capture complex time evolutions
from very large Hilbert spaces with very small dimenions :math:`M`, but also in
the computing speed-up it presents.

For exceptional cases, the Lanczos algorithm might arrive at the exact evolution
of the initial state at a dimension :math:`M_{hb}<M`. This is called a happy
breakdown. For example, if a Hamiltonian has a symmetry subspace :math:`D_{\text{sim}}<M`,
then the algorithm will optimize using the value math:`M_{hb}<M`:, at which the
evolution is not only exact but also cheap.

.. _krylov-qutip:

Krylov Solver in QuTiP
======================

In QuTiP, Krylov-subspace evolution is implemented as the function :func:`.krylovsolve`.
Arguments are nearly the same as :func:`.sesolve` function for master-equation
evolution, except that the Hamiltonian cannot depend on time, the initial state
must always be a ket vector, (it cannot be used to compute propagators) and an
additional parameter ``krylov_dim`` is needed. ``krylov_dim`` defines the
maximum allowed Krylov-subspace dimension.

Let's solve a simple example using the algorithm in QuTiP to get familiar with the method.

.. plot::
    :context: reset

    >>> dim = 100
    >>> jx = jmat((dim - 1) / 2.0, "x")
    >>> jy = jmat((dim - 1) / 2.0, "y")
    >>> jz = jmat((dim - 1) / 2.0, "z")
    >>> e_ops = [jx, jy, jz]
    >>> H = (jz + jx) / 2
    >>> psi0 = rand_ket(dim, seed=1)
    >>> tlist = np.linspace(0.0, 10.0, 200)
    >>> results = krylovsolve(H, psi0, tlist, krylov_dim=20, e_ops=e_ops)
    >>> plt.figure()
    >>> for expect in results.expect:
    >>>    plt.plot(tlist, expect)
    >>> plt.legend(('jmat x', 'jmat y', 'jmat z'))
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Expectation values')
    >>> plt.show()

.. plot::
    :context: reset
    :include-source: false
    :nofigs:
