.. _krylov:

*******************************************
Krylov Solver
*******************************************

.. _krylov-intro:

The Krylov Subspace Method
==========================

To calculate quantum dynamics in systems with a high number of degrees of
freedom, the Krylov-subspace method is a standard method for efficient
calculation. Let :math:`\left|\psi\right\rangle` be a state in a
:math:`D`-dimensional complex Hilbert space that evolves under a
time-independent Hamiltonian :math:`H`. Then, the :math:`N`-dimensional Krylov
subspace associated with that initial state and Hamiltonian is given by

.. math::
	:label: krylovsubspace

	\mathcal{K}_{N}=\operatorname{span}\left\{|\psi\rangle, H|\psi\rangle, \ldots, H^{N-1}|\psi\rangle\right\},

where the dimension :math:`N<D` is a parameter of choice. To construct an
orthonormal basis for :math:`\mathcal{K}_{N}`, the simplest algorithm is the
well-known Lanczos algorithm, which provides a sort of Gram-Schmidt procedure
and casts the Hamiltonian in a tridiagonal form. Written in this basis the
time-evolved state can be calculated as

.. math::
	:label: lanczoskrylov

	|\psi(t)\rangle=e^{-iHt}|\psi\rangle\approx\mathbb{P}_{N}e^{-iHt}\mathbb{P}_{N}|\psi\rangle=\mathbb{V}_{N}^{\dagger}e^{-iT_{N}t}\mathbb{V}_{N}|\psi\rangle\equiv\left|\psi_{N}(t)\right\rangle,

where :math:`T_{N}=\mathbb{V}_{N} H \mathbb{V}_{N}^{\dagger}` is the Hamiltonian
reduced to the Krylov subspace, and :math:`\mathbb{V}_{N}^{\dagger}` is the
matrix containing the vectors of the Krylov basis as columns.

With the above representation, the time evolution is calculated only with a
smaller square matrix of the desired size. Therefore, the Krylov method provides
huge speed-ups in computation of short-time evolutions when the dimension of the
Hamiltonian is very large, a point at which exact calculations on the full
Hilbert space are practically impossible.

The size of the Krylov subspace has fundamental influence on the efficiency of
this method. In general, if :math:`N<D` (or :math:`N<D^2-D+1` for open
systems), *one* Krylov subspace cannot correctly represent the dynamics of the
system for arbitrary times. It must either be extended to the maximum dimension
or a new Krylov subspace must be constructed repeatedly (see below for more
details).

Exceptional cases can lead to the Krylov subspace closing under the action of
:math:`H` even at a dimension :math:`N_{\text{hb}}<N`. This is called a happy
breakdown. It happens, for example, if the initial state lies entirely within a
symmetry sector of the Hamiltonian of dimension :math:`D_{\text{sym}}<N`, in
which case :math:`N_{\text{hb}} \leq D_{\text{sym}}`. In QuTiP, a happy breakdown is
detected when the norm of the residual vector falls below the option
``sub_system_tol``. The construction then stops at :math:`N_{\text{hb}}`, and the
evolution is not only exact for all times (no restarts are needed), but also
cheap.

.. _krylov-closed:

Krylov Solver in QuTiP
======================

In QuTiP, Krylov-subspace evolution is implemented as the function
:func:`.krylovsolve`. Arguments are nearly the same as those of :func:`.sesolve`
(or :func:`.mesolve` for master-equation evolution), except that the
Hamiltonian and collapse operators cannot depend on time and an additional
parameter ``krylov_dim`` is needed. ``krylov_dim`` defines the maximum allowed
Krylov-subspace dimension.

Let's solve a simple example using the algorithm in QuTiP to get familiar with
the method.

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
    ...    plt.plot(tlist, expect)
    >>> plt.legend(('jmat x', 'jmat y', 'jmat z'))
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Expectation values')
    >>> plt.show()

.. _krylov-mixed:

Krylov for Mixed States
=======================

In a similar manner, the Liouvillian :math:`\mathcal{L}` of a system can be
taken as an input for the construction of the Krylov basis. This extends the
applicability of this method to density operator dynamics in both closed and
open systems. In these cases, the subspace is spanned by repeated application of
the Liouvillian onto the vectorized form of the density operator, thereby
forming

.. math::
	:label: krylovsubspace_mixed

	\mathcal{K}_{N}=\operatorname{span}\left\{|\rho), \mathcal{L}|\rho), \ldots, \mathcal{L}^{N-1}|\rho)\right\}.

In order to find an orthonormal basis for this space with respect to the
Hilbert-Schmidt inner product :math:`(A|B) = \operatorname{tr}(A^\dagger B)`,
the fully reorthogonalized Lanczos (FRO) algorithm (for closed systems) or the
Arnoldi iteration (for open systems) is chosen by default (see
:ref:`krylov-algorithms`).

.. plot::
    :context: reset

    >>> N = 20
    >>> a = destroy(N)
    >>> omega, kappa, alpha = 1.0, 0.2, 2.0
    >>> H = omega * a.dag() * a
    >>> c_ops = [np.sqrt(kappa) * a]
    >>> rho0 = coherent_dm(N, alpha)
    >>> e_ops = [a.dag() * a, a + a.dag()]
    >>> tlist = np.linspace(0.0, 20.0, 200)
    >>> results = krylovsolve(H, rho0, tlist, krylov_dim=20, c_ops=c_ops,
    ...                       e_ops=e_ops)
    >>> plt.figure()
    >>> plt.plot(tlist[::5], results.expect[0][::5], 'o',
    ...          label=r'$\langle a^\dagger a \rangle$ (Krylov)')
    >>> plt.plot(tlist[::5], results.expect[1][::5], 's',
    ...          label=r'$\langle a + a^\dagger \rangle$ (Krylov)')
    >>> plt.legend()
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Expectation values')
    >>> plt.show()

.. plot::
    :context: reset
    :include-source: false
    :nofigs:

.. _krylov-matrix-form:

Matrix-Form Krylov Evolution
============================

Analogous to the matrix-form solver of :func:`.mesolve` (see
:ref:`master-matrix-form`), the Krylov basis can also be constructed in
*matrix form*. Here, the Krylov basis elements are kept as :math:`D \times D`
operators and the action of the Liouvillian is evaluated directly with
matrix-matrix products. Orthogonalization of the basis elements uses the
Hilbert-Schmidt inner product.

Since the Krylov basis elements are, in general, non-Hermitian, they are treated
as general operators during the construction. Consequently, the faster
Hermitian-state path of the matrix-form :func:`.mesolve` is not used for Krylov
evolution.

Note that the matrix form only removes the cost of storing :math:`\mathcal{L}`.
The Krylov basis itself still consists of :math:`N` dense elements of size
:math:`D^2`, i.e. it requires :math:`N D^2` complex numbers in both forms.
For sparse Liouvillians, whose number of non-zero elements is often much
smaller than :math:`D^4`, the memory savings are therefore moderate. They are
largest for dense Hamiltonians or systems with many collapse operators.

.. _krylov-algorithms:

Alternative Algorithms for Krylov Construction
==============================================

As the number of degrees of freedom grows, the standard Lanczos algorithm shows
significant errors due to numerical inaccuracies. This is why the *default*
algorithm in QuTiP to construct the Krylov basis is the
*fully reorthogonalized* Lanczos algorithm (FRO) for closed systems and the
Arnoldi iteration for open systems. The FRO algorithm orthogonalizes new basis
vectors with respect to *all* (as opposed to just the last two) previously
created vectors to greatly limit the accumulation of numerical errors. However,
by specifically setting ``options["algorithm"]="lanczos"``, the standard Lanczos
algorithm is still available.

The Arnoldi iteration extends Krylov-subspace methods to open quantum systems
described by a Liouvillian. It is the default method provided ``c_ops`` are
given. Unlike the standard and FRO Lanczos algorithms, which produce a
tridiagonal matrix, the Arnoldi iteration provides a non-Hermitian
upper Hessenberg form.

.. _krylov-dimension:

Finding a Good Krylov-Subspace Dimension
========================================

With the Krylov-subspace method, great computational speed-ups can be achieved
especially for large, dense systems and short-time evolutions. In QuTiP, the
integrator calculates the subspace up to the parameter ``krylov_dim`` (or until
a happy breakdown is detected). It then evolves the state up to the time
:math:`t + \Delta t_{\max}`, where accuracy up to the provided tolerance
``atol`` is still guaranteed, and then recalculates the subspace. By repeating
this, the full time evolution is calculated.

The size of :math:`\Delta t_{\max}` is naturally connected to the Krylov
dimension. It can be calculated a priori, using the lower off-diagonal elements
(Lanczos coefficients) of the tridiagonal (upper Hessenberg) matrix obtained
from the Lanczos (Arnoldi) algorithm. Naturally, a larger Krylov subspace leads
to a bigger :math:`\Delta t_{\max}`.

The step length :math:`\Delta t_{\max}` can also be controlled through the
``max_step`` and ``min_step`` options. The step is capped at ``max_step``
(default ``1e5``). If the estimated step falls below ``min_step`` (default
``1e-5``), the solver raises a ``ValueError``, since the chosen ``krylov_dim``
is too small to reach the tolerance ``atol`` in a reasonable time. In this case,
increasing ``krylov_dim`` or relaxing ``atol`` is recommended. By default, the
step length is estimated only once, for the first Krylov subspace, and then
reused. Setting ``always_compute_step=True`` re-estimates it every time a new
subspace is built, which can be beneficial if the dynamics change considerably
over time.

To achieve the promised speed-up, it is essential to choose a suitable
Krylov-subspace dimension. Too small a dimension leads to frequent recalculation
of the Krylov basis and thus high computational load. Too large a basis,
however, leads to an increasingly expensive calculation of basis vectors and
little gain in time.

Generally, aiming for a Krylov dimension that is much smaller than the dimension
of the Hilbert space (or Liouville space) is recommended. However, the optimal
dimension very much depends on both the Hamiltonian and the initial state.
