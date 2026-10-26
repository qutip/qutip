.. _krylov:

*******************************************
Krylov Solver
*******************************************

.. _krylov-intro:

The Krylov Subspace Method
==========================

To approximate quantum dynamics in systems with a high number of degrees of
freedom, the Krylov-subspace method is a standard method for efficient
calculation.
Let :math:`\left|\psi\right\rangle` be a state in a :math:`D`-dimensional
complex Hilbert space that evolves under a time-independent Hamiltonian :math:`H`.
Then, the :math:`N`-dimensional Krylov subspace associated with that state and
Hamiltonian is given by

.. math::
	:label: krylovsubspace

	\mathcal{K}_{N}=\operatorname{span}\left\{|\psi\rangle, H|\psi\rangle, \ldots, H^{N-1}|\psi\rangle\right\},

where the dimension :math:`N<D` is a parameter of choice. To construct an
orthonormal basis for :math:`\mathcal{K}_{N}`, the simplest algorithm is the
well-known Lanczos algorithm, which provides a sort of Gram-Schmidt procedure.
Written in this basis the time-evolved state can be approximated as

.. math::
	:label: lanczoskrylov

	|\psi(t)\rangle=e^{-iHt}|\psi\rangle\approx\mathbb{P}_{N}e^{-iHt}\mathbb{P}_{N}|\psi\rangle=\mathbb{V}_{N}^{\dagger}e^{-iT_{N}t}\mathbb{V}_{N}|\psi\rangle\equiv\left|\psi_{N}(t)\right\rangle,

where :math:`T_{N}=\mathbb{V}_{N} H \mathbb{V}_{N}^{\dagger}` is the Hamiltonian
reduced to the Krylov subspace (which takes a tridiagonal matrix form), and
:math:`\mathbb{V}_{N}^{\dagger}` is the matrix containing the vectors of the
Krylov basis as columns.

With the above approximation, the time-evolution is calculated only with a
smaller square matrix of the desired size. Therefore, the Krylov method provides
huge speed-ups in computation of short-time evolutions when the dimension of the
Hamiltonian is very large, a point at which exact calculations on the complete
subspace are practically impossible.

One of the biggest problems with this type of method is the control of the
error.  If :math:`N` is smaller than the dimension :math:`D`, the error
grows exponentially once a certain time threshold is hit. This threshold can be
bounded a priori using the lower off-diagonal elements :math:`\beta_j` of the
tridiagonal matrix :math:`T_N` obtained from the Lanczos / Arnoldi algorithm.
The error of the approximation is estimated by

.. math::
	:label: krylov_error

	\left\| |\psi(t)\rangle - |\psi_N(t)\rangle \right\|
	\lesssim \frac{t^{N}}{N!} \prod_{j=1}^{N-1} \beta_j ,

so that requiring it to stay below the tolerance ``atol`` gives the maximal
time step

.. math::
	:label: krylov_max_step

	\Delta t_{\max} = \left( \frac{\mathrm{atol} \cdot N!}
	{\prod_{j=1}^{N-1} \beta_j} \right)^{1/N}.

After each step of length :math:`\Delta t_{\max}`, the Krylov subspace is
rebuilt from the current state. To reach an arbitrary simulation time, a series
of :math:`M` such Krylov-subspace time evolutions therefore provides accurate
solutions for the complete time evolution. Within this scheme, the magic of
Krylov resides not only in its ability to capture complex time evolutions from
very large Hilbert spaces with very small dimensions :math:`N`, but also in the
computing speed-up it presents.

Exceptional cases can lead to the Krylov subspace closing under the action of
:math:`H` at a dimension :math:`N_{hb}<N`. This is called a happy breakdown. It
happens, for example, if the initial state lies entirely within a symmetry
sector of the Hamiltonian of dimension :math:`D_{\text{sym}}<N`, in which case
:math:`N_{hb} \leq D_{\text{sym}}`. In QuTiP, a happy breakdown is detected
when the norm of the residual vector falls below the option
``sub_system_tol``. The construction then stops at :math:`N_{hb}`, and the
evolution is not only exact for all times (no restarts are needed), but also
cheap.

Krylov for Mixed States
-----------------------

In a similar manner, the Liouvillian :math:`\mathcal{L}` of a system can be
taken as an input for the construction of the Krylov basis.  This extends the
applicability of this method to density operator dynamics in both closed and
open systems. In these cases, the subspace is spanned by repeated application of
the Liouvillian onto the vectorized form of the density operator, thereby
forming

.. math::
	:label: krylovsubspace_mixed

	\mathcal{K}_{N}=\operatorname{span}\left\{|\rho), \mathcal{L}|\rho), \ldots, \mathcal{L}^{N-1}|\rho)\right\}.

Same as for the pure states mentioned above, the Krylov basis vectors are
calculated using one of the algorithms below.

Matrix-Form Krylov Evolution
----------------------------

Building the Liouvillian :math:`\mathcal{L}` explicitly requires storing a
:math:`D^2 \times D^2` superoperator, which quickly becomes a memory
bottleneck for large systems. As an alternative, the Krylov basis can be
constructed in *matrix form*, analogous to the matrix-form solver of
:func:`.mesolve` (see :ref:`master-matrix-form`). Here, the Krylov basis
elements are kept as :math:`D \times D` operators and the action of the
Liouvillian is evaluated directly with matrix-matrix products.
Orthogonalization of the basis elements uses the Hilbert-Schmidt inner product
:math:`(A|B) = \operatorname{tr}(A^\dagger B)`.
Since the Krylov basis elements are in general not Hermitian, they are treated
as general operators during the construction. Consequently, the faster
Hermitian-state path of the matrix-form :func:`.mesolve`, cannot be used for
Krylov evolution.

Note that the matrix form only removes the cost of storing :math:`\mathcal{L}`.
The Krylov basis itself still consists of :math:`N` dense elements of size
:math:`D^2`, i.e. it requires :math:`N D^2` complex numbers in both forms.
For sparse Liouvillians, whose number of non-zero elements is often much
smaller than :math:`D^4`, the memory savings are therefore moderate.  They are
largest for dense Hamiltonians or systems with many collapse operators.

Alternative Algorithms for Krylov Construction
----------------------------------------------

As the number of degrees of freedom grows, the standard Lanczos algorithm shows
significant errors due to numerical inaccuracies.
This is why the *default* algorithm in QuTiP to construct the Krylov basis is
the *fully-reorthogonalized* Lanczos algorithm (FRO) for closed systems and
the Arnoldi iteration for open systems.  It orthogonalizes new basis vectors
with respect to *all* (contrary to just the last two) previously created vectors
to greatly limit the accumulation of numerical errors.  However, by specifically
setting ``options["algorithm"]="lanczos"``, the standard Lanczos algorithm is
still available.

The Arnoldi iteration extends the Krylov space methods to open quantum systems
described by a Liouvillian.  It is the default method provided ``c_ops`` are
given.  Contrary to the standard Lanczos and FRO Lanczos which produce a
tri-diagonal matrix, the Arnoldi iteration provides a non-Hermitian
upper-Hessenberg form.

Finding a Good Krylov-Subspace Dimension
----------------------------------------

With the Krylov-subspace method, great computational speed-ups can be achieved
especially for large systems and short time evolutions. In QuTiP, the integrator
calculates the subspace up to the parameter ``krylov_dim`` (or until a happy
breakdown is detected). It then evolves the state up to the time where accuracy
up to the provided tolerance ``atol`` is still guaranteed, and then recalculates
the subspace. By repeating this, the time evolution is calculated.

The step length :math:`\Delta t_{\max}` from :eq:`krylov_max_step` can be
further controlled through the options ``max_step`` and ``min_step``. The step
is capped at ``max_step`` (default ``1e5``). If the estimated step falls below
``min_step`` (default ``1e-5``), the solver raises a ``ValueError``, since the
chosen ``krylov_dim`` is too small to reach the tolerance ``atol`` in a
reasonable time. In this case, increasing ``krylov_dim`` or relaxing ``atol``
is recommended. By default, the step length is estimated only once, for the
first Krylov subspace, and then reused. Setting ``always_compute_step=True``
re-estimates it every time a new subspace is built, which can be beneficial if
the dynamics change considerably over time.

To achieve the promised speed-up, it is essential to choose a suitable
Krylov-subspace dimension. Too small a dimension leads to frequent recalculation
of the Krylov basis and thus high computational load. Too big of a basis
however leads to an increasingly expensive calculation of basis vectors and
little gain in accuracy.

Therefore, aiming for a Krylov-dimension that is much smaller than the provided
Hilbert space (or Liouvillian space) is recommended. However, the optimal
dimension very much depends on both the Hamiltonian and the initial state.

.. _krylov-qutip:

Krylov Solver in QuTiP
======================

In QuTiP, Krylov-subspace evolution is implemented as the function
:func:`.krylovsolve`. Arguments are nearly the same as :func:`.sesolve` or
:func:`.mesolve` function for master-equation evolution, except that the
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
    >>>    plt.plot(tlist, expect)
    >>> plt.legend(('jmat x', 'jmat y', 'jmat z'))
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Expectation values')
    >>> plt.show()

Open systems are solved in the same way by passing collapse operators.
As an example, consider a damped harmonic oscillator with Hamiltonian
:math:`H = \omega a^\dagger a` and collapse operator :math:`\sqrt{\kappa} a`,
initialized in a coherent state :math:`|\alpha\rangle` with real
:math:`\alpha`. The state stays coherent with a decaying amplitude, so the
expectation values are known analytically:

.. math::

   \langle a^\dagger a \rangle(t) = |\alpha|^2 e^{-\kappa t}, \qquad
   \langle a + a^\dagger \rangle(t) = 2 \alpha e^{-\kappa t / 2}
   \cos(\omega t).

Here, the Arnoldi iteration is chosen automatically. For density-matrix
evolution, the matrix-form Krylov construction described above can be enabled
by passing ``matrix_form=True`` in the options:

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
    ...                       e_ops=e_ops, options={"matrix_form": True})
    >>> n_exact = alpha**2 * np.exp(-kappa * tlist)
    >>> x_exact = (2 * alpha * np.exp(-kappa * tlist / 2)
    ...            * np.cos(omega * tlist))
    >>> plt.figure()
    >>> plt.plot(tlist, n_exact, 'k-', label='analytical')
    >>> plt.plot(tlist, x_exact, 'k-')
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
