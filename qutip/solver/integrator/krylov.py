import numpy as np
import warnings
from math import factorial
from qutip.core import data as _data
from qutip.core import liouvillian
from ..integrator import IntegratorException, Integrator
from ..sesolve import SESolver
from ..mesolve import MESolver


__all__ = ["IntegratorKrylov"]


class IntegratorKrylov(Integrator):
    """
    Evolve the state ("rho0") finding an approximation for the time evolution
    operator of a Hamiltonian ("H") by obtaining the projection on a set of
    small dimensional Krylov subspaces (m <= dim(H)). The construction of this
    subspace is performed by the Lanczos, fully-reorthogonalized Lanczos or
    Arnoldi algorithm.
    """
    integrator_options = {
        'atol': 1e-7,
        'nsteps': 100,
        'max_step': 1e5,
        'min_step': 1e-5,
        'always_compute_step': False,
        'krylov_dim': 0,
        'sub_system_tol': 1e-7,
        'algorithm': 'auto',
    }
    support_time_dependant = False
    supports_blackbox = False
    method = 'krylov'

    def _prepare(self):
        if not self.system.isconstant:
            raise ValueError("Krylov method only supports constant systems.")

        if self.options["krylov_dim"] <= 0:
            raise ValueError("The option 'krylov_dim', must be an integer "
                             "greater than zero.")

        self._max_step = -np.inf
        self._krylov_dim = self.options["krylov_dim"]
        self._hermitian = (1j*self.system(0)).isherm

        if self.options['algorithm'] == 'auto':
            if self._hermitian:
                self._algorithm = self._lanczos_full_reorth_algorithm
            else:
                self._algorithm = self._arnoldi_algorithm
        elif self.options['algorithm'] == 'arnoldi':
            self._algorithm = self._arnoldi_algorithm
        elif self.options['algorithm'] == 'lanczos_fro':
            self._algorithm = self._lanczos_full_reorth_algorithm
        elif self.options['algorithm'] == 'lanczos':
            self._algorithm = self._lanczos_algorithm
        else:
            raise ValueError("The requested algorithm "
                             f"{self.options['algorithm']} "
                             "for Krylov space construction is not available. "
                             "Possible options are: \'lanczos\', "
                             "\'lanczos_fro\', \'arnoldi\'.")

        if not self._hermitian and self._algorithm != self._arnoldi_algorithm:
            # Arnoldi is the only algorithm for open systems in QuTiP atm
            raise ValueError(f"The requested Krylov algorithm "
                             f"{self.options['algorithm']} "
                             "is not supported for non-Hermitian systems.")

    def _lanczos_algorithm(self, psi):
        return self._lanczos_core(psi, max_orthog_steps=2)

    def _lanczos_full_reorth_algorithm(self, psi):
        return self._lanczos_core(psi)

    def _lanczos_core(self, psi , max_orthog_steps=0):
        """
        Computes a basis of the Krylov subspace for the time independent
        Hamiltonian 'H', a system state 'psi' and Krylov dimension 'krylov_dim'
        using the Lanczos algorithm with a given orthogonalization function.
        The space is spanned by
        {psi, H psi, H^2 psi, ..., H^(krylov_dim - 1) psi}.

        Parameters
        ------------
        psi: np.ndarray
            State used to calculate Krylov subspace (= first basis state).
        max_orthog_steps: int
            Maximum number of previous basis vectors to reorthogonalize against.

        Returns
        ------------
        krylov_trid: np.ndarray
            The tridiagonal matrix of the Krylov subspace.
        krylov_basis: np.ndarray
            The basis vectors of the Krylov subspace.
        """
        krylov_dim = self._krylov_dim
        H = (1j * self.system(0)).data
        p0 = _data.inner(psi, psi)  # purity
        sp0 = np.sqrt(p0)

        diag = np.zeros(krylov_dim, dtype=complex)
        subdiag = np.zeros(krylov_dim, dtype=complex)
        Q = [psi]

        k = 1
        v = _data.matmul(H, Q[-1])
        diag[0] = _data.inner(Q[-1], v) / p0
        v = _data.add(v, Q[-1], -diag[0])
        subdiag[0] = _data.norm.l2(v) / sp0
        while k < krylov_dim and subdiag[k-1] > self.options['sub_system_tol']:
            Q.append(_data.mul(v, 1 / subdiag[k-1]))
            v = _data.matmul(H, Q[-1])
            k += 1
            v, ol = self._orthogonalize(Q, v, p0, steps=max_orthog_steps)
            diag[k-1] = ol
            subdiag[k-1] = _data.norm.l2(v) / sp0

        krylov_trid = _data.diag["dense"](
            [subdiag[:k-1], diag[:k], subdiag[:k-1]],
            [-1, 0, 1]
        )
        krylov_basis = _data.Dense(np.hstack([p.to_array() for p in Q]))

        return krylov_trid, krylov_basis

    def _orthogonalize(self, Q, v, p0, steps=0):
        """
        Orthogonalizes a new vector `v` against the previous `max_orthog_steps`
        number of Krylov basis vectors in the list `Q`.
        
        Parameters
        ------------
        Q: list of np.ndarray
            The list of previous Krylov basis vectors.
        v: np.ndarray
            The new vector to orthogonalize.
        p0: float
            The purity of the initial state.
        steps: int, default: 0
            The number of previous basis vectors to reorthogonalize against.
            The default `0` will reorthogonalize w.r.t all.

        Returns
        ------------
        v: np.ndarray
            The orthogonalized vector = new Krylov basis vector.
        ol: float
            The overlap of the orthogonalized vector with the last basis vector
            in `Q`. This will be the new off diagonal element of the tridiagonal
            matrix. 
        """
        ol = 0
        for q in Q[-steps:]:
            ol = _data.inner(q, v) / p0
            v = _data.add(v, q, -ol)
        return v, ol
        
    def _arnoldi_algorithm(self, psi):
        """
        Computes the Krylov subspace basis for a Hamiltonian 'H', a system
        state 'psi' and Krylov dimension 'krylov_dim' using the Arnoldi
        interation. This results in an upper Hessenberg matrix that in general
        is non-Hermitian. The space is spanned by
        {psi, H psi, H^2 psi, ..., H^(krylov_dim - 1) psi}.

        Parameters
        ------------
        psi: np.ndarray
            State used to calculation Krylov subspace (= first basis state).
        
        Returns
        ------------
        krylov_hesse: np.ndarray
            The upper triangular matrix of the Krylov subspace.
        krylov_basis: np.ndarray
            The basis vectors of the Krylov subspace.
        """
        krylov_dim = self._krylov_dim
        H = (1j * self.system(0)).data
        p0 = _data.inner(psi, psi)  # purity
        sp0 = np.sqrt(p0)

        h = np.zeros((krylov_dim + 1, krylov_dim), dtype=complex)
        Q = [psi]

        k = 1
        v = _data.matmul(H, Q[-1])
        h[0, 0] = _data.inner(Q[-1], v) / p0
        v = _data.add(v, Q[-1], -h[0, 0])
        h[1, 0] = _data.norm.l2(v) / sp0
        while k < krylov_dim and h[k, k-1] > self.options['sub_system_tol']:
            Q.append(_data.mul(v, 1 / h[k, k-1]))
            v = _data.matmul(H, Q[-1])
            k += 1
            for j in range(k):  # removes projections, create upper Hessenberg
                h[j, k-1] = _data.inner(Q[j], v) / p0
                v = _data.add(v, Q[j], -h[j, k-1])
            h[k, k-1] = _data.norm.l2(v) / sp0

        krylov_hesse = _data.Dense(h[:k, :k])
        krylov_basis = _data.Dense(
            np.hstack([p.to_array() for p in Q])
        )
        return krylov_hesse, krylov_basis

    def _compute_krylov_set(self, krylov_tridiag, krylov_basis):
        """
        Compute the eigen energies, basis transformation operator (U) and e0.
        """
        evals, evecs = _data.eigs(krylov_tridiag, self._hermitian)
        N = evals.shape[0]
        U = _data.matmul(krylov_basis, evecs)

        e0 = _data.one_element_dense((N, 1), (0, 0), 1.0)
        if self._hermitian:
            e0 = evecs.adjoint() @ e0
        else:
            e0 = _data.inv(evecs) @ e0

        return evals, U, e0

    def _compute_psi(self, dt, eigenvalues, U, e0):
        """
        Compute the state at time ``t``.
        """
        phases = _data.Dense(np.exp(-1j * dt * eigenvalues))
        aux = _data.multiply(phases, e0)
        return _data.matmul(U, aux)

    def _compute_max_step(
        self,
        krylov_tridiag,
        krylov_basis,
        krylov_state=None
    ):
        """
        Compute the maximum step length to stay under the desired tolerance.
        """
        if not krylov_state:
            krylov_state = \
                self._compute_krylov_set(krylov_tridiag, krylov_basis)

        bsprod = np.prod(np.diag(krylov_tridiag.as_ndarray(), k=-1))
        num = self.options["atol"] * factorial(krylov_tridiag.shape[0])
        dt = np.real(np.power(num / bsprod, 1 / krylov_tridiag.shape[0]))
        if dt < self.options["min_step"]:
            raise ValueError(
                f"With the krylov dimension of {self.options['krylov_dim']} "
                f"and desired tolerance of {self.options['atol']}, the maximum "
                f"possible time step size is {dt}. But is smaller than the "
                f"minimum desired time step size of {self.options['min_step']}."
            )
        return np.min(dt, self.options["max_step"])
        
    def set_state(self, t, state0):
        self._t_0 = t

        if state0.shape[1] > 1 and not self.system.issuper:
            self.system = -1j * liouvillian(self.system)

        krylov_tridiag, krylov_basis = self._algorithm(state0)
        self._krylov_state = \
            self._compute_krylov_set(krylov_tridiag, krylov_basis)

        if (
            krylov_tridiag.shape[0] < self._krylov_dim
            or krylov_tridiag.shape == self.system.shape
        ):
            # happy_breakdown
            self._max_step = np.inf
            return

        if (
            not np.isfinite(self._max_step)
            or self.options["always_compute_step"]
        ):
            self._max_step = self._compute_max_step(krylov_tridiag, krylov_basis)

    def get_state(self, copy=True):
        return self._t_0, self._compute_psi(0, *self._krylov_state)

    def integrate(self, t, copy=True):
        step = 0
        while t > self._t_0 + self._max_step:
            # The approximation in only valid in the range t_0, t_0 + max step
            # If outside, advance the range
            step += 1
            if step >= self.options["nsteps"]:
                raise IntegratorException(
                    "Maximum number of integration steps "
                    f"({self.options['nsteps']}) exceeded. "
                    "Increase the number of steps or krylov dimension or "
                    "reduce tolerance."
                )
            new_psi = self._compute_psi(self._max_step, *self._krylov_state)
            self.set_state(self._t_0 + self._max_step, new_psi)

        delta_t = t - self._t_0
        return t, self._compute_psi(delta_t, *self._krylov_state)

    @property
    def options(self):
        """
        Supported options by krylov method:

        atol : float, default: 1e-7
            Absolute tolerance.

        nsteps : int, default: 100
            Max. number of internal steps/call.

        min_step, max_step : float, default: (1e-5, 1e5)
            Minimum and maximum time step size before the Krylov basis is
            recalculated.
        
        krylov_dim: int, default: 0
            Dimension of Krylov approximation subspaces used for the time
            evolution approximation.

        algorithm: str, default: "auto"
            Algorithm for Krylov space constructions. The default ``auto`` will
            choose ``lanczos_fro`` for Hermitian and ``arnoldi`` for
            non-Hermitian systems. Alternatively the standard ``lanczos`` can be
            set.

        sub_system_tol: float, default: 1e-7
            Tolerance to detect a happy breakdown. A happy breakdown occurs
            when the initial ket is in a subspace of the Hamiltonian smaller
            than ``krylov_dim``.

        always_compute_step: bool, default: False
            If True, the step length is computed each time a new Krylov
            subspace is computed. Otherwise it is computed only once when
            creating the integrator.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


SESolver.add_integrator(IntegratorKrylov, 'krylov')
MESolver.add_integrator(IntegratorKrylov, 'krylov')
