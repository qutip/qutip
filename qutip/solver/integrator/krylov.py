from ..integrator import IntegratorException, Integrator
import numpy as np
from qutip.core import data as _data
from scipy.optimize import root_scalar
from scipy.special import factorial
from ..sesolve import SESolver
from ..mesolve import MESolver


__all__ = ["IntegratorKrylov"]


class IntegratorKrylov(Integrator):
    """
    Evolve the state ("rho0") finding an approximation for the time evolution
    operator of a hermitian Hamiltonian ("H") by obtaining the projection of the
    time evolution operator on a set of small dimensional Krylov subspaces (m <=
    dim(H)).
    """
    integrator_options = {
        'atol': 1e-7,
        'nsteps': 100,
        'max_step': 1e5,
        'krylov_dim': 0,
        'sub_system_tol': 1e-7,
        'algorithm': 'lanczos_fro',
    }
    support_time_dependant = False
    supports_blackbox = False
    method = 'krylov'

    def _prepare(self):
        if not self.system.isconstant:
            raise ValueError("Krylov method only supports constant systems.")
        # TODO currenlty only supports hermitian, add warning? thorw error?
        self._max_step = -np.inf
        krylov_dim = self.options["krylov_dim"]
        if krylov_dim < 0 or krylov_dim > self._max_krylov_dim():
            raise ValueError("The options 'krylov_dim', must be a positive "
                             "integer that does not exceed the maximum "
                             "dimension")
        if krylov_dim == 0:
            krylov_dim = self._max_krylov_dim()
            self.options["krylov_dim"] = krylov_dim
        
        if self.options['algorithm'] == 'lanczos':
            self._algorithm = self._lanczos_algorithm
        elif self.options['algorithm'] == 'lanczos_fro':
            self._algorithm = self._lanczos_full_reorth_algorithm
        elif self.options['algorithm'] == 'arnoldi':
            self._algorithm = self._arnoldi_algorithm
        else:
            raise ValueError("The requested algorithm "
                             f"{self.options['algorithm']}"
                             "for Krylov space construction is not available. "
                             "Possible options are: \'lanczos\', "
                             "\'lanczos_fro\', \'arnoldi\'.")

        self._max_step = np.inf

    def _max_krylov_dim(self):
        """
        Calculates the maximum dimension of the Krylov space for the provided
        Hamiltonian or Liouvillian.

        Returns
        ------------
        dims: int
            Maximum dimension the Krylov space can have.
        """
        if self.system.issuper:
            d = self.system.dims[0][0][0]
            return d**2 - d + 1
        else:
            return self.system.shape[0]

    def _lanczos_algorithm(self, psi):
        """
        Computes a basis of the Krylov subspace for the time independent
        Hamiltonian 'H', a system state 'psi' and Krylov dimension 'krylov_dim'
        using the Lanczos algorithm. The space is spanned by
        {psi, H psi, H^2 psi, ..., H^(krylov_dim - 1) psi}.

        Parameters
        ------------
        psi: np.ndarray
            State used to calculate Krylov subspace (= first basis state).
        """
        krylov_dim = self.options['krylov_dim']
        H = (1j * self.system(0)).data
        p0 = _data.inner(psi, psi) # purity
        sp0 = np.sqrt(p0)

        T_diag = np.zeros(krylov_dim, dtype=complex)
        T_subdiag = np.zeros(krylov_dim, dtype=complex)
        v = [psi]

        w_prime = _data.matmul(H, v[-1])
        T_diag[0] = _data.inner(w_prime, v[-1]) / p0
        w = _data.add(w_prime, v[-1], -T_diag[0])
        T_subdiag[0] = _data.norm.l2(w) / sp0
        j = 1

        while j < krylov_dim and T_subdiag[j-1] > self.options['sub_system_tol']:
            v.append(_data.mul(w, 1 / T_subdiag[j-1]))
            w_prime = _data.matmul(H, v[-1])
            T_diag[j] = _data.inner(w_prime, v[-1]) / p0
            w = _data.add(w_prime, v[-1], -T_diag[j])
            w = _data.add(w, v[-2], -T_subdiag[j-1])
            T_subdiag[j] = _data.norm.l2(w) / sp0
            j += 1

        krylov_tridiag = _data.diag["dense"](
            [T_subdiag[:j-1], T_diag[:j], T_subdiag[:j-1]],
            [-1, 0, 1]
        )
        krylov_basis = _data.Dense(np.hstack([p.to_array() for p in v]))

        return krylov_tridiag, krylov_basis

    def _lanczos_full_reorth_algorithm(self, psi):
        """
        Computes the Krylov subspace basis for a Hamiltonian 'H', a system
        state 'psi' and Krylov dimension 'krylov_dim' using the Lanczos
        algorithm and reorthogonalising the basis vectors with respect to all
        previous ones. This can drastically reduce numerical errors. The
        difference to the Arnoldi algorithm is that the upper triangular values
        are discarded since they originate from numerical errors and not
        physical properties. The result is a tridiagonal matrix.

        The space is spanned by
        {psi, H psi, H^2 psi, ..., H^(krylov_dim - 1) psi}.

        Parameters
        ------------
        psi: np.ndarray
            State used to calculation Krylov subspace (= first basis state).

        t: float, default: 0
            Time at which to evaluate the Hamiltonian.
        """
        krylov_dim = self.options['krylov_dim']
        H = (1j * self.system(0)).data
        p0 = _data.inner(psi, psi) # purity
        sp0 = np.sqrt(p0)

        h_diag = np.zeros(krylov_dim, dtype=complex)
        h_subdiag = np.zeros(krylov_dim, dtype=complex)
        Q = [psi]

        k = 1
        v = _data.matmul(H, Q[-1])
        h_diag[0] = _data.inner(Q[-1], v) / p0
        v = _data.add(v, Q[-1], -h_diag[0])
        h_subdiag[0] = _data.norm.l2(v) / sp0
        while k < krylov_dim and h_subdiag[k-1] > self.options['sub_system_tol']:
            Q.append(_data.mul(v, 1 / h_subdiag[k-1]))
            v = _data.matmul(H, Q[-1])
            k += 1
            for j in range(k):  # removes projections
                ol = _data.inner(Q[j], v) / p0
                v = _data.add(v, Q[j], -ol)
            h_diag[k-1] = ol
            h_subdiag[k-1] = _data.norm.l2(v) / sp0

        krylov_trid = _data.diag["dense"](
            [h_subdiag[:k-1], h_diag[:k], h_subdiag[:k-1]],
            [-1, 0, 1]
        )
        krylov_basis = _data.Dense(np.hstack([p.to_array() for p in Q]))

        return krylov_trid, krylov_basis

    def _arnoldi_algorithm(self, psi):
        """
        Computes the Krylov subspace basis for a Hamiltonian 'H', a system
        state 'psi' and Krylov dimension 'krylov_dim' using the Arnoldi
        interation. This results in an upper Hessenberg matrix. The space is
        spanned by
        {psi, H psi, H^2 psi, ..., H^(krylov_dim - 1) psi}.

        Parameters
        ------------
        psi: np.ndarray
            State used to calculation Krylov subspace (= first basis state).

        t: float, default: 0
            Time at which to evaluate the Hamiltonian.
        """
        krylov_dim = self.options['krylov_dim']
        H = (1j * self.system(0)).data
        p0 = _data.inner(psi, psi) # purity
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
            for j in range(k):  # removes projections
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
        evals, evecs = _data.eigs(krylov_tridiag, True)
        N = evals.shape[0]
        U = _data.matmul(krylov_basis, evecs)
        e0 = evecs.adjoint() @ _data.one_element_dense((N, 1), (0, 0), 1.0)
        return evals, U, e0

    def _compute_psi(self, dt, eigenvalues, U, e0):
        """
        compute the state at time ``t``.
        """
        phases = _data.Dense(np.exp(-1j * dt * eigenvalues))
        aux = _data.multiply(phases, e0)
        return _data.matmul(U, aux)

    def _compute_max_step(
            self,
            krylov_tridiag
    ):
        """
        Compute the maximum step length to stay under the desired tolerance.
        """
        #if not krylov_state:
            #krylov_state = \
                #self._compute_krylov_set(krylov_tridiag, krylov_basis)
        dim = krylov_tridiag.shape[0]
        if dim > 100:
            # TODO necessary?
            fac = (np.sqrt(2*np.pi*dim))**(1/dim) * (dim / np.exp(1)) # equals factorial at large kdim
        else:
            fac = factorial(dim)
        prod = np.multiply.reduce(_data.diag(krylov_tridiag, -1))
        if prod == 0:
            return self._max_step
        return fac * (self.options['atol'] / prod)**(1/dim)

    def set_state(self, t, state0):
        self._t_0 = t
        krylov_tridiag, krylov_basis = self._algorithm(state0)
        self._krylov_state = \
            self._compute_krylov_set(krylov_tridiag, krylov_basis)

        if (
            krylov_tridiag.shape[0] <= self.options['krylov_dim']
            or krylov_tridiag.shape == self.system.shape
        ):
            # happy_breakdown
            self._max_step = np.inf
            return

        if (
            not np.isfinite(self._max_step)
            or self.options["always_compute_step"]
        ):
            self._max_step = self._compute_max_step(krylov_tridiag)

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
                    f"({self.options['nsteps']}) exceeded"
                )
            new_psi = self._compute_psi(self._max_step, *self._krylov_state)
            self.set_state(self._t_0 + self._max_step, new_psi)

        delta_t = t - self._t_0
        out = self._compute_psi(delta_t, *self._krylov_state)
        return t, out

    @property
    def options(self):
        """
        Supported options by krylov method:

        atol : float, default: 1e-7
            Absolute tolerance.

        nsteps : int, default: 100
            Max. number of internal steps/call.

        min_step, max_step : float, default: (1e-5, 1e5)
            Minimum and maximum step size.

        krylov_dim: int, default: 0
            Dimension of Krylov approximation subspaces used for the time
            evolution approximation. If the defaut 0 is given, the dimension
            is calculated from the system size N, using
            `min(int((N + 100)**0.5), N-1)`.

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
