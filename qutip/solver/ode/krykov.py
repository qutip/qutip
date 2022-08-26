from ..integrator import IntegratorException, Integrator
import numpy as np
from qutip.core import data as _data
from scipy.optimize import root_scalar
from ..sesolve import SeSolver


class IntegratorKrylov(Integrator):
    """
    Evolve the state vector ("psi0") finding an approximation for the time
    evolution operator of Hamiltonian ("H") by obtaining the projection of
    the time evolution operator on a set of small dimensional Krylov
    subspaces (m << dim(H)).
    """
    integrator_options = {
        'atol': 1e-6,
        'nsteps': 100,
        'min_step': 1e-5,
        'max_step': 1e5,
        'krylov_dim': 0,
        'sub_system_tol': 1e-7,
    }
    support_time_dependant = False
    supports_blackbox = False
    method = 'krylov'

    def _prepare(self):
        self._max_step = -np.inf
        self._step = 0
        krylov_dim = self.options["krylov_dim"]
        if krylov_dim < 0 or krylov_dim > self.system.shape[0]:
            raise ValueError("The options 'krylov_dim', must be a positive "
                             "integer smaller that the system size")

        if krylov_dim == 0:
            # TODO: krylov_dim, max_step and error (atol) are related by
            # err ~= exp(-krylov_dim / dt**(1~2))
            # We could ask for 2 and determine the third one.
            N = self.system.shape[0]
            krylov_dim = min(int((N + 100)**0.5), N)
            self.options["krylov_dim"]  = krylov_dim

    def _lanczos_algorithm(self, psi):
        """
        Computes a basis of the Krylov subspace for Hamiltonian 'H', a system
        state 'psi' and Krylov dimension 'krylov_dim'. The space is spanned
        by {psi, H psi, H^2 psi, ..., H^(krylov_dim) psi}.

        Parameters
        ------------
        psi: np.ndarray
            State used to calculate Krylov subspace.

        Returns
        ---------
        v: np.ndarray
            Lanczos eigenvector.
        T: np.ndarray
            Tridiagonal decomposition.
        """
        krylov_dim = self.options['krylov_dim']
        H = (1j * self.system(0)).data

        v = []
        T_diag = np.zeros(krylov_dim + 1, dtype=complex)
        T_subdiag = np.zeros(krylov_dim, dtype=complex)

        w_prime = _data.matmul(H, psi)
        T_diag[0] = _data.inner(w_prime, psi)
        v.append(psi)
        w = _data.add(w_prime, v[-1], -T_diag[0])

        for j in range(1, krylov_dim + 1):
            T_subdiag[j-1] = _data.norm.l2(w)

            if T_subdiag[j-1] < self.options['sub_system_tol']:
                # Happy breakdown
                break

            v.append(_data.mul(w, 1 / T_subdiag[j-1]))
            w_prime = _data.matmul(H, v[-1])
            T_diag[j] = _data.inner(w_prime, v[-1])

            w = _data.add(w_prime, v[-1], -T_diag[j])
            w = _data.add(w, v[-2], -T_subdiag[j-1])

        T_m = _data.diag["dense"](
            [T_subdiag[:j], T_diag[:j+1], T_subdiag[:j]],
            [-1, 0, 1]
        )
        v = _data.Dense(np.hstack([psi.to_array() for psi in v]))

        return T_m, v

    def _compute_krylov_set(self, T_m, v):
        eigenvalues, eigenvectors = _data.eigs(T_m, True)
        N = eigenvalues.shape[0]
        U = _data.matmul(v, eigenvectors)
        e0 = eigenvectors.adjoint() @ _data.one_element_dense((N, 1), (0, 0), 1.0)
        return eigenvalues, U, e0

    def _compute_psi(self, dt, eigenvalues, U, e0):
        phases = _data.Dense(np.exp(-1j * dt * eigenvalues))
        aux = _data.multiply(phases, e0)
        return _data.matmul(U, aux)

    def _compute_max_step(self, krylov_state, reduced_state):

        def krylov_error(t):
            return np.log(np.linalg.norm(
                self._compute_psi(t, *krylov_state) -
                self._compute_psi(t, *reduced_state)
            ) / self.options["atol"])

        dt = self.options["min_step"]
        err = krylov_error(dt)
        if err > 0:
            ValueError(
                f"With the krylov dim of {self.options['krylov_dim']}, the "
                f"error with the minimum step {dt} is {err}, higher than the "
                f"desired tolerance of {self.options['atol']}."
            )

        while krylov_error(dt * 10) < 0 and dt < self.options["max_step"]:
            dt *= 10

        if dt > self.options["max_step"]:
            return self.options["max_step"]

        sol = root_scalar(f=krylov_error, bracket=[dt, dt * 10],
                          method="brentq", xtol=self.options['atol'])
        if sol.converged:
            return sol.root
        else:
            return dt

    def set_state(self, t, state0):
        self._t_0 = t
        T_m, v = self._lanczos_algorithm(state0)
        self._krylov_state = self._compute_krylov_set(T_m, v)

        if T_m.shape[0] <= self.options['krylov_dim']:
            # happy_breakdown
            self._max_step = np.inf
            return

        if not np.isfinite(self._max_step):
            reduced_state = self._compute_krylov_set(T_m[:-1, :-1], v[:-1, :])
            self._max_step = self._compute_max_step(self._krylov_state, reduced_state)

    def get_state(self, copy=True):
        return self._t_0, self._compute_psi(0, *self._krylov_state)

    def integrate(self, t, copy=True):
        while t > self._t_0 + self._max_step:
            # The approximation in only valid in the range t_0, t_0 + max step
            # If outside, advance the range
            self._step += 1
            if self._step >= self.options["nsteps"]:
                self._step = 0
                raise IntegratorException
            self.set_state(*self.integrate(self._t_0 + self._max_step))

        self._step = 0
        delta_t = t - self._t_0
        return t, self._compute_psi(delta_t, *self._krylov_state)


SeSolver.add_integrator(IntegratorKrylov, 'krylov')
