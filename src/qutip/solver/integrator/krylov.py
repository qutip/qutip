from ..integrator import IntegratorException, Integrator
import numpy as np
from qutip.core import data as _data
from scipy.optimize import root_scalar
from ..sesolve import SESolver


__all__ = ["IntegratorKrylov"]


class IntegratorKrylov(Integrator):
    """
    Evolve the state vector ("psi0") finding an approximation for the time
    evolution operator of Hamiltonian ("H") by obtaining the projection of
    the time evolution operator on a set of small dimensional Krylov
    subspaces (m << dim(H)).
    """
    integrator_options = {
        'atol': 1e-7,
        'nsteps': 100,
        'min_step': 1e-5,
        'max_step': 1e5,
        'krylov_dim': 0,
        'sub_system_tol': 1e-7,
        'always_compute_step': False,
    }
    support_time_dependant = False
    supports_blackbox = False
    method = 'krylov'

    def _prepare(self):
        if not self.system.isconstant:
            raise ValueError("krylov method only support constant system.")
        self._max_step = -np.inf
        krylov_dim = self.options["krylov_dim"]
        if krylov_dim < 0 or krylov_dim > self.system.shape[0]:
            raise ValueError("The options 'krylov_dim', must be a positive "
                             "integer smaller that the system size")

        if krylov_dim == 0:
            # TODO: krylov_dim, max_step and error (atol) are related by
            # err ~= exp(-krylov_dim / dt**(1~2))
            # We could ask for 2 and determine the third one.
            N = self.system.shape[0]
            krylov_dim = min(int((N + 100)**0.5), N-1)
            self.options["krylov_dim"]  = krylov_dim

        if not self.options["always_compute_step"]:
            from qutip import rand_ket
            N = self.system.shape[0]
            krylov_tridiag, krylov_basis = \
                self._lanczos_algorithm(rand_ket(N).data)
            if (
                krylov_tridiag.shape[0] < krylov_dim
                or krylov_tridiag.shape[0] == N
            ):
                self._max_step = np.inf
            else:
                self._max_step = self._compute_max_step(krylov_tridiag,
                                                        krylov_basis)

    def _lanczos_algorithm(self, psi):
        """
        Computes a basis of the Krylov subspace for Hamiltonian 'H', a system
        state 'psi' and Krylov dimension 'krylov_dim'. The space is spanned
        by {psi, H psi, H^2 psi, ..., H^(krylov_dim) psi}.

        Parameters
        ------------
        psi: np.ndarray
            State used to calculate Krylov subspace.
        """
        krylov_dim = self.options['krylov_dim']
        H = (1j * self.system(0)).data

        v = []
        T_diag = np.zeros(krylov_dim + 1, dtype=complex)
        T_subdiag = np.zeros(krylov_dim + 1, dtype=complex)

        w_prime = _data.matmul(H, psi)
        T_diag[0] = _data.inner(w_prime, psi)
        v.append(psi)
        w = _data.add(w_prime, v[-1], -T_diag[0])
        T_subdiag[0] = _data.norm.l2(w)
        j = 0

        while j < krylov_dim and T_subdiag[j] > self.options['sub_system_tol']:
            j += 1
            v.append(_data.mul(w, 1 / T_subdiag[j-1]))
            w_prime = _data.matmul(H, v[-1])
            T_diag[j] = _data.inner(w_prime, v[-1])
            w = _data.add(w_prime, v[-1], -T_diag[j])
            w = _data.add(w, v[-2], -T_subdiag[j-1])
            T_subdiag[j] = _data.norm.l2(w)

        krylov_tridiag = _data.diag["dense"](
            [T_subdiag[:j], T_diag[:j+1], T_subdiag[:j]],
            [-1, 0, 1]
        )
        krylov_basis = _data.Dense(np.hstack([psi.to_array() for psi in v]))

        return krylov_tridiag, krylov_basis

    def _compute_krylov_set(self, krylov_tridiag, krylov_basis):
        """
        Compute the eigen energies, basis transformation operator (U) and e0.
        """
        eigenvalues, eigenvectors = _data.eigs(krylov_tridiag, True)
        N = eigenvalues.shape[0]
        U = _data.matmul(krylov_basis, eigenvectors)
        e0 = eigenvectors.adjoint() @ _data.one_element_dense((N, 1), (0, 0), 1.0)
        return eigenvalues, U, e0

    def _compute_psi(self, dt, eigenvalues, U, e0):
        """
        compute the state at time ``t``.
        """
        phases = _data.Dense(np.exp(-1j * dt * eigenvalues))
        aux = _data.multiply(phases, e0)
        return _data.matmul(U, aux)

    def _compute_max_step(self, krylov_tridiag, krylov_basis, krylov_state=None):
        """
        Compute the maximum step length to stay under the desired tolerance.
        """
        if not krylov_state:
            krylov_state = self._compute_krylov_set(krylov_tridiag, krylov_basis)

        small_tridiag = _data.Dense(krylov_tridiag.as_ndarray()[:-1, :-1])
        small_basis = _data.Dense(krylov_basis.as_ndarray()[:, :-1])
        reduced_state = self._compute_krylov_set(small_tridiag, small_basis)

        def krylov_error(t):
            # we divide by atol and take the log so that the error returned is 0
            # at atol, which is convenient for calling root_scalar with.
            return np.log(_data.norm.l2(
                self._compute_psi(t, *krylov_state) -
                self._compute_psi(t, *reduced_state)
            ) / self.options["atol"])

        # Under 0 will cause an infinite loop in the while loop bellow.
        dt = max(self.options["min_step"], 1e-14)
        max_step = max(self.options["max_step"], dt)
        err = krylov_error(dt)
        if err > 0:
            raise ValueError(
                f"With the krylov dim of {self.options['krylov_dim']}, the "
                f"error with the minimum step {dt} is {err}, higher than the "
                f"desired tolerance of {self.options['atol']}."
            )

        while krylov_error(dt * 10) < 0 and dt < max_step:
            dt *= 10

        if dt > max_step:
            return max_step

        sol = root_scalar(f=krylov_error, bracket=[dt, dt * 10],
                          method="brentq", xtol=self.options['atol'])
        if sol.converged:
            return sol.root
        else:
            return dt

    def set_state(self, t, state0):
        self._t_0 = t
        krylov_tridiag, krylov_basis = self._lanczos_algorithm(state0)
        self._krylov_state = self._compute_krylov_set(krylov_tridiag, krylov_basis)

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
            self._max_step = self._compute_max_step(
                krylov_tridiag, krylov_basis, self._krylov_state,
            )

    def get_state(self, copy=True):
        return self._t_0, self._compute_psi(0, *self._krylov_state)

    def integrate(self, t, copy=True):
        step = 0
        while t > self._t_0 + self._max_step:
            # The approximation in only valid in the range t_0, t_0 + max step
            # If outside, advance the range
            step += 1
            if step >= self.options["nsteps"]:
                raise IntegratorException(f"Maximum number of integration steps ({self.options['nsteps']}) exceeded")
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
            evolution approximation. If the defaut 0 is given, the dimension is calculated
            from the system size N, using `min(int((N + 100)**0.5), N-1)`.

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
