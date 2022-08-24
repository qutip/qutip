from ..integrator import IntegratorException, Integrator
import numpy as np


class IntegratorKrylov(Integrator):
    """
    Evolve the state vector ("psi0") finding an approximation for the time
    evolution operator of Hamiltonian ("H") by obtaining the projection of
    the time evolution operator on a set of small dimensional Krylov
    subspaces (m << dim(H)).
    """
    integrator_options = {
        'tol': 1e-6,
        'nsteps': 100,
        'min_step': 1e-5,
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
            # TODO: The error is proportional to exp(-krylov_dim)
            # This could be obtianed from atol
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
        H = self.system.full()

        v = np.zeros((krylov_dim + 1, psi.shape[0]), dtype=complex)
        T_m = np.zeros((krylov_dim + 1, krylov_dim + 1), dtype=complex)

        v[0, :] = psi.to_array().squeeze()

        w_prime = H.dot(v[0, :])

        alpha = np.vdot(w_prime, v[0, :])

        w = w_prime - alpha * v[0, :]

        T_m[0, 0] = alpha

        for j in range(1, krylov_dim + 1):

            beta = np.linalg.norm(w)

            if beta < self.options['sub_system_tol']:
                # Happy breakdown
                v = v[0:j, :]
                T_m = T_m[0:j, 0:j]
                break

            v[j, :] = w / beta
            w_prime = H.dot(v[j, :])
            alpha = np.vdot(w_prime, v[j, :])

            w = w_prime - alpha * v[j, :] - beta * v[j - 1, :]

            T_m[j, j] = alpha
            T_m[j, j - 1] = beta
            T_m[j - 1, j] = beta

        return T_m, v

    def _compute_evolution_matrices(self, T_m, v):
        eigenvalues, eigenvectors = _data.eigs(_data.Dense(T_m), True)
        eigenvectors = eigenvectors.to_array()
        U = np.matmul(v.T, eigenvectors)
        e0 = eigenvectors.conj().T[:, 0]
        return eigenvalues, U, e0

    def _compute_psi(self, dt, eigenvalues, U, e0):
        aux = np.multiply(np.exp(-1j * delta_t * self._eigenvalues), self._e0)
        return np.matmul(self._U, aux)

    def set_state(self, t, state0):
        self._t_0 = t
        T_m, v = self._lanczos_algorithm(state0)
        self._krylov_state = self._compute_evolution_matrices(T_m, v)

        if T_m.shape[0] <= self.options['krylov_dim']:
            # happy_breakdown
            self._max_step = np.inf
            return

        if not np.isfinite(self._max_step):
            reduced_state = self._compute_evolution_matrices(T_m[:-1, :-1], v[:-1, :])

            def get_error(t):
                return np.log(np.linalg.norm(
                    self._compute_psi(t, *self._krylov_state) -
                    self._compute_psi(t, *reduced_state)
                ) / self.options["atol"])

            dt = min_step
            err = get_error(dt)
            if err > 0:
                ValueError( "Place holder ###################################################"
                    "No solution exists with the given combination of parameters 'krylov_dim', "
                    "tolerance = 'options['atol']', maximum number allowed of krylov internal "
                    "partitions = 'options['nsteps']' and 'tlist'. Try reducing the tolerance, or "
                    "increasing 'krylov_dim'. If nothing works, then a deeper analysis of the "
                    "problem is recommended."
                    "#####################################################################################################"
                )

            while err < 0:
                dt *= 2
                err = get_error(dt)

            sol = root_scalar(f=get_error, bracket=[dt/2,dt], method="brentq", xtol=self.options['atol'])
            if sol.converged:
                delta_t = sol.root
                return delta_t
            else:
                raise Exception(
                    "Method did not converge, try increasing 'krylov_dim', "
                    "taking a lesser final time 'tlist[-1]' or decreasing the "
                    "tolerance via SolverOptions().atol. "
                    "If nothing works, this problem might not be suitable for "
                    "Krylov or a deeper analysis might be required."
                )









    def get_state(self, copy=True):
        return self._t_0, _data.Dense(np.matmul(self._U, self._e0))

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
        return t, _data.Dense(self._compute_psi(delta_t, *self._krylov_state))
