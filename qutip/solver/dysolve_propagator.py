from qutip import Qobj, qeye_like
from numpy.typing import ArrayLike
import numpy as np
import scipy as sp
from numbers import Number
import itertools
from scipy.special import factorial


__all__ = ['DysolvePropagator', 'dysolve_propagator']


FACTORIAL_LOOKUP = {
    0: 1, 1: 1, 2: 2, 3: 6, 4: 24, 5: 120, 6: 720, 7: 5040, 8: 40320,
    9: 362880, 10: 3628800, 11: 39916800, 12: 479001600, 13: 6227020800,
    14: 87178291200, 15: 1307674368000
}


class DysolvePropagator:
    """
    A generator of propagator using Dysolve.
    https://arxiv.org/abs/2012.09282

    Parameters
    ----------
    H_0 : Qobj
        The base hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    options : dict, optional
        Extra parameters.

        - "max_order"

            A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.

        - "a_tol"

            The absolute tolerance used when computing the propagators
            (default is 1e-10).

        - "max_dt"

            The maximum time increment used when computing propagators
            (default is 0.1).

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    For the moment, only a cosine perturbation is allowed. Dysolve can
    manage more exotic perturbations, but this is not implemented yet.

    .. note:: Experimental.

    """

    def __init__(
        self,
        H_0: Qobj,
        X: Qobj,
        omega: float,
        options: dict[str] = None,
    ):
        # System
        self._eigenenergies, self._basis = H_0.eigenstates()
        self._H_0 = H_0
        self._X = X
        self._omega = omega

        # Options
        if options is None:
            self.max_order = 4
            self.a_tol = 1e-10
            self.max_dt = 0.1
        else:
            self.max_order = options.get('max_order', 4)
            self.max_dt = options.get('max_dt', 0.1)
            self.a_tol = options.get('a_tol', 1e-10)

        self._dt_Sns = {}
        self.U = None

    def __call__(self, t_f: float, t_i: float = 0.0) -> Qobj:
        """
        Computes the propagator from t_i to t_f. If t_i is not provided,
        computes the propagator from 0 to t_f.

        Parameters
        ----------
        t_f : float
            Final time of the evolution.

        t_i : float, default = 0.0
            Initial time of the evolution.

        Returns
        -------
        U : Qobj
            The propagator U(t_f, t_i) from t_i to t_f.

        Notes
        -----
        If t_f - t_i > max_dt, splits the evolution into smaller ones
        to then reconstruct U(t_f, t_i).

        Memoization is used. First call may be slow but the next calls
        should be faster.

        """
        time_diff = t_f - t_i
        dt = self.max_dt * np.sign(time_diff)
        n_steps = abs(int(time_diff / self.max_dt))

        U = np.eye(len(self._eigenenergies), dtype=np.complex128)

        for j in range(n_steps):
            U_step = np.zeros_like(U)

            Uns = self._compute_Uns(t_i + j*dt, dt)
            for n in range(self.max_order + 1):
                U_step += Uns[n]

            U = U_step @ U

        if abs(time_diff - n_steps*dt) > self.a_tol:
            dt = time_diff - n_steps*dt

            U_extra = np.zeros_like(U)
            Uns = self._compute_Uns(t_f - dt, dt)
            for n in range(self.max_order + 1):
                U_extra += Uns[n]
            U = U_extra @ U

        self.U = Qobj(U, self._H_0.dims).transform(self._basis, True)

        return self.U

    def _compute_integrals(self, ws: ArrayLike, dt: float) -> Number:
        """
        Computes the value of the nested integrals for a given array of
        effective omegas. See eq. (7) in Ref.

        Parameters
        ----------
        ws : ArrayLike
            An array of effective omegas. ws[0] is the omega for the rightmost
            integral.

        dt : float
            The time increment.

        Returns
        -------
        value : Number
            The value of the nested integrals.

        Notes
        -----
        Integrals are done analytically from right to left with integration
        by parts.

        """
        if len(ws) == 0:
            return 1
        elif len(ws) == 1:
            if np.abs(ws[0]) < self.a_tol:
                return dt
            else:
                return (-1j / ws[0]) * (np.exp(1j * ws[0] * dt) - 1)
        else:
            if np.abs(ws[0]) < self.a_tol:
                return self._compute_tn_integrals(ws[1:], 1, dt)
            else:
                ws_prime = ws[1:].copy()
                ws_prime[0] += ws[0]
                return (-1j / ws[0]) * (
                    self._compute_integrals(
                        ws_prime, dt) - self._compute_integrals(ws[1:], dt)
                )

    def _compute_tn_integrals(self, ws: ArrayLike, n: int, dt: float) -> Number:
        """
        Helper function to compute nested integrals when the function to
        integrate is t^n/factorial(n) * exp(1j*omega*t). This happens when
        some effective omegas are 0. In that case, the recursion differs a
        bit from _compute_integrals(). See eq. (7) in Ref.

        Paramaters
        ----------
        ws : ArrayLike
            An array of effective omegas. ws[0] is the omega for the rightmost
            integral.

        n : int
            The variable in t^n/factorial(n).

        dt : float
            The time increment.

        Returns
        -------
        value : Number
            The value of the nested integrals when the function to integrate is
            t^n/factorial(n) * exp(1j*omega*t).

        Notes
        -----
        Integrals are done analytically from right to left with integration
        by parts.

        """
        if n == 0:
            return self._compute_integrals(ws, dt)

        if len(ws) == 1:
            if np.abs(ws[0]) < self.a_tol:
                return (dt ** (n + 1)) / FACTORIAL_LOOKUP[n + 1]
            else:
                factor = (-1j/ws[0]) * np.exp(1j*ws[0]*dt)
                term1 = 0
                for j in range(n+1):
                    term1 += ((1j/ws[0])**j) * \
                        (dt**(n-j) / FACTORIAL_LOOKUP[n-j])
                term2 = (1j / ws[0])**(n+1)
                return factor*term1 + term2

                # Recursive version of this case:
                # factor = -1j / ws[0]
                # term1 = (self.dt**n / factorial(n)) * \
                #     np.exp(1j * ws[0] * self.dt)
                # term2 = self._compute_tn_integrals(ws, n - 1)
                # return factor * (term1 - term2)
        else:
            if np.abs(ws[0]) < self.a_tol:
                return self._compute_tn_integrals(ws[1:], n + 1, dt)
            else:
                factor = -1j / ws[0]
                ws_prime = ws[1:].copy()
                ws_prime[0] += ws[0]
                term1 = self._compute_tn_integrals(ws_prime, n, dt)
                term2 = self._compute_tn_integrals(ws, n - 1, dt)
                return factor * (term1 - term2)

    def _update_matrix_elements(self, current: ArrayLike) -> ArrayLike:
        """
        Reuses the current matrix elements (order n-1) to compute the
        matrix elements for the order n.

        Parameters
        ----------
        current : ArrayLike
            The current matrix elements (for the order n-1).

        Returns
        -------
        matrix_elements : ArrayLike
            The new matrix elements for the order n.
        """
        elems = self._X.transform(self._basis).full().flatten()
        if current is None:
            return elems
        else:
            shape = self._X.shape[0]
            a = np.tile(current, shape)
            b = np.repeat(elems, len(current)//shape)
            return a * b

        # elems = sp.sparse.csr_array(
        #     self.X.transform(self._basis).full().flatten()
        # )
        # if current is None:
        #     return elems
        # else:
        #     x_shape = self.X.shape[0]
        #     a = sp.sparse.hstack([current]*x_shape)
        #     b = sp.sparse.vstack(
        #         [elems]*(current.shape[0]//x_shape)
        #     ).transpose().reshape(a.shape)
        #     return a.multiply(b).reshape((a.shape[1],))

    def _compute_Sns(self, dt: float) -> dict:
        """
        Computes Sns for each omega vector. This implements a similar equation
        to eq. (14) in Ref, but the function "f" is not used to avoid dealing
        explicitly with limits.

        Parameters
        ----------
        dt : float
            The time increment.

        Returns
        -------
        Sns : dict
            Sns for each omega vector. key = order with the result for each
            omega vector.

        """
        if dt in self._dt_Sns:
            return self._dt_Sns[dt]

        else:
            Sns = {}
            length = len(self._eigenenergies)
            exp_H_0 = (-1j*dt*self._H_0.transform(self._basis)
                       ).expm().full()
            current_matrix_elements = None

            Sns[0] = exp_H_0

            for n in range(1, self.max_order + 1):
                omega_vectors = np.fromiter(
                    itertools.product([self._omega, -self._omega], repeat=n),
                    np.dtype((float, (n,)))
                )

                lambdas = np.fromiter(
                    itertools.product(self._eigenenergies, repeat=n + 1),
                    np.dtype((float, (n+1,)))
                )
                diff_lambdas = -np.diff(lambdas)[:, ::-1]

                ket_bra_idx = np.vstack(
                    (np.repeat(np.arange(0, length), length**n),
                     np.tile(np.arange(0, length), length**n))
                ).T

                Sn = np.zeros((len(omega_vectors), length, length),
                              dtype=np.complex128
                              )

                # Compute matrix elements
                current_matrix_elements = self._update_matrix_elements(
                    current_matrix_elements
                )

                for i, omega_vector in enumerate(omega_vectors):
                    # Compute integrals
                    ls_ws = omega_vector + diff_lambdas
                    integrals = np.zeros(ls_ws.shape[0], dtype=np.complex128)
                    for j, ws in enumerate(ls_ws):
                        integrals[j] = self._compute_integrals(ws, dt)

                    x = integrals * current_matrix_elements

                    for row in range(ket_bra_idx.shape[0]):
                        Sn[i, ket_bra_idx[row, 0],
                            ket_bra_idx[row, 1]] += x[row]

                    Sn[i] *= (-1j / 2) ** n
                    Sn[i] = exp_H_0 @ Sn[i]

                Sns[n] = Sn

            self._dt_Sns[dt] = Sns
            return Sns

    def _compute_Uns(self, current_time: float, dt: float) -> dict:
        """
        Computes Un for each order n from time current_time to
        current_time + dt. See eq. (5) in Ref.

        Parameters
        ----------
        current_time : float
            The current time where to start the evolution for
            a time dt. current_time can be positive or negative.

        dt : float
            The time increment.

        Returns
        -------
        Uns : dict
            Un for each order from time current_time to
            current_time + dt. Key = order
        """
        Uns = {}
        Sns = self._compute_Sns(dt)
        Uns[0] = Sns[0]

        for n in range(1, self.max_order + 1):
            omega_vectors = np.fromiter(
                itertools.product([self._omega, -self._omega], repeat=n),
                np.dtype((float, (n,)))
            )

            U_n = np.zeros(
                (len(self._eigenenergies), len(self._eigenenergies)),
                dtype=np.complex128
            )

            for i, omega_vector in enumerate(omega_vectors):
                U_n += np.exp(1j * np.sum(omega_vector)
                              * current_time) * Sns[n][i]
            Uns[n] = U_n
        return Uns


def dysolve_propagator(
        H_0: Qobj,
        X: Qobj,
        omega: float,
        t: float | list[float],
        options: dict[str] = None
) -> Qobj | list[Qobj]:
    """
    A generator of propagator(s) using Dysolve.
    https://arxiv.org/abs/2012.09282.

    Parameters
    ----------
    H_0 : Qobj
        The hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    t : float | list[float]
        Time or list of times for which to evaluate the propagator(s). If t
        is a single number, the propagator from 0 to t is computed. When
        t is a list, the propagators from the first time to each elements in
        t is returned. In that case, the first output will always be the
        identity matrix. Also, in that case, have the same time increment in
        between elements for better performance.

    options : dict, optional
        Extra parameters.

        - "max_order"

            A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.

        - "a_tol"

            The absolute tolerance used when computing the propagators
            (default is 1e-10).

        - "max_dt"

            The maximum time increment used when computing propagators
            (default is 0.1).

    Returns
    -------
    Us : Qobj | list[Qobj]
        The time evolution propagator U(t,0) if t is a single number or else
        a list of propagators [U(t[i], t[0])] for all elements t[i] in t.

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    For the moment, only a cosine perturbation is allowed. Dysolve can
    manage more exotic perturbations, but this is not implemented yet.

    .. note:: Experimental.

    """
    if isinstance(t, Number):
        dysolve = DysolvePropagator(H_0, X, omega, options)
        return dysolve(t)

    else:
        Us = []
        Us.append(qeye_like(H_0))  # U(t_0, t_0) = identity

        dysolve = DysolvePropagator(H_0, X, omega, options)
        for i in range(len(t[:-1])):  # Compute individual U(t[i+1], t[i])
            U = dysolve(t[i+1], t[i])
            Us.append(U)

        for i in range(1, len(Us)):  # [U(t[i], t[0])]
            Us[i] = Us[i] @ Us[i - 1]

    return Us
