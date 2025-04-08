from qutip import Qobj, qeye_like
from numpy.typing import ArrayLike
import numpy as np
import scipy as sp
from numbers import Number
import itertools
from scipy.special import factorial

__all__ = ['DysolvePropagator', 'dysolve_propagator']


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

    omega : Number
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

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    For the moment, only a cosine perturbation is allowed. Dysolve can
    manage more exotic perturbations, but this is not implemented yet.

    Experimental.

    """

    def __init__(
        self,
        H_0: Qobj,
        X: Qobj,
        omega: Number,
        options: dict[str] = None,
    ):
        # System
        self._eigenenergies, self._basis = H_0.eigenstates()
        self.H_0 = H_0
        self.X = X
        self.omega = omega

        # Times
        self.t_i = None
        self.t_f = None
        self._dt = None
        self._max_dt = 0.1

        # Options
        if options is None:
            self.max_order = 4
            self.a_tol = 1e-10
        else:
            self.max_order = options.get('max_order', 4)
            self.a_tol = options.get('a_tol', 1e-10)

        self._Sns = None
        self._dt_Sns = {}
        self.U = None

    def __call__(self, t_f: Number, t_i: Number = 0) -> Qobj:
        """
        Computes the propagator from t_i to t_f. If t_i is not provided,
        computes the propagator from 0 to t_f.

        Parameters
        ----------
        t_f : Number
            Final time of the evolution.

        t_i : Number, default = 0
            Initial time of the evolution.

        Returns
        -------
        U : Qobj
            The propagator U(t_f, t_i) from t_i to t_f.

        """
        self.t_i = t_i
        self.t_f = t_f
        time_diff = t_f - t_i
        n_steps = abs(int(time_diff / self._max_dt))

        U = np.eye(len(self._eigenenergies), dtype=np.complex128)

        self._dt = self._max_dt * np.sign(time_diff)
        if self._dt not in self._dt_Sns:
            self._Sns = self._compute_Sns()
            self._dt_Sns[self._dt] = self._Sns
        else:
            self._Sns = self._dt_Sns[self._dt]

        for j in range(n_steps):
            U_step = np.zeros_like(U)

            Uns = self._compute_Uns(t_i + j*self._dt)
            for n in range(self.max_order + 1):
                U_step += Uns[n]

            U = U_step @ U

        if time_diff - n_steps*self._dt != 0:
            self._dt = time_diff - n_steps*self._dt
            if self._dt not in self._dt_Sns:
                self._Sns = self._compute_Sns()
                self._dt_Sns[self._dt] = self._Sns
            else:
                self._Sns = self._dt_Sns[self._dt]

            U_extra = np.zeros_like(U)
            Uns = self._compute_Uns(t_f - self._dt)
            for n in range(self.max_order + 1):
                U_extra += Uns[n]
            U = U_extra @ U

        self.U = Qobj(U, self.H_0.dims).transform(self._basis, True)

        return self.U

    def _compute_integrals(self, ws: ArrayLike) -> Number:
        """
        Computes the value of the nested integrals for a given array of
        effective omegas. See eq. (7) in Ref.

        Parameters
        ----------
        ws : ArrayLike
            An array of effective omegas. ws[0] is the omega for the rightmost
            integral.

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
                return self._dt
            else:
                return (-1j / ws[0]) * (np.exp(1j * ws[0] * self._dt) - 1)
        else:
            if np.abs(ws[0]) < self.a_tol:
                return self._compute_tn_integrals(ws[1:], 1)
            else:
                ws_prime = ws[1:].copy()
                ws_prime[0] += ws[0]
                return (-1j / ws[0]) * (
                    self._compute_integrals(
                        ws_prime) - self._compute_integrals(ws[1:])
                )

    def _compute_tn_integrals(self, ws: ArrayLike, n: int) -> Number:
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
            return self._compute_integrals(ws)

        if len(ws) == 1:
            if np.abs(ws[0]) < self.a_tol:
                return (self._dt ** (n + 1)) / factorial(n + 1)
            else:
                factor = (-1j/ws[0]) * np.exp(1j*ws[0]*self._dt)
                term1 = 0
                for j in range(n+1):
                    term1 += ((1j/ws[0])**j) * \
                        (self._dt**(n-j) / factorial(n-j))
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
                return self._compute_tn_integrals(ws[1:], n + 1)
            else:
                factor = -1j / ws[0]
                ws_prime = ws[1:].copy()
                ws_prime[0] += ws[0]
                term1 = self._compute_tn_integrals(ws_prime, n)
                term2 = self._compute_tn_integrals(ws, n - 1)
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
        elems = self.X.transform(self._basis).full().flatten()
        if current is None:
            return elems
        else:
            shape = self.X.shape[0]
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

    def _compute_Sns(self) -> dict:
        """
        Computes Sns for each omega vector. This implements a similar equation
        to eq. (14) in Ref, but the function "f" is not used to avoid dealing
        explicitly with limits.

        Returns
        -------
        Sns : dict
            Sns for each omega vector. key = order with the result for each
            omega vector.

        """
        Sns = {}
        length = len(self._eigenenergies)
        exp_H_0 = (-1j*self._dt*self.H_0.transform(self._basis)).expm().full()
        current_matrix_elements = None

        Sns[0] = exp_H_0

        for n in range(1, self.max_order + 1):
            omega_vectors = np.fromiter(
                itertools.product([self.omega, -self.omega], repeat=n),
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
                    integrals[j] = self._compute_integrals(ws)

                x = integrals * current_matrix_elements

                for row in range(ket_bra_idx.shape[0]):
                    Sn[i, ket_bra_idx[row, 0],
                        ket_bra_idx[row, 1]] += x[row]

                Sn[i] *= (-1j / 2) ** n
                Sn[i] = exp_H_0 @ Sn[i]

            Sns[n] = Sn

        return Sns

    def _compute_Uns(self, current_time: Number) -> dict:
        """
        Computes Un for each order n from time current_time to
        current_time + dt. See eq. (5) in Ref.

        Parameters
        ----------
        current_time : Number
            The current time where to start the evolution for
            a time dt. current_time can be positive or negative.

        Returns
        -------
        Uns : dict
            Un for each order from time current_time to
            current_time + dt. Key = order
        """
        Uns = {}
        Sns = self._Sns
        Uns[0] = Sns[0]

        for n in range(1, self.max_order + 1):
            omega_vectors = np.fromiter(
                itertools.product([self.omega, -self.omega], repeat=n),
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
        omega: Number,
        t: Number | list[Number],
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

    omega : Number
        The frequency of the cosine perturbation.

    t : Number | list[Number]
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

    Experimental.

    """
    if isinstance(t, Number):
        dysolve = DysolvePropagator(H_0, X, omega, options)
        U = dysolve(t)
        return U

    else:
        Us = []
        dts = np.diff(t)
        dt_Sns = {}  # memoize Sns of a given dt

        Us.append(qeye_like(H_0))  # U(t_0, t_0) = identity

        dysolve = DysolvePropagator(H_0, X, omega, options)
        for i in range(len(t[:-1])):  # Compute individual U(t[i+1], t[i])
            if dt_Sns.get(dts[i]) is None:
                U = dysolve(t[i+1], t[i])
                Us.append(U)
                dt_Sns[dts[i]] = dysolve._Sns
            else:
                dysolve.t_i = t[i]
                dysolve.t_f = t[i+1]
                dysolve.dt = dts[i]
                dysolve._Sns = dt_Sns[dts[i]]

                U = np.zeros(
                    (len(dysolve._eigenenergies), len(dysolve._eigenenergies)),
                    dtype=np.complex128
                )

                Uns = dysolve._compute_Uns(t[i])
                for n in range(dysolve.max_order + 1):
                    U += Uns[n]

                dysolve.U = Qobj(U, H_0.dims)
                Us.append(dysolve.U)

        for i in range(1, len(Us)):  # [U(t[i], t[0])]
            Us[i] = Us[i] @ Us[i - 1]

    return Us
