from qutip import Qobj, qeye_like
from numpy.typing import ArrayLike
import numpy as np
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

    omega : float
        The frequency of the cosine perturbation.

    options : dict, optional
        Extra parameters.

        - "max_order"

            A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.

        - "a_tol"

            The absolute tolerance used whencomputing the propagators
            (default is 1e-10).

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    """

    def __init__(
        self,
        H_0: Qobj,
        X: Qobj,
        omega: float,
        options: dict[str] = None,
    ):
        # System
        self.eigenenergies, self.basis = H_0.eigenstates()
        self.H_0 = H_0.transform(self.basis)
        self.X = X.transform(self.basis)
        self.omega = omega

        # Times
        self.t_i = None
        self.t_f = None
        self.dt = None

        # Options
        if options is None:
            self.max_order = 4
            self.a_tol = 1e-10
        else:
            self.max_order = options.get('max_order', 4)
            self.a_tol = options.get('a_tol', 1e-10)

        self._Sns = None
        self.U = None

    def __call__(self, t_f: float, t_i: float = 0) -> Qobj:
        """
        Computes the propagator from t_i to t_f. If t_i is not provided,
        computes the propagator from 0 to t_f.

        Parameters
        ----------
        t_f : float
            Final time of the evolution.

        t_i : float, default = 0
            Initial time of the evolution.

        Returns
        -------
        U : Qobj
            The propagator U(t_f, t_i) from t_i to t_f.

        """
        self.t_i = t_i
        self.t_f = t_f
        self.dt = t_f - t_i

        self._Sns = self._compute_Sns()

        U = np.zeros(
            (len(self.eigenenergies), len(self.eigenenergies)),
            dtype=np.complex128
        )

        Uns = self._compute_Uns(t_i)
        for n in range(self.max_order + 1):
            U += Uns[n]

        self.U = Qobj(U, self.H_0.dims)

        return self.U

    def _compute_integrals(self, ws: ArrayLike) -> float:
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
        value : float
            The value of the nested integrals.

        Notes
        -----
        Integrals are done analytically from right to left.

        """
        if len(ws) == 0:
            return 1
        elif len(ws) == 1:
            if np.abs(ws[0]) < self.a_tol:
                return self.dt
            else:
                return (-1j / ws[0]) * (np.exp(1j * ws[0] * self.dt) - 1)
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

    def _compute_tn_integrals(self, ws: ArrayLike, n: int) -> float:
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
            An increment used in t^n/factorial(n).

        Returns
        -------
        value : float
            The value of the nested integrals when the function to integrate is
            t^n/factorial(n) * exp(1j*omega*t).

        Notes
        -----
        Integrals are done analytically from right to left.

        """
        if n == 0:
            return self._compute_integrals(ws)

        if len(ws) == 1:
            if np.abs(ws[0]) < self.a_tol:
                return (self.dt ** (n + 1)) / factorial(n + 1)
            else:
                factor = (-1j/ws[0]) * np.exp(1j*ws[0]*self.dt)
                term1 = 0
                for j in range(n+1):
                    term1 += ((1j/ws[0])**j) * \
                        (self.dt**(n-j) / factorial(n-j))
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

    def _update_matrix_elements(self, current: ArrayLike, n: int,
                                indices: ArrayLike) -> ArrayLike:
        """
        Reuses the current matrix elements to compute the matrix elements
        for the next order.

        Parameters
        ----------
        current : ArrayLike
            The current matrix elements (for the order n-1).

        n : int
            The current order.

        indices : ArrayLike
            The indices for the eigenenergies/eigenstates. Matchs the order
            of combination used to compute the effective omegas.

        Returns
        -------
        matrix_elements : ArrayLike
            The new matrix elements for the order n.
        """
        if n == 0:
            return np.ones((indices.shape[0], 1), dtype=np.complex128)
        elif n == 1:
            return self.X.full().reshape((indices.shape[0], 1))
        else:
            a = np.tile(current, self.X.shape[0]).reshape(
                (indices.shape[0], 1))
            b = current.repeat(self.X.shape[0]).reshape((indices.shape[0], 1))
            return a * b

    def _compute_Sns(self) -> dict:
        """
        Computes Sns for each omega vector.

        Returns
        -------
        Sns : dict
            Sns for each omega vector (key = order).

        """
        Sns = {}
        length = len(self.eigenenergies)
        exp_H_0 = (-1j*self.dt*self.H_0).expm()
        current_matrix_elements = None

        for n in range(self.max_order + 1):
            if n == 0:
                Sns[0] = exp_H_0
            else:
                omega_vectors = np.array(
                    list(
                        itertools.product([self.omega, -self.omega], repeat=n)
                    )
                )
                lambdas = np.array(
                    list(
                        itertools.product(self.eigenenergies, repeat=n + 1)
                    )
                )
                diff_lambdas = np.diff(lambdas)
                indices = np.array(
                    list(
                        itertools.product(range(length), repeat=n + 1)
                    )
                )
                Sn = np.zeros((len(omega_vectors), length, length),
                              dtype=np.complex128
                              )

                # Compute matrix elements
                current_matrix_elements = self._update_matrix_elements(
                    current_matrix_elements, n, indices
                )

                for i, omega_vector in enumerate(omega_vectors):
                    # Compute integrals
                    ls_ws = omega_vector + diff_lambdas
                    integrals = np.zeros(
                        (ls_ws.shape[0], ls_ws.shape[1]), dtype=np.complex128)
                    for j, ws in enumerate(ls_ws):
                        integrals[j] = self._compute_integrals(ws)

                    x = integrals * current_matrix_elements
                    ket_bra_indices = indices[:, [0, -1]]

                    k = 0
                    for idx in ket_bra_indices:
                        Sn[i, idx[1], idx[0]] += x[k][0]
                        k += 1

                    Sn[i] *= (-1j / 2) ** n
                    Sn[i] = exp_H_0.full() @ Sn[i]

                Sns[n] = Sn

        return Sns

    def _compute_Uns(self, current_time: float) -> dict:
        """
        Computes Un for each order from time current_time to
        current_time + dt.

        Parameters
        ----------
        current_time : float
            The current time where to start the evolution for
            a time dt. current_time can be positive or negative.

        Returns
        -------
        Uns : dict
            Un for each order from time current_time to
            current_time + dt.
        """
        Uns = {}
        Sns = self._Sns
        for n in range(self.max_order + 1):
            omega_vectors = np.array(
                list(itertools.product([self.omega, -self.omega], repeat=n))
            )

            U_n = np.zeros(
                (len(self.eigenenergies), len(self.eigenenergies)),
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
    A generator of propagator(s) using the Dysolve algorithm.
    See https://arxiv.org/abs/2012.09282.

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
        Extra parameters. "max_order" is a given integer to indicate the
        highest order of approximation used to compute the propagators
        (default is 4). "a_tol" is the absolute tolerance used when
        computing the propagators (default is 1e-10).

    Returns
    -------
    Us : Qobj | list[Qobj]
        The time evolution propagator U(t,0) if t is a single number or else
        a list of propagators [U(t[i], t[0])] for all elements t[i] in t.

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    """
    if isinstance(t, float):
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
                    (len(dysolve.eigenenergies), len(dysolve.eigenenergies)),
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
