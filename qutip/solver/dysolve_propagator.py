from qutip import Qobj
import numpy as np
import itertools
from scipy.special import factorial

__all__ = ['DysolvePropagator', 'dysolve_propagator']


class DysolvePropagator:
    """
    A generator of propagator(s) using the Dysolve algorithm
    (see https://arxiv.org/abs/2012.09282).

    Parameters
    ----------
    H_0 : Qobj
        The hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    options : dict, optional
        Extra parameters when creating a DysolvePropagator instance.
        "max_order" is a given integer to indicate the highest order
        of approximation used to compute the propagators (default is 4).
        "a_tol" is the absolute tolerance used when computing the propagators
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
        self.H_0 = H_0
        self.eigenenergies = H_0.eigenenergies()
        self.X = X.transform(H_0)
        self.omega = omega

        # Times
        self.t_i = None
        self.t_f = None
        self.dt = None
        self.times = None

        # Options
        if options is not None:
            if options.get('max_order') is not None:
                self.max_order = options['max_order']
            if options.get('a_tol') is not None:
                self.a_tol = options['a_tol']
            else:
                raise KeyError('Incorrect keys for options have been given')
        else:
            self.max_order = 4
            self.a_tol = 1e-10

        self._Sns = None
        self.Us = None

    def __call__(self, t_i: float, t_f: float, dt: float) -> list[Qobj]:
        """
        Computes propagators for all time increments that fit in the
        range [t_i, t_f]. Each increment is separated by dt.

        Parameters
        ----------
        t_i : float
            Initial time of the evolution.

        t_f : float
            Final time of the evolution.

        dt : float
            The time increment in between each step of the
            evolution. If a time increment exceeds t_f, the propagator
            for that time period will not be calculated. 

        Returns
        -------
        Us : list[Qobj]
            The propagators for all time increments in the given range
            of time. So, [U(times[1], times[0]), U(times[2], times[1]) ...].

        """
        self.t_i = t_i
        self.t_f = t_f
        self.dt = dt

        if np.arange(t_i, t_f, dt)[-1] < self.t_f:
            if np.arange(t_i, t_f, dt)[-1] + self.dt > self.t_f:
                self.times = np.arange(t_i, t_f, dt)[:-1]
            else:
                self.times = np.arange(t_i, t_f, dt)
        else:
            self.times = np.arange(t_i, t_f, dt)[:-1]

        self._Sns = self._compute_Sns()

        Us = []
        for time in self.times[:]:
            U = np.zeros(
                (len(self.eigenenergies), len(self.eigenenergies)),
                dtype=np.complex128
            )
            Uns = self._compute_Uns(time)
            for n in range(self.max_order + 1):
                U += Uns[n]
            Us.append(Qobj(U, dims=self.H_0.dims))

        self.Us = Us
        return Us

    def _compute_integrals(self, ws: list) -> float:
        """
        Computes the value of the nested integrals for a given list of
        effective omegas.

        Parameters
        ----------
        ws : list
            A list of effective omegas. ws[0] is the omega for the rightmost
            integral.

        Returns
        -------
        value : float
            The value of the nested integrals.

        Notes
        -----
        Integrals are done from right to left.

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
                ws_prime = ws[1:]
                ws_prime[0] += ws[0]
                return (-1j / ws[0]) * (
                    self._compute_integrals(
                        ws_prime) - self._compute_integrals(ws[1:])
                )

    def _compute_tn_integrals(self, ws: list, n: int) -> float:
        """
        Helper function to compute nested integrals when the function to
        integrate is t^n/factorial(n) * exp(1j*omega*t). This happens when
        some effective omegas are 0.

        Paramaters
        ----------
        ws : list
            A list of effective omegas. ws[0] is the omega for the rightmost
            integral.

        n : int
            An increment used in t^n/factorial(n).

        Returns
        -------
        value : float
            The value of the nested integrals when the function to integrate is
            t^n/factorial(n) * exp(1j*omega*t).

        """
        if n == 0:
            return self._compute_integrals(ws)

        if len(ws) == 1:
            if np.abs(ws[0]) < self.a_tol:
                return self.dt ** (n + 1) / factorial(n + 1)
            else:
                factor = -1j / ws[0]
                term1 = (self.dt**n / factorial(n)) * \
                    np.exp(1j * ws[0] * self.dt)
                term2 = self._compute_tn_integrals(ws, n - 1)
                return factor * (term1 - term2)
        else:
            if np.abs(ws[0]) < self.a_tol:
                return self._compute_tn_integrals(ws[1:], n + 1)
            else:
                factor = -1j / ws[0]
                ws_prime = ws[1:]
                ws_prime[0] += ws[0]
                term1 = self._compute_tn_integrals(ws_prime, n)
                term2 = self._compute_tn_integrals(ws, n - 1)
                return factor * (term1 - term2)

    def _compute_matrix_elements(self, i_j_indices: list) -> list:
        """
        Computes the products of matrix elements for each term in the
        sum for Sns.

        Parameters
        ----------
        i_j_indices : list
            The indices for the eigenenergies/eigenstates. Matchs the order of
            combination used to compute the effective omegas.

        Returns
        -------
        matrix_elements : list
            The products of matrix elements for each term in the sum for Sns.

        """
        matrix_elements_products = np.ones(
            (i_j_indices.shape[0]), dtype=np.complex128)

        for i in range(i_j_indices.shape[1] - 1):
            matrix_elements_products *= self.X[
                i_j_indices[:, 0 + i: 2 + i][:, 0],
                i_j_indices[:, 0 + i: 2 + i][:, 1]
            ].A1

        return matrix_elements_products * np.exp(
            -1j * self.eigenenergies[i_j_indices[:, -1]] * self.dt
        )

    def _compute_Sns(self) -> dict:
        """
        Computes Sns for each omega vector.

        Returns
        -------
        Sns : dict
            Sns for each omega vector (key = order).

        """
        Sns = {}
        for n in range(self.max_order + 1):
            omega_vectors = np.array(
                list(itertools.product([self.omega, -self.omega], repeat=n))
            )
            diff_lambdas = np.diff(
                np.array(list(itertools.product(
                    self.eigenenergies, repeat=n + 1
                )))[:, ::-1],
                axis=1,
            )
            eff_omegas = omega_vectors[:, None, :] + diff_lambdas[None, :, :]

            # Compute integrals
            integrals = np.zeros(
                (eff_omegas.shape[0], eff_omegas.shape[1]), dtype=np.complex128
            )
            for i in range(eff_omegas.shape[0]):
                for j in range(eff_omegas.shape[1]):
                    integrals[i, j] = self._compute_integrals(
                        list(eff_omegas[i, j, :]))

            # Compute matrix elements
            i_j_indices = np.array(
                list(itertools.product(
                    range(len(self.eigenenergies)), repeat=n + 1
                )))[:, ::-1]
            ket_bra_indices = i_j_indices[:, [0, -1]]
            matrix_elements = self._compute_matrix_elements(i_j_indices)

            # Compute Sns
            factor = (-1j / 2) ** n
            x = factor * matrix_elements * integrals
            Sn = np.zeros(
                (len(omega_vectors), len(self.eigenenergies),
                 len(self.eigenenergies)),
                dtype=np.complex128,
            )

            for i in range(len(omega_vectors)):
                for j, idx in enumerate(ket_bra_indices):
                    Sn[i, idx[1], idx[0]] += x[i, j]

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
            Corresponds to one of the times in self.times.

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
        max_order: int,
        H_0: Qobj,
        X: Qobj,
        omega: float,
        t_i: float,
        t_f: float,
        dt: float,
        a_tol: float = 1e-10,
) -> tuple[DysolvePropagator, list[Qobj]]:
    """
    Calculates the time evolution propagators from t_i to
    all time increments that fit in the range [t_i, t_f]
    using the Dysolve algorithm.

    See https://arxiv.org/abs/2012.09282.

    Parameters
    ----------
    max_order : int
        The maximum order of approximation for the time evolution.
        The bigger this variable is, the more terms are calculated,
        so more accuracy but a longer time to compute the propagator(s).

    H_0 : Qobj
        The hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    t_i : float
        Initial time of the evolution.

    t_f : float
        Final time of the evolution.

    dt : float
        The time increment in between each step of the
        evolution. If time + dt exceeds t_f the propagator for that
        time period will not be calculated. Only the propagators
        with time period inside [t_i, t_f] are returned.

    a_tol : float, default: 1e-10
        The absolute tolerance when it comes to say if values that
        are computed are small enough to be considered as 0.

    Returns
    -------
    (dysolve_instance, propagators): tuple[DysolvePropagator, list[Qobj]]
        The DysolvePropagator class instance formed the entries and
        the time evolution propagators from t_i to all time increments
        that fit in the range [t_i, t_f].

        So, [U(t_i + dt, t_i), U(t_i + 2*dt, t_i), ...].

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    """
    dysolve = DysolvePropagator(max_order, H_0, X, omega, a_tol)
    dysolve(t_i, t_f, dt,)
    Us = dysolve.Us

    for i in range(len(Us)):
        if i != 0:
            Us[i] = Us[i] @ Us[i - 1]

    return dysolve, Us
