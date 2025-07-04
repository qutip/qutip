from qutip import Qobj, qeye_like
from .cy.dysolve import cy_compute_integrals
from numpy.typing import ArrayLike
import numpy as np
import scipy as sp
from numbers import Number
import itertools


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
        self._H_0 = H_0.transform(self._basis)
        self._X = X.transform(self._basis)
        self._elems = self._X.full().flatten()
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

        # Memoization
        self._dt_Sns = {}

        # Time propagator
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

        U = self._compute_subprop(t_i, dt)

        for j in range(1, n_steps):
            U = self._compute_subprop(t_i + j*dt, dt) @ U

        remaining = time_diff - n_steps*dt
        if abs(remaining) > self.a_tol:
            dt = remaining
            U = self._compute_subprop(t_f - dt, dt) @ U

        self.U = Qobj(U, self._H_0.dims, copy=False).transform(
            self._basis, True
        )

        return self.U

    def _update_matrix_elements(self, current: ArrayLike) -> ArrayLike:
        """
        Reuses the current matrix elements (order n-1) to compute the
        matrix elements for the order n.

        Parameters
        ----------
        current : ArrayLike
            The current matrix elements (for the order n-1)..

        Returns
        -------
        matrix_elements : ArrayLike
            The new matrix elements for the order n.
        """
        if current is None:
            return self._elems
        else:
            shape = self._X.shape[0]
            a = np.tile(current, shape)
            b = np.repeat(self._elems, len(current)//shape)
            return a * b

        # WIP:
        # The following is an attempt to use scipy.sparse to store "current".
        # There can be a lot of zeros, so scipy.sparse could be useful here.
        # This involves more operations than the implementation above because
        # "current" is a vector and scipy.sparse only accepts matrices.
        # A better approach is necessary to use the full potential of
        # scipy.sparse.

        # if current is None:
        #     return elems
        # else:
        #     x_shape = self._X.shape[0]
        #     a = sp.sparse.hstack([current]*x_shape)
        #     b = sp.sparse.vstack(
        #             [elems]*(current.shape[1]//x_shape)
        #     ).transpose().reshape(a.shape)
        #     return a.multiply(b)

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
            exp_H_0 = (-1j*dt*self._H_0).expm().full()
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

                    for j, ws in enumerate(ls_ws):
                        if current_matrix_elements[j] != 0:
                            x = cy_compute_integrals(
                                ws, dt) * current_matrix_elements[j]
                            Sn[i, ket_bra_idx[j, 0], ket_bra_idx[j, 1]] += x

                    Sn[i] *= (-1j / 2) ** n
                    Sn[i] = exp_H_0 @ Sn[i]

                Sns[n] = Sn

            self._dt_Sns[dt] = Sns
            return Sns

    def _compute_subprop(self, current_time: float, dt: float) -> ArrayLike:
        """
        Computes a subpropagator U(current_time + dt, current_time).

        Parameters
        ----------
        current_time : float
            The starting time of the evolution. Can be positive or negative.

        dt : float
            The time increment.

        Returns
        -------
        subpropagator : ArrayLike
            U(current_time + dt, current_time).

        """
        Sns = self._compute_Sns(dt)

        subpropagator = np.zeros(
            (len(self._eigenenergies), len(self._eigenenergies)),
            dtype=np.complex128
        )
        subpropagator += Sns[0]

        for n in range(1, self.max_order + 1):
            omega_vectors = np.fromiter(
                itertools.product([self._omega, -self._omega], repeat=n),
                np.dtype((float, (n,)))
            )
            subpropagator += sum(
                Sns[n] * np.exp(1j*np.sum(omega_vectors, axis=1)*current_time)
                [:, np.newaxis, np.newaxis]
            )

        return subpropagator


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
