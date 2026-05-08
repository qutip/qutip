from qutip import Qobj, qeye_like, Coefficient, coefficient
from . import Result
from ..typing import EopsLike
from .cy.dysolve import cy_compute_integrals
from ..core import data as _data

from numpy.typing import ArrayLike
import numpy as np
import scipy.sparse
from scipy.linalg import polar
from numbers import Number
from typing import Literal, NamedTuple, Any
from collections import namedtuple
import itertools

__all__ = ["Dysolve", "dysolve_propagator", "dysolve"]


class Drive(NamedTuple):
    operator: Qobj
    frequency: float
    form: Literal["cos", "sin", "exp"] = "cos"
    envelope: Coefficient | complex = None


DriveType = (
    Drive
    | tuple[Qobj, float]
    | tuple[Qobj, float, Literal["cos", "sin", "exp"]]
    | tuple[Qobj, float, Literal["cos", "sin", "exp"], Coefficient]
)


class Dysolve:
    """
    A solver of closed system using Dyson series.

    Compute the evolution for an Hamiltonian of the form:

        H = H0 + sum_i X_i exp(1j * w_i * t) * E_i(t)

    With ``X_i`` drive operators, ``w_i`` drive frequencies and ``E_i``, an
    envelope over the drive.

    This method computes analytically the propagator over a time interval.
    This allows very fast computation of driven system. It converge faster the
    higher the frequency is, which is the oposite of sesolve.

    Original paper: https://arxiv.org/abs/2012.09282

    Parameters
    ----------
    H_0 : Qobj
        The base Hamiltonian of the system.

    drives : list[tuple[Qobj, float, /, str, Coefficient]]
        The drive of the Hamiltonian
        List of perturbations applied on the system.
        Each perturbation is a tuple of 2 to 4 values:

            (operator, frequency, form, envelope)

        operator : Qobj
            Operator of the perturbation.

        frequency : float
            Frequency of the perturbation.

        form : Literal["cos", "sin", "exp"], default : "cos"
            Which of ``cos(w*t)``, ``sin(w*t)``, ``exp(1j*w*t)`` is the form of
            the drive for the perturbation.

        envelope : Coefficient, optional
            If provided, a slowly varying envelope to the drive. The envelope
            is estimated to be flat over each step of ``options["step_size"]``.
            Note that when used, it is usually good to use very small step size
            with a lower order.

        A namedtuple with these elements is available as `qutip.dysolve.Drive`.

    options : dict, optional
        Extra parameters.

        - | "order"
          | A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.
        - | "a_tol"
          | The absolute tolerance used when computing the propagators
            (default is 1e-10).
        - | "step_size"
          | The maximum time increment used when computing propagators
            (default is 0.1).
        - | store_final_state : bool, False
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | normalize_output : bool, False
          | Normalize output state to hide ODE numerical errors.
        - | eigen : bool, False
          | Whether to diagonalize the base Hamiltonian ``H_0``using eigen
            states or extracting the diagonal and using the non-diagonal as
            a drive with frequency 0. Precomputation is much faster without
            using the eigen basis, but the extra drive increase the numerical
            error significantly.
        - | polar : bool, False
          | Whether to use polar decomposition on the propagator. Improve
            accuracy for low odd order.
            *Can't be used with non-hermitian Hamiltonian.*

    Note
    ----
    The effective Hamiltonian is expected to be hermitian, but this is only
    required for the base part ``H_0``. The drive can be non-hermitian.
    """

    def __init__(
        self,
        H_0: Qobj,
        drives: list[DriveType],
        options: dict[str] = None,
    ):
        # System

        self.options = {
            "order": 4,
            "a_tol": 1e-10,
            "step_size": 0.1,
            "store_final_state": False,
            "store_states": None,
            "normalize_output": False,
            "polar": False,
            "eigen": False,
        }
        if options:
            self.options.update(options)

        self._dims = H_0._dims
        self.dims = H_0.dims
        if not isinstance(drives, list):
            raise TypeError("drives must be a list of tuple")
        drives = drives.copy()

        if self.options["eigen"]:
            self._eigenenergies, self._basis = H_0.eigenstates(
                output_type="oper"
            )
            self._Ddims = H_0.transform(self._basis, inverse=True)._dims
        else:
            self._eigenenergies = np.diag(H_0.full()).real
            self._H_0 = np.diag(self._eigenenergies)
            offdiag = H_0 - Qobj(self._H_0, dims=H_0._dims)
            if offdiag.norm():
                drives.append((offdiag, 0, "exp"))
            self._basis = None
            self._Ddims = H_0._dims

        self.td = False
        perturbations = []
        for perturbation in drives:
            if not isinstance(perturbation, Drive):
                perturbation = Drive(*perturbation)
            oper, omega, form, given_coeff = perturbation
            factor = 1.0
            coeff = coefficient(1.0)

            if isinstance(given_coeff, Coefficient):
                self.td = True
                coeff = given_coeff
            elif given_coeff is not None:
                factor = given_coeff

            oper = self._transform(oper, inverse=True)
            if form == "cos":
                oper = oper * 0.5
                perturbations.append((oper, omega, coeff, factor))
                perturbations.append((oper, -omega, coeff, factor))
            elif form == "sin":
                perturbations.append((oper, omega, coeff, factor * -0.5j))
                perturbations.append((oper, -omega, coeff, factor * 0.5j))
            elif form == "exp":
                perturbations.append((oper, omega, coeff, factor))

        if perturbations:
            self.perturbations = list(zip(*perturbations))
            self.perturbations[1] = np.array(self.perturbations[1])
            self.perturbations[3] = np.array(self.perturbations[3])
        else:
            self.perturbations = [[], [], [], []]

        # Memoization
        self._dt_Sns = {}

        # Time propagator
        self.U = np.eye(len(self._eigenenergies))
        self.t = 0

    def _transform(self, oper, inverse):
        if self.options["eigen"]:
            return oper.transform(self._basis, inverse=inverse)
        else:
            return oper

    def propagator(
        self, t_f: float, t_i: float = 0.0, *, args: dict[str, Any] = None
    ) -> Qobj:
        """
        Computes the propagator from t_i to t_f. If t_i is not provided,
        computes the propagator from 0 to t_f.

        Parameters
        ----------
        t_f : float
            Final time of the evolution.

        t_i : float, default = 0.0
            Initial time of the evolution.

        args : dict, optional
            If provided, update the ``args`` of all envelopes.
            The new arguments will be used for all subsequent calls.

        Returns
        -------
        U : Qobj
            The propagator U(t_f, t_i) from t_i to t_f.

        Notes
        -----
        If t_f - t_i > step_size, splits the evolution into smaller ones
        to then reconstruct U(t_f, t_i).

        Memoization is used. When ``t_f`` is a multiple of step_size,
        first call may be slow but the next calls should be faster.
        """
        if args:
            self.perturbations[2] = [
                coeff.replace_arguments(args)
                for coeff in self.perturbations[2]
            ]
            self.t = 0
            self.U = np.eye(len(self._eigenenergies))

        dt = self.options["step_size"]
        if t_i == 0 and abs(t_f - self.t) < abs(t_f - t_i):
            t_i = self.t
            U = self.U
        else:
            U = np.eye(len(self._eigenenergies))
        time_diff = t_f - t_i
        n_steps = abs(int(time_diff / dt))

        if n_steps == 0:
            pass
        elif time_diff > 0:
            for j in range(0, n_steps):
                U = self._get_subprop(t_i + j * dt) @ U
        else:
            for j in range(n_steps):
                U = self._get_subprop(t_i - j * dt, -dt) @ U

        # We only save propagator at time multiple of step_size
        self.U = U
        self.t = t_i + n_steps * dt * np.sign(time_diff)

        remaining = time_diff - n_steps * dt * np.sign(time_diff)
        if abs(remaining) > self.options["a_tol"]:
            U = self._get_subprop(t_f - remaining, remaining) @ U

        if self.options["polar"]:
            U = polar(U)[0]

        return self._transform(Qobj(U, self._Ddims, copy=None), False)

    def run(
        self,
        state0: Qobj,
        tlist: ArrayLike,
        *,
        e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
        args: dict[str, Any] = None,
    ) -> Result:
        """
        Do the evolution of the Quantum system.

        For a ``state0`` at time ``tlist[0]`` do the evolution as directed by
        given hamiltonian and drive and for each time in ``tlist`` store the
        state and/or expectation values in a :class:`.Result`.

        Parameters
        ----------
        state0 : :obj:`.Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Each times of the list must be increasing, but does not
            need to be uniformy distributed.

        args : dict, optional
            Change the ``args`` of the envelopesn.
            The new arguments will be used for all subsequent calls.

        e_ops : Qobj, QobjEvo, callable, list, or dict optional
            Single, list or dict of Qobj, QobjEvo or callable to compute the
            expectation values. Function[s] must have the signature
            f(t : float, state : Qobj) -> Any.

        Returns
        -------
        results : :obj:`.Result`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        if args:
            self.perturbations[2] = [
                coeff.replace_arguments(args)
                for coeff in self.perturbations[2]
            ]
            self.t = 0
            self.U = np.eye(len(self._eigenenergies))

        dt = self.options["step_size"]
        if state0._dims[0] != self._dims[1]:
            raise TypeError(
                f"incompatible dimensions {self.dims}"
                f" and {state0.dims}"
            )

        results = Result(
            e_ops,
            self.options,
            solver="Dyson Series",
            stats={},
        )
        results.add(tlist[0], state0)

        t_state = tlist[0]

        state = self._transform(state0, inverse=True).data

        for t in tlist[1:]:
            time_diff = t - t_state
            n_steps = int(time_diff / dt)
            if n_steps == 0:
                pass
            elif time_diff > 0:
                for _ in range(0, n_steps):
                    state = _data.Dense(self._get_subprop(t_state)) @ state
                    t_state += dt
            remaining = t - t_state
            if abs(remaining) > self.options["a_tol"]:
                state_t = (
                    _data.Dense(self._get_subprop(t - remaining, remaining))
                    @ state
                )
            else:
                state_t = state
            results.add(
                t,
                self._transform(
                    Qobj(state_t, dims=[self._Ddims[0], [1]]), inverse=False
                ),
            )

        return results

    def _get_subprop(self, current_time: float, dt: float = None) -> ArrayLike:
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
        if dt is None:
            dt = self.options["step_size"]
        Sns = self._compute_Sns(dt)

        N = len(self._eigenenergies)
        num_omega = len(self.perturbations[1])

        subpropagator = np.zeros((N, N), dtype=np.complex128)
        subpropagator += Sns[0]

        ws_vec = self.perturbations[1]
        ws = np.zeros((N, N))
        if self.td:
            envelopes = np.array([
                coeff(current_time + dt / 2)
                for coeff in self.perturbations[2]
            ])

        for n in range(1, self.options["order"] + 1):
            ws = np.add.outer(ws_vec, ws)
            Sns_n = Sns[n]
            if self.td:
                Sns_n = np.einsum(
                    Sns_n,
                    np.arange(n + 2),
                    *[envelopes, [0]] * n,
                    np.arange(n + 2),
                )
            subpropagator += (
                (Sns_n * np.exp(1j * ws * current_time))
                .reshape((-1, N, N))
                .sum(axis=0)
            )

        return subpropagator

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
            self._dt_Sns[dt] = self._make_SNS_sparse(dt)
            return self._dt_Sns[dt]

    def _make_SNS_sparse(self, dt: float) -> dict:

        def _to_COO_format(matrix):
            """
            Convert to COO, (location, data) pair.
            """
            coo = scipy.sparse.coo_array(matrix)
            idx = np.c_[coo.row, coo.col]
            return idx, coo.data

        def _outer_matmul(left_coo, right_coo):
            """
            Outer matmul of COO arrays: einsum("abc,cde->abcde")
            """
            idx_l, data_l = left_coo
            idx_r, data_r = right_coo
            new_idx = []
            new_data = []

            for j in range(idx_r.shape[0]):
                idx = np.where(idx_l[:, -1] == idx_r[j, 0])[0]
                for i in idx:
                    new_idx.append(np.r_[idx_l[i, :], idx_r[j, 1:]])
                    new_data.append(data_l[i] * data_r[j])
            return np.array(new_idx), np.array(new_data)

        Sns = {}
        energies = self._eigenenergies
        num_ls = len(energies)
        omegas = self.perturbations[1]
        num_ws = len(omegas)
        factors = self.perturbations[3]

        dE = energies[:, np.newaxis] - energies[np.newaxis, :]
        exp_H_0_diag = np.exp(-1j * energies * dt)
        exp_H_0 = np.diag(exp_H_0_diag)
        Sns[0] = exp_H_0

        # Since usually the operators comes in pairs, we only
        # store unique operators and keeps and index from the
        # perturbation index to it's location.
        path_cache = []
        opers_loc = []
        for i, oper in enumerate(self.perturbations[0]):
            if oper not in path_cache:
                opers_loc.append(len(path_cache))
                path_cache.append(oper)
            else:
                opers_loc.append(path_cache.index(oper))

        path_cache = {
            (idx,): _to_COO_format(oper.to("csr").data_as())
            for idx, oper in enumerate(path_cache)
        }

        for n in range(1, self.options["order"] + 1):
            shape = [num_ws] * n + [num_ls, num_ls]
            Sn = np.zeros(shape, dtype=np.complex128)

            for pert_idx in itertools.product(range(num_ws), repeat=n):
                current_omegas = [omegas[i] for i in pert_idx]
                unique_idx = tuple(opers_loc[idx] for idx in pert_idx)
                if unique_idx not in path_cache:
                    path_cache[unique_idx] = _outer_matmul(
                        path_cache[unique_idx[:1]], path_cache[unique_idx[1:]]
                    )

                paths, amplitudes = path_cache[unique_idx]
                if paths.size == 0:
                    continue
                factor = factors[list(pert_idx)].prod()
                ws_matrix = np.zeros((len(amplitudes), n))
                for i in range(n):
                    ws_matrix[:, i] = (
                        current_omegas[i]
                        + dE[paths[:, i], paths[:, i + 1]]
                    )

                integrals = np.array(
                    [cy_compute_integrals(row[::-1], dt) for row in ws_matrix]
                )
                start_indices = paths[:, 0]
                end_indices = paths[:, -1]
                np.add.at(
                    Sn[*pert_idx],
                    (start_indices, end_indices),
                    amplitudes * integrals * factor,
                )

                Sn[*pert_idx] = exp_H_0 @ Sn[*pert_idx]

            Sn *= (-1j) ** n
            Sns[n] = Sn

        return Sns


def dysolve_propagator(
    H_0: Qobj,
    drives: list[DriveType],
    t: float | list[float],
    args: dict[str, Any] = None,
    options: dict[str] = None,
) -> Qobj | list[Qobj]:
    """
    A generator of propagator(s) using Dyson series.

    Compute the propagator for an Hamiltonian of the form:

        H = H0 + sum_i X_i exp(1j * w_i * t) * E_i(t)

    This method computes analytically the propagator over a time interval.
    This allows very fast computation of driven system. It converge faster the
    higher the frequency is, which is the oposite of sesolve.

    Original paper: https://arxiv.org/abs/2012.09282

    Parameters
    ----------
    H_0 : Qobj
        The hamiltonian of the system.

    drives : list[tuple[Qobj, float, /, str, Coefficient]]
        The drives of the Hamiltonian
        List of perturbations applied on the system.
        Each perturbation is a tuple of 2 to 4 values:

            (operator, frequency, form, envelope)

        operator : Qobj
            Operator of the perturbation.

        frequency : float
            Frequency of the perturbation.

        form : Literal["cos", "sin", "exp"], default : "cos"
            Which of ``cos(w*t)``, ``sin(w*t)``, ``exp(1j*w*t)`` is the form of
            the drive for the perturbation.

        envelope : Coefficient, optional
            If provided, a slowly varying envelope to the drive. The envelope
            is estimated to be flat over each step of ``options["step_size"]``.
            Note that when used, it is usually good to use very small step size
            with a lower order.

        A namedtuple with these elements is available as `qutip.dysolve.Drive`.

    t : float | list[float]
        Time or list of times for which to evaluate the propagator(s). If t
        is a single number, the propagator from 0 to t is computed. When
        t is a list, the propagators from the first time to each elements in
        t is returned. In that case, the first output will always be the
        identity matrix. Also, in that case, having a time increment multiple
        of ``option["step_size"]`` between elements will give better
        performance.

    args : dict, optional
        If provided, update the ``args`` of all envelopes.

    options : dict, optional
        Extra parameters.

        - | "order"
          | A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.
        - | "a_tol"
          | The absolute tolerance used when computing the propagators
            (default is 1e-10).
        - | "step_size"
          | The maximum time increment used when computing propagators
            (default is 0.1).

    Returns
    -------
    Us : Qobj | list[Qobj]
        The time evolution propagator U(t,0) if t is a single number or else
        a list of propagators [U(t[i], t[0])] for all elements t[i] in t.

    Note
    ----
    The effective Hamiltonian is expected to be hermitian, but this is only
    needed when computing the propagator with negative times differences:
    ``Dysolve.propagator(-1)`` or ``Dysolve.propagator(0, 1)``.
    """
    dysolve = Dysolve(H_0, drives, options)
    if args:
        dysolve.propagator(0, args=args)

    if isinstance(t, Number):
        Us = dysolve.propagator(t)
    else:
        Us = []
        Us.append(qeye_like(H_0))
        if t[0] != 0.0:
            pre = dysolve.propagator(t[0]).inv()
        else:
            pre = None
        for i in range(len(t[:-1])):
            U = dysolve.propagator(t[i + 1])
            if pre is not None:
                U = U @ pre
            Us.append(U)

    return Us


def dysolve(
    H_0: Qobj,
    drives: list[DriveType],
    psi0: Qobj,
    tlist: ArrayLike,
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict["str", Any] = None,
    options: dict[str] = None,
) -> Qobj | list[Qobj]:
    """
    Solve the Schrodinger equation for a driven system using a Dyson series
    expansion.

    Compute the evolution for an Hamiltonian of the form:

        H = H0 + sum_i X_i exp(1j * w_i * t) * E_i(t)

    It's optimized for high-frequency perturbations where standard ODE solvers
    (like sesolve) become computationally expensive.

    Original paper: https://arxiv.org/abs/2012.09282

    Parameters
    ----------
    H_0 : Qobj
        The hamiltonian of the system.

    drives : list[tuple[Qobj, float, /, str, Coefficient]]
        The drives of the Hamiltonian
        List of perturbations applied on the system.
        Each perturbation is a tuple of 2 to 4 values:

            (operator, frequency, form, envelope)

        operator : Qobj
            Operator of the perturbation.

        frequency : float
            Frequency of the perturbation.

        form : Literal["cos", "sin", "exp"], default : "cos"
            Which of ``cos(w*t)``, ``sin(w*t)``, ``exp(1j*w*t)`` is the form of
            the drive for the perturbation.

        envelope : Coefficient, optional
            If provided, a slowly varying envelope to the drive. The envelope
            is estimated to be flat over each step of ``options["step_size"]``.
            Note that when used, it is usually good to use very small step size
            with a lower order.

        A namedtuple with these elements is available as `qutip.dysolve.Drive`.

    psi0 : :obj:`.Qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`

    e_ops : :obj:`.Qobj`, callable, list or dict, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.

    args : dict, optional
        If provided, update the ``args`` of all envelopes.

    options : dict, optional
        Extra parameters.

        - | "order"
          | A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.
        - | "a_tol"
          | The absolute tolerance used when computing the propagators
            (default is 1e-10).
        - | "step_size"
          | The maximum time increment used when computing propagators
            (default is 0.1).
        - | store_final_state : bool, False
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | normalize_output : bool, False
          | Normalize output state to hide ODE numerical errors.

    Returns
    -------
    Us : Qobj | list[Qobj]
        The time evolution propagator U(t,0) if t is a single number or else
        a list of propagators [U(t[i], t[0])] for all elements t[i] in t.
    """
    dysolve = Dysolve(H_0, drives, options)
    return dysolve.run(psi0, tlist, e_ops=e_ops, args=args)


# There is a name collision with the filename and function.
# ``Drive`` is too generic to be in the global namespace so we make it
# available as qutip.dysolve.Drive artificially with this:
dysolve.Drive = Drive
