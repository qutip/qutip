# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = [
    "FloquetBasis",
    "floquet_tensor",
    "fsesolve",
    "fmmesolve",
    "FMESolver",
]

from typing import Any, overload, TypeVar, Literal, Callable
import warnings
import numpy as np
from numpy.typing import ArrayLike
from qutip.core import data as _data
from qutip.core.data import Data
from qutip import Qobj, QobjEvo
from .propagator import Propagator
from .mesolve import MESolver
from .solver_base import Solver
from .integrator import Integrator
from .result import Result
from time import time
from ..ui.progressbar import progress_bars
from ..typing import EopsLike, QobjEvoLike, QobjOrData


class FloquetBasis:
    """
    Utility to compute floquet modes and states.

    Attributes
    ----------
    U : :class:`.Propagator`
        The propagator of the Hamiltonian over one period.

    evecs : :class:`.Data`
        Matrix where each column is an initial Floquet mode.

    e_quasi : np.ndarray[float]
        The quasi energies of the Hamiltonian.
    """

    def __init__(
        self,
        H: QobjEvoLike,
        T: float,
        args: dict[str, Any] = None,
        options: dict[str, Any] = None,
        sparse: bool = False,
        sort: bool = True,
        precompute: ArrayLike = None,
        times: ArrayLike = None,
    ):
        """
        Parameters
        ----------
        H : :obj:`.Qobj`, :obj:`.QobjEvo`, QobjEvo compatible format.
            System Hamiltonian, with period `T`.

        T : float
            Period of the Hamiltonian.

        args : None / *dictionary*
            dictionary of parameters for time-dependent Hamiltonians and
            collapse operators.

        options : dict [None]
            Options used by sesolve to compute the floquet modes.

        sparse : bool [False]
            Whether to use the sparse eigen solver when computing the
            quasi-energies.

        sort : bool [True]
            Whether to sort the quasi-energies.

        precompute : list [None]
            If provided, a list of time at which to store the propagators
            for later use when computing modes and states. Default is
            ``linspace(0, T, 101)`` corresponding to the default integration
            steps used for the floquet tensor computation.

        times : ArrayLike [None]
            Time for array
        """
        if not T > 0:
            raise ValueError("The period need to be a positive number.")
        self.T = T
        H = QobjEvo(H, args=args, tlist=times)
        if precompute is not None:
            tlist = np.unique(np.atleast_1d(precompute) % self.T)
            memoize = len(tlist)
            if tlist[0] != 0:
                memoize += 1
            if tlist[-1] != T:
                memoize += 1
        else:
            # Default computation
            tlist = np.linspace(0, T, 101)
            memoize = 101
        if H._feedback_functions or H._solver_only_feedback:
            raise NotImplementedError("FloquetBasis does not support feedback")
        self.U = Propagator(H, options=options, memoize=memoize)
        for t in tlist:
            # Do the evolution by steps to save the intermediate results.
            self.U(t)
        U_T = self.U(self.T)
        if not sparse and isinstance(U_T.data, _data.CSR):
            U_T = U_T.to("Dense")
        evals, evecs = _data.eigs(U_T.data)
        e_quasi = -np.angle(evals) / T
        if sort:
            perm = np.argsort(e_quasi)
            self.evecs = _data.permute.indices(evecs, col_perm=np.argsort(perm))
            self.e_quasi = e_quasi[perm]
        else:
            self.evecs = evecs
            self.e_quasi = e_quasi

    def _as_ketlist(self, kets_mat):
        """
        Split the Data array in a list of kets.
        """
        dims = [self.U(0).dims[0], [1]]
        return [
            Qobj(ket, dims=dims)
            for ket in _data.split_columns(kets_mat)
        ]

    @overload
    def mode(self, t: float, data: Literal[False]) -> Qobj: ...

    @overload
    def mode(self, t: float, data: Literal[True]) -> Data: ...

    def mode(self, t: float, data: bool = False):
        """
        Calculate the Floquet modes at time ``t``.

        Parameters
        ----------
        t : float
            The time for which to evaluate the Floquet mode.

        data : bool [False]
            Whether to return the states as a single data matrix or a list of
            ket states.

        Returns
        -------
        output : list[:obj:`.Qobj`], :class:`.Data`
            A list of Floquet states for the time ``t`` or the states as column
            in a single matrix.
        """
        t = t % self.T
        if t == 0.0:
            kets_mat = self.evecs
        else:
            U = self.U(t).data
            phases = _data.diag(np.exp(1j * t * self.e_quasi))
            kets_mat = U @ self.evecs @ phases
        if data:
            return kets_mat
        else:
            return self._as_ketlist(kets_mat)

    @overload
    def state(self, t: float, data: Literal[False]) -> Qobj: ...

    @overload
    def state(self, t: float, data: Literal[True]) -> Data: ...

    def state(self, t: float, data: bool = False):
        """
        Evaluate the floquet states at time t.

        Parameters
        ----------
        t : float
            The time for which to evaluate the Floquet states.

        data : bool [False]
            Whether to return the states as a single data matrix or a list of
            ket states.

        Returns
        -------
        output : list[:obj:`.Qobj`], :class:`.Data`
            A list of Floquet states for the time ``t`` or the states as column
            in a single matrix.
        """
        if t:
            phases = _data.diag(np.exp(-1j * t * self.e_quasi))
            states_mat = self.mode(t, True) @ phases
        else:
            states_mat = self.evecs
        if data:
            return states_mat
        else:
            return self._as_ketlist(states_mat)

    def from_floquet_basis(
        self,
        floquet_basis: QobjOrData,
        t: float = 0
    ) -> QobjOrData:
        """
        Transform a ket or density matrix from the Floquet basis at time ``t``
        to the lab basis.

        Parameters
        ----------
        floquet_basis : :obj:`.Qobj`, :class:`.Data`
            Initial state in the Floquet basis at time ``t``. May be either a
            ket or density matrix.

        t : float [0]
            The time at which to evaluate the Floquet states.

        Returns
        -------
        output : :obj:`.Qobj`, :class:`.Data`
            The state in the lab basis. The return type is the same as the type
            of the input state.
        """
        is_Qobj = isinstance(floquet_basis, Qobj)
        if is_Qobj:
            dims = floquet_basis.dims
            floquet_basis = floquet_basis.data
            if dims[0] != self.U(0).dims[1]:
                raise ValueError(
                    "Dimensions of the state do not match the Hamiltonian"
                )

        state_mat = self.state(t, True)
        lab_basis = state_mat @ floquet_basis
        if floquet_basis.shape[1] != 1:
            lab_basis = lab_basis @ state_mat.adjoint()

        if is_Qobj:
            return Qobj(lab_basis, dims=dims)
        return lab_basis

    def to_floquet_basis(
        self,
        lab_basis: QobjOrData,
        t: float = 0
    ) -> QobjOrData:
        """
        Transform a ket or density matrix in the lab basis
        to the Floquet basis at time ``t``.

        Parameters
        ----------
        lab_basis : :obj:`.Qobj`, :class:`.Data`
            Initial state in the lab basis.

        t : float [0]
            The time at which to evaluate the Floquet states.

        Returns
        -------
        output : :obj:`.Qobj`, :class:`.Data`
            The state in the Floquet basis. The return type is the same as the
            type of the input state.
        """
        is_Qobj = isinstance(lab_basis, Qobj)
        if is_Qobj:
            dims = lab_basis.dims
            lab_basis = lab_basis.data
            if dims[0] != self.U(0).dims[1]:
                raise ValueError(
                    "Dimensions of the state do not match the Hamiltonian"
                )

        state_mat = self.state(t, True)
        floquet_basis = state_mat.adjoint() @ lab_basis
        if lab_basis.shape[1] != 1:
            floquet_basis = floquet_basis @ state_mat

        if is_Qobj:
            return Qobj(floquet_basis, dims=dims)
        return floquet_basis


def _floquet_delta_tensor(f_energies, kmax, T):
    """
    Floquet-Markov master equation X matrices.

    Parameters
    ----------
    f_energies : np.ndarray
        The Floquet energies.

    kmax : int
        The truncation of the number of sidebands (default 5).

    T : float
        The period of the time-dependence of the Hamiltonian.

    Returns
    -------
    delta : np.ndarray
        Floquet delta tensor.
    """
    delta = np.subtract.outer(f_energies, f_energies)
    return np.add.outer(delta, np.arange(-kmax, kmax + 1) * (2 * np.pi / T))


def _floquet_X_matrices(floquet_basis, c_ops, kmax, ntimes=100):
    """
    Floquet-Markov master equation X matrices.

    Parameters
    ----------
    floquet_basis : :class:`FloquetBasis`
        The system Hamiltonian wrapped in a FloquetBasis object.

    c_ops : list of :obj:`.Qobj`
        The collapse operators describing the dissipation.

    kmax : int
        The truncation of the number of sidebands (default 5).

    ntimes : int [100]
        The number of integration steps (for calculating X) within one period.

    Returns
    -------
    X : list of dict of :class:`.Data`
        A dict of the sidebands ``k`` for the X matrices of each c_ops
    """
    T = floquet_basis.T
    N = floquet_basis.U(0).shape[0]
    omega = (2 * np.pi) / T
    tlist = np.linspace(T / ntimes, T, ntimes)
    ks = np.arange(-kmax, kmax + 1)
    out = {k: [_data.csr.zeros(N, N)] * len(c_ops) for k in ks}

    for t in tlist:
        mode = floquet_basis.mode(t, data=True)
        FFs = [mode.adjoint() @ c_op.data @ mode for c_op in c_ops]
        for k, phi in zip(ks, np.exp(-1j * ks * omega * t) / ntimes):
            out[k] = [
                _data.add(prev, new, phi) for prev, new in zip(out[k], FFs)
            ]

    return [{k: out[k][i] for k in ks} for i in range(len(c_ops))]


def _floquet_gamma_matrices(X, delta, J_cb):
    """
    Floquet-Markov master equation gamma matrices.

    Parameters
    ----------
    X : list of dict of :class:`.Data`
        Floquet X matrices created by :func:`_floquet_X_matrices`.

    delta : np.ndarray
        Floquet delta tensor created by :func:`_floquet_delta_tensor`.

    J_cb : list of callables
        A list callback functions that compute the noise power spectrum as
        a function of frequency. The list should contain one callable for each
        collapse operator `c_op`, in the same order as the elements of `X`.
        Each callable should accept a numpy array of frequencies and return a
        numpy array of corresponding noise power.

    Returns
    -------
    gammas : dict of :class:`.Data`
        A dict mapping the sidebands ``k`` to their gamma matrices.
    """
    N = delta.shape[0]
    kmax = (delta.shape[2] - 1) // 2
    gamma = {k: _data.csr.zeros(N, N) for k in range(-kmax, kmax + 1, 1)}

    for X_c_op, sp in zip(X, J_cb):
        response = sp(delta) * ((2 + 0j) * np.pi)
        response = [
            _data.Dense(response[:, :, k], copy=False)
            for k in range(2 * kmax + 1)
        ]
        for k in range(-kmax, kmax + 1, 1):
            gamma[k] = _data.add(
                gamma[k],
                _data.multiply(
                    _data.multiply(X_c_op[k].conj(), X_c_op[k]),
                    response[k + kmax],
                ),
            )
    return gamma


def _floquet_A_matrix(delta, gamma, w_th):
    """
    Floquet-Markov master equation rate matrix.

    Parameters
    ----------
    delta : np.ndarray
        Floquet delta tensor created by :func:`_floquet_delta_tensor`.

    gamma : dict of :class:`.Data`
        Floquet gamma matrices created by :func:`_floquet_gamma_matrices`.

    w_th : float
        The temperature in units of frequency.
    """
    kmax = (delta.shape[2] - 1) // 2

    if w_th > 0.0:
        deltap = np.copy(delta)
        deltap[deltap == 0.0] = np.inf
        thermal = 1.0 / (np.exp(np.abs(deltap) / w_th) - 1.0)
        thermal = [_data.Dense(thermal[:, :, k]) for k in range(2 * kmax + 1)]

        gamma_kk = _data.add(gamma[0], gamma[0].transpose())
        A = _data.add(gamma[0], _data.multiply(thermal[kmax], gamma_kk))

        for k in range(1, kmax + 1):
            g_kk = _data.add(gamma[k], gamma[-k].transpose())
            thermal_kk = _data.multiply(thermal[kmax + k], g_kk)
            A = _data.add(A, _data.add(gamma[k], thermal_kk))
            thermal_kk = _data.multiply(thermal[kmax - k], g_kk.transpose())
            A = _data.add(A, _data.add(gamma[-k], thermal_kk))
    else:
        # w_th is 0, thermal = 0s
        A = gamma[0]
        for k in range(1, kmax + 1):
            A = _data.add(gamma[k], A)
            A = _data.add(gamma[-k], A)

    return A


def _floquet_master_equation_tensor(A):
    """
    Construct a tensor that represents the master equation in the floquet
    basis (with constant Hamiltonian and collapse operators?).

    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]

    Parameters
    ----------
    A : :class:`.Data`
        Floquet-Markov master equation rate matrix.

    Returns
    -------
    output : array
        The Floquet-Markov master equation tensor `R`.
    """
    N = A.shape[0]

    # R[i+N*i, j+N*j] = A[j, i]
    cols = np.arange(N, dtype=np.int32)
    rows = np.linspace(1 - 1 / (N + 1), N, N**2 + 1, dtype=np.int32)
    data = np.ones(N, dtype=complex)
    expand = _data.csr.CSR((data, cols, rows), shape=(N**2, N))

    R = expand @ A.transpose() @ expand.transpose()

    # S[i+N*j, j+N*i] = -1/2 * sum_k(A[i, k] + A[j, k])
    ket_1 = _data.Dense(np.ones(N, dtype=complex))
    Asum = A @ ket_1
    to_super = _data.add(_data.kron(Asum, ket_1), _data.kron(ket_1, Asum))
    S = _data.diag(to_super.to_array().flatten() * -0.5, 0)

    return _data.add(R, S)


def floquet_tensor(
    H: QobjEvo | FloquetBasis,
    c_ops: list[Qobj],
    spectra_cb: list[Callable[[float], complex]],
    T: float = 0,
    w_th: float = 0.0,
    kmax: int = 5,
    nT: int = 100,
) -> Qobj:
    """
    Construct a tensor that represents the master equation in the floquet
    basis.

    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]

    Parameters
    ----------
    H : :obj:`.QobjEvo`, :obj:`.FloquetBasis`
        Periodic Hamiltonian a floquet basis system.

    c_ops : list of :class:`.Qobj`
        list of collapse operators.

    spectra_cb : list callback functions
        List of callback functions that compute the noise power spectrum as
        a function of frequency for the collapse operators in `c_ops`.

    T : float, optional
        The period of the time-dependence of the hamiltonian. Optional if ``H``
        is a ``FloquetBasis`` object.

    w_th : float, default: 0.0
        The temperature in units of frequency.

    kmax : int, default: 5
        The truncation of the number of sidebands (default 5).

    nT : int, default: 100
        The number of integration steps (for calculating X) within one period.

    Returns
    -------
    output : array
        The Floquet-Markov master equation tensor `R`.
    """
    if isinstance(H, FloquetBasis):
        floquet_basis = H
        T = H.T
    else:
        floquet_basis = FloquetBasis(H, T)
    energy = floquet_basis.e_quasi
    delta = _floquet_delta_tensor(energy, kmax, T)
    x = _floquet_X_matrices(floquet_basis, c_ops, kmax, nT)
    gamma = _floquet_gamma_matrices(x, delta, spectra_cb)
    a = _floquet_A_matrix(delta, gamma, w_th)
    r = _floquet_master_equation_tensor(a)
    dims = floquet_basis.U(0).dims
    return Qobj(r, dims=[dims, dims], superrep="super", copy=False)


def fsesolve(
    H: QobjEvoLike | FloquetBasis,
    psi0: Qobj,
    tlist: ArrayLike,
    T: float = 0.0,
    *pos_args,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
) -> Result:
    """
    Solve the Schrodinger equation using the Floquet formalism.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        Periodic system Hamiltonian as :obj:`.QobjEvo`. List of
        [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    psi0 : :class:`.Qobj`
        Initial state vector (ket). If an operator is provided,

    tlist : *list* / *array*
        List of times for :math:`t`.

    T : float, default=tlist[-1]
        The period of the time-dependence of the hamiltonian.

    e_ops : list or dict of :class:`.Qobj` / callback function, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.
        See :func:`~qutip.core.expect.expect` for more detail of operator
        expectation.

    args : dictionary, optional
        Dictionary with variables required to evaluate H.

    options : dict, optional
        Options for the results.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors. Only normalize
            the state if the initial state is already normalized.

    Returns
    -------
    output : :class:`.Result`
        An instance of the class :class:`.Result`, which
        contains either an *array* of expectation values or an array of
        state vectors, for the times specified by `tlist`.
    """
    if pos_args or not isinstance(T, (int, float)):
        # Old signature used
        warnings.warn(
            f"e_ops, args and options will be keyword only"
            " for all solvers from qutip 5.3",
            FutureWarning
        )
        # Re order for previous signature
        e_ops = T
        T = 0.0
        if len(pos_args) >= 1:
            T = pos_args[0]
        if len(pos_args) >= 2:
            args = pos_args[1]
        if len(pos_args) >= 3:
            options = pos_args[2]

    if isinstance(H, FloquetBasis):
        floquet_basis = H
    else:
        T = T or tlist[-1]
        # `fsesolve` is a fallback from `fmmesolve`, for the later, options
        # are for the open system evolution.
        H = QobjEvo(H, args=args, tlist=tlist, copy=False)
        floquet_basis = FloquetBasis(H, T, precompute=tlist)

    f_coeff = floquet_basis.to_floquet_basis(psi0)
    result_options = {
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
    }
    result_options.update(options or {})
    result = Result(e_ops, result_options, solver="fsesolve")
    for t in tlist:
        state_t = floquet_basis.from_floquet_basis(f_coeff, t)
        result.add(t, state_t)

    return result


def fmmesolve(
    H: QobjEvoLike | FloquetBasis,
    rho0: Qobj,
    tlist: ArrayLike,
    c_ops: list[Qobj] = None,
    spectra_cb: list[Callable[[float], complex]] = None,
    T: float = 0.0,
    w_th: float = 0.0,
    *pos_args,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
 ) -> "FloquetResult":
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        Periodic system Hamiltonian as :obj:`.QobjEvo`. List of
        [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    rho0 / psi0 : :class:`.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    c_ops : list of :class:`.Qobj`, optional
        List of collapse operators. Time dependent collapse operators are not
        supported. Fall back on :func:`fsesolve` if not provided.

    e_ops : list of :class:`.Qobj` / callback function, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.
        See :func:`~qutip.core.expect.expect` for more detail of operator
        expectation.
        The states are reverted to the lab basis before computing the
        expectation values.

    spectra_cb : list callback functions, default: ``lambda w: (w > 0)``
        List of callback functions that compute the noise power spectrum as
        a function of frequency for the collapse operators in `c_ops`.

    T : float, default=tlist[-1]
        The period of the time-dependence of the hamiltonian. The default value
        ``0`` indicates that the 'tlist' spans a single period of the driving.

    w_th : float, default: 0.0
        The temperature of the environment in units of frequency.
        For example, if the Hamiltonian written in units of 2pi GHz, and the
        temperature is given in K, use the following conversion:

            temperature = 25e-3 # unit K
            h = 6.626e-34
            kB = 1.38e-23
            args['w_th'] = temperature * (kB / h) * 2 * pi * 1e-9

    args : dict, optional
        Dictionary of parameters for time-dependent Hamiltonian

    options : dict, optional
        Dictionary of options for the solver.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | store_floquet_states : bool
          | Whether or not to store the density matrices in the floquet basis
            in ``result.floquet_states``.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors. Only normalize
            the state if the initial state is already normalized.
        - | progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          | How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.
        - | progress_kwargs : dict
          | kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - | method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          | Which differential equation integration method to use.
        - | atol, rtol : float
          | Absolute and relative tolerance of the ODE integrator.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
            ``tlist`` step.
        - | max_step : float
          | Maximum lenght of one internal step. When using pulses, it should
            be less than half the width of the thinnest pulse.

        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    result: :class:`.Result`

        An instance of the class :class:`.Result`, which contains
        the expectation values for the times specified by ``tlist``, and/or the
        state density matrices corresponding to the times.
    """
    if c_ops is None and rho0.isket:
        return fsesolve(
            H,
            rho0,
            tlist,
            e_ops=e_ops,
            T=T,
            args=args,
            options=options,
        )
    if pos_args:
        # Old signature used
        warnings.warn(
            "e_ops, args and options will be keyword only"
            " for all solver from qutip 5.3",
            FutureWarning
        )
        e_ops = spectra_cb
        spectra_cb = T
        T = w_th
        w_th = pos_args[0]
    elif e_ops or spectra_cb is None:
        # New signature or extra param not used
        pass
    elif isinstance(T, list):
        # After the `c_ops` we could have the e_ops (old) or spectra_cb (new)
        # If old signature is used, the spectra_cb will overflow to `T`
        warnings.warn(
            "e_ops will be keyword only from qutip 5.3 for all solver",
            FutureWarning
        )
        e_ops = spectra_cb
        spectra_cb = T
        if w_th:
            T = w_th

    if isinstance(H, FloquetBasis):
        floquet_basis = H
    else:
        T = T or tlist[-1]
        t_precompute = np.concatenate([tlist, np.linspace(0, T, 101)])
        # `fsesolve` is a fallback from `fmmesolve`, for the later, options
        # are for the open system evolution.
        H = QobjEvo(H, args=args, tlist=tlist, copy=False)
        floquet_basis = FloquetBasis(H, T, precompute=t_precompute)

    if not w_th and args:
        w_th = args.get("w_th", 0.0)

    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if spectra_cb is None:
        spectra_cb = [lambda w: (w > 0)]
    elif callable(spectra_cb):
        spectra_cb = [spectra_cb]
    if len(spectra_cb) == 1:
        spectra_cb = spectra_cb * len(c_ops)

    a_ops = list(zip(c_ops, spectra_cb))

    solver = FMESolver(floquet_basis, a_ops, w_th=w_th, options=options)
    return solver.run(rho0, tlist, e_ops=e_ops)


class FloquetResult(Result):
    def _post_init(self, floquet_basis):
        self.floquet_basis = floquet_basis
        if self.options["store_floquet_states"]:
            self.floquet_states = []
        else:
            self.floquet_states = None
        super()._post_init()

    def add(self, t, state):
        if self.options["store_floquet_states"]:
            self.floquet_states.append(state)

        state = self.floquet_basis.from_floquet_basis(state, t)
        super().add(t, state)


class FMESolver(MESolver):
    """
    Solver for the Floquet-Markov master equation.

    .. note ::
        Operators (``c_ops`` and ``e_ops``) are in the laboratory basis.

    Parameters
    ----------
    floquet_basis : :class:`.FloquetBasis`
        The system Hamiltonian wrapped in a FloquetBasis object. Choosing a
        different integrator for the ``floquet_basis`` than for the evolution
        of the floquet state can improve the performance.

    a_ops : list of tuple(:class:`.Qobj`, callable)
        List of collapse operators and the corresponding function for the noise
        power spectrum. The collapse operator must be a :obj:`.Qobj` and
        cannot be time dependent. The spectrum function must take and return
        an numpy array.

    w_th : float
        The temperature of the environment in units of Hamiltonian frequency.

    kmax : int [5]
        The truncation of the number of sidebands..

    nT : int [20*kmax]
        The number of integration steps (for calculating X) within one period.

    options : dict, optional
        Options for the solver, see :obj:`FMESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.
    """

    name = "fmmesolve"
    _avail_integrators = {}
    _resultclass = FloquetResult
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        "method": "adams",
        "store_floquet_states": False,
    }

    def __init__(
        self,
        floquet_basis: FloquetBasis,
        a_ops: list[tuple[Qobj, Callable[[float], float]]],
        w_th: float = 0.0,
        *,
        kmax: int = 5,
        nT: int = None,
        options: dict[str, Any] = None,
    ):
        self.options = options
        if isinstance(floquet_basis, FloquetBasis):
            self.floquet_basis = floquet_basis
        else:
            raise TypeError("The ``floquet_basis`` must be a FloquetBasis")

        nT = nT or max(100, 20 * kmax)
        self._num_collapse = len(a_ops)
        c_ops, spectra_cb = zip(*a_ops)
        if not all(
            isinstance(c_op, Qobj) and callable(spectrum)
            for c_op, spectrum in a_ops
        ):
            raise TypeError("a_ops must be tuple of (Qobj, callable)")
        self.rhs = QobjEvo(
            floquet_tensor(
                self.floquet_basis,
                c_ops,
                spectra_cb,
                w_th=w_th,
                kmax=kmax,
                nT=nT,
            )
        )

        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()

    def _initialize_stats(self):
        stats = Solver._initialize_stats(self)
        stats.update(
            {
                "solver": "Floquet-Markov master equation",
                "num_collapse": self._num_collapse,
            }
        )
        return stats

    def _argument(self, args):
        if args:
            raise ValueError("FMESolver cannot update arguments")

    def start(self, state0: Qobj, t0: float, *, floquet: bool = False) -> None:
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state0 : :obj:`.Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        floquet : bool, optional {False}
            Whether the initial state is in the floquet basis or laboratory
            basis.
        """
        if not floquet:
            state0 = self.floquet_basis.to_floquet_basis(state0, t0)
        super().start(state0, t0)

    def step(
        self,
        t: float,
        *,
        args: dict[str, Any] = None,
        copy: bool = True,
        floquet: bool = False,
    ) -> Qobj:
        """
        Evolve the state to ``t`` and return the state as a :obj:`.Qobj`.

        Parameters
        ----------
        t : double
            Time to evolve to, must be higher than the last call.

        copy : bool, optional {True}
            Whether to return a copy of the data or the data in the ODE solver.

        floquet : bool, optional {False}
            Whether to return the state in the floquet basis or laboratory
            basis.

        args : dict, optional {None}
            Not supported

        Notes
        -----
        The state must be initialized first by calling ``start`` or
        ``run``. If ``run`` is called, ``step`` will continue from the last
        time and state obtained.
        """
        if args:
            raise ValueError("FMESolver cannot update arguments")
        state = super().step(t)
        if not floquet:
            state = self.floquet_basis.from_floquet_basis(state, t)
        elif copy:
            state = state.copy()
        return state

    def run(
        self,
        state0: Qobj,
        tlist: ArrayLike,
        *,
        floquet: bool = False,
        args: dict[str, Any] = None,
        e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    ) -> FloquetResult:
        """
        Calculate the evolution of the quantum system.

        For a ``state0`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :class:`.Result`. The evolution method and
        stored results are determined by ``options``.

        Parameters
        ----------
        state0 : :obj:`.Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Each times of the list must be increasing, but does not
            need to be uniformy distributed.

        floquet : bool, optional {False}
            Whether the initial state in the floquet basis or laboratory basis.

        args : dict, optional
            Not supported

        e_ops : list or dict, optional
            List or dict of Qobj, QobjEvo or callable to compute the
            expectation values. Function[s] must have the signature
            ``f(t : float, state : Qobj) -> expect``.

        Returns
        -------
        results : :class:`FloquetResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        if args:
            raise ValueError("FMESolver cannot update arguments")
        if not floquet:
            state0 = self.floquet_basis.to_floquet_basis(state0, tlist[0])
        _time_start = time()
        _data0 = self._prepare_state(state0)
        self._integrator.set_state(tlist[0], _data0)
        stats = self._initialize_stats()
        results = self._resultclass(
            e_ops,
            self.options,
            solver=self.name,
            stats=stats,
            floquet_basis=self.floquet_basis,
        )
        results.add(tlist[0], self._restore_state(_data0, copy=False))
        stats["preparation time"] += time() - _time_start

        progress_bar = progress_bars[self.options["progress_bar"]](
            len(tlist) - 1, **self.options["progress_kwargs"]
        )
        for t, state in self._integrator.run(tlist):
            progress_bar.update()
            results.add(t, self._restore_state(state, copy=False))
        progress_bar.finished()

        stats["run time"] = progress_bar.total_time()
        # TODO: It would be nice if integrator could give evolution statistics
        # stats.update(_integrator.stats)
        return results

    @classmethod
    def ExpectFeedback(cls):
        """
        Expect of the state of the evolution to be used in a time-dependent
        operator.

        Not not implemented for FMESolver
        """
        raise NotImplementedError(
            "Feedback not implemented for floquet solver."
        )

    @classmethod
    def StateFeedback(cls):
        """
        State of the evolution to be used in a time-dependent operator.

        Not not implemented for FMESolver
        """
        raise NotImplementedError(
            "Feedback not implemented for floquet solver."
        )
