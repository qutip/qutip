# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ["smesolve", "SMESolver", "ssesolve", "SSESolver"]

import numpy as np
from numpy.typing import ArrayLike
from numpy.random import SeedSequence
from typing import Any, Callable, Literal, overload
from functools import partial
from time import time
from collections.abc import Sequence
from .multitrajresult import MultiTrajResult
from .sode.ssystem import StochasticOpenSystem, StochasticClosedSystem
from .sode._noise import PreSetWiener
from .result import Result, ExpectOp
from .multitraj import _MultiTrajRHS, MultiTrajSolver
from .. import Qobj, QobjEvo
from ..core.dimensions import Dimensions
from ..core import data as _data
from .solver_base import _solver_deprecation
from ._feedback import _QobjFeedback, _DataFeedback, _WienerFeedback
from ..typing import QobjEvoLike, EopsLike


class StochasticTrajResult(Result):
    def _post_init(self, m_ops=(), dw_factor=(), heterodyne=False):
        super()._post_init()
        self.noise = []
        self.heterodyne = heterodyne
        if self.options["store_measurement"]:
            self.m_ops = []
            self.m_expect = []
            self.dW_factor = dw_factor
            for op in m_ops:
                f = self._e_op_func(op)
                self.m_expect.append([])
                self.m_ops.append(ExpectOp(op, f, self.m_expect[-1].append))
                self.add_processor(self.m_ops[-1]._store)

    def add(self, t, state, noise=None):
        super().add(t, state)
        if noise is not None:
            self.noise.append(noise)

    @property
    def wiener_process(self) -> np.typing.NDArray[float]:
        """
        Wiener processes for each stochastic collapse operators.

        The output shape is
            (len(sc_ops), len(tlist))
        for homodyne detection, and
            (len(sc_ops), 2, len(tlist))
        for heterodyne detection.
        """
        W = np.zeros(
            (self.noise[0].shape[0], len(self.times)),
            dtype=np.float64
        )
        np.cumsum(np.array(self.noise).T, axis=1, out=W[:, 1:])
        if self.heterodyne:
            W = W.reshape(-1, 2, W.shape[1])
        return W

    @property
    def dW(self) -> np.typing.NDArray[float]:
        """
        Wiener increment for each stochastic collapse operators.

        The output shape is
            (len(sc_ops), len(tlist)-1)
        for homodyne detection, and
            (len(sc_ops), 2, len(tlist)-1)
        for heterodyne detection.
        """
        noise = np.array(self.noise).T
        if self.heterodyne:
            return noise.reshape(-1, 2, noise.shape[1])
        return noise

    @property
    def measurement(self) -> np.typing.NDArray[float]:
        """
        Measurements for each stochastic collapse operators.

        The output shape is
            (len(sc_ops), len(tlist)-1)
        for homodyne detection, and
            (len(sc_ops), 2, len(tlist)-1)
        for heterodyne detection.
        """
        if not self.options["store_measurement"]:
            return None
        elif len(self.m_ops) == 0:
            if self.heterodyne:
                return np.empty(shape=(0, 2, len(self.times) - 1))
            else:
                return np.empty(shape=(0, len(self.times) - 1))
        elif self.options["store_measurement"] == "start":
            m_expect = np.array(self.m_expect)[:, :-1]
        elif self.options["store_measurement"] == "middle":
            m_expect = np.apply_along_axis(
                lambda m: np.convolve(m, [0.5, 0.5], "valid"),
                axis=1, arr=self.m_expect,
            )
        elif self.options["store_measurement"] in ["end", True]:
            m_expect = np.array(self.m_expect)[:, 1:]
        else:
            raise ValueError(
                "store_measurement must be in {'start', 'middle', 'end', ''}, "
                f"not {self.options['store_measurement']}"
            )
        noise = np.array(self.noise).T
        noise_scaled = np.einsum(
            "i,ij,j->ij", self.dW_factor, noise, (1 / np.diff(self.times))
        )
        if self.heterodyne:
            m_expect = m_expect.reshape(-1, 2, m_expect.shape[1])
            noise_scaled = noise_scaled.reshape(-1, 2, noise_scaled.shape[1])
        return m_expect + noise_scaled


class StochasticResult(MultiTrajResult):
    def _post_init(self, heterodyne=False):
        super()._post_init()
        self.heterodyne = heterodyne

        store_measurement = self.options["store_measurement"]
        keep_runs = self.options["keep_runs_results"]

        if not keep_runs and store_measurement:
            self.add_processor(
                partial(self._reduce_attr, attr="wiener_process")
            )
            self._wiener_process = []
            self.add_processor(partial(self._reduce_attr, attr="dW"))
            self._dW = []

        if not keep_runs and store_measurement:
            self.add_processor(partial(self._reduce_attr, attr="measurement"))
            self._measurement = []

    def _reduce_attr(self, trajectory, attr, *, rel=None, abs=None):
        """
        Add a result attribute to a list when the trajectories are not stored.
        """
        if abs is None:
            getattr(self, "_" + attr).append(getattr(trajectory, attr))

    def _trajectories_attr(self, attr):
        """
        Get the result associated to the attr, whether the trajectories are
        saved or not.
        """
        if hasattr(self, "_" + attr):
            return getattr(self, "_" + attr)
        elif self.options["keep_runs_results"]:
            return np.array([
                getattr(traj, attr) for traj in self.trajectories
            ])
        return None

    @property
    def measurement(self) -> np.typing.NDArray[float]:
        """
        Measurements for each trajectories and stochastic collapse operators.

        The output shape is
            (ntraj, len(sc_ops), len(tlist)-1)
        for homodyne detection, and
            (ntraj, len(sc_ops), 2, len(tlist)-1)
        for heterodyne detection.
        """
        return self._trajectories_attr("measurement")

    @property
    def dW(self) -> np.typing.NDArray[float]:
        """
        Wiener increment for each trajectories and stochastic collapse
        operators.

        The output shape is
            (ntraj, len(sc_ops), len(tlist)-1)
        for homodyne detection, and
            (ntraj, len(sc_ops), 2, len(tlist)-1)
        for heterodyne detection.
        """
        return self._trajectories_attr("dW")

    @property
    def wiener_process(self) -> np.typing.NDArray[float]:
        """
        Wiener processes for each trajectories and stochastic collapse
        operators.

        The output shape is
            (ntraj, len(sc_ops), len(tlist)-1)
        for homodyne detection, and
            (ntraj, len(sc_ops), 2, len(tlist)-1)
        for heterodyne detection.
        """
        return self._trajectories_attr("wiener_process")

    def merge(self, other: "StochasticResult", p: float = None) -> "StochasticResult":
        if not isinstance(other, StochasticResult):
            return NotImplemented
        if self.stats["solver"] != other.stats["solver"]:
            raise ValueError("Can't merge smesolve and ssesolve results")
        if self.heterodyne != other.heterodyne:
            raise ValueError("Can't merge heterodyne and homodyne results")
        if p is not None:
            raise ValueError(
                "Stochastic solvers does not support custom weights"
            )
        new = super().merge(other, p)

        if (
            self.options["store_measurement"]
            and other.options["store_measurement"]
            and not new.trajectories
        ):
            new._measurement = np.concatenate(
                (self.measurement, other.measurement), axis=0
            )
            new._wiener_process = np.concatenate(
                (self.wiener_process, other.wiener_process), axis=0
            )
            new._dW = np.concatenate((self.dW, other.dW), axis=0)

        return new


class _StochasticRHS(_MultiTrajRHS):
    """
    In between object to store the stochastic system.

    It store the Hamiltonian (not Liouvillian when possible), and sc_ops.
    dims and flags are provided to be usable the the base ``Solver`` class.

    We don't want to use the cython rhs (``StochasticOpenSystem``, etc.) since
    the rouchon integrator need the part but does not use the usual drift and
    diffusion computation.
    """

    def __init__(self, issuper, H, sc_ops, c_ops, heterodyne):

        if not isinstance(H, (Qobj, QobjEvo)) or not H.isoper:
            raise TypeError("The Hamiltonian must be an operator")
        self.H = QobjEvo(H)

        if isinstance(sc_ops, (Qobj, QobjEvo)):
            sc_ops = [sc_ops]
        self.sc_ops = [QobjEvo(c_op) for c_op in sc_ops]

        if isinstance(c_ops, (Qobj, QobjEvo)):
            c_ops = [c_ops]
        self.c_ops = [QobjEvo(c_op) for c_op in c_ops]

        if any(not c_op.isoper for c_op in c_ops):
            raise TypeError("c_ops must be operators")

        if any(not c_op.isoper for c_op in sc_ops):
            raise TypeError("sc_ops must be operators")

        self.issuper = issuper
        self.heterodyne = heterodyne
        self._noise_key = None

        if heterodyne:
            sc_ops = []
            for c_op in self.sc_ops:
                sc_ops.append(c_op / np.sqrt(2))
                sc_ops.append(c_op * (-1j / np.sqrt(2)))
            self.sc_ops = sc_ops

        if self.issuper and not self.H.issuper:
            self.dims = [self.H.dims, self.H.dims]
            self._dims = Dimensions([self.H._dims, self.H._dims])
        else:
            self.dims = self.H.dims
            self._dims = self.H._dims

    def __call__(self, options):
        if self.issuper:
            return StochasticOpenSystem(
                self.H, self.sc_ops, self.c_ops, options.get("derr_dt", 1e-6)
            )
        else:
            return StochasticClosedSystem(self.H, self.sc_ops)

    def arguments(self, args):
        self.H.arguments(args)
        for c_op in self.c_ops:
            c_op.arguments(args)
        for sc_op in self.sc_ops:
            sc_op.arguments(args)

    def _register_feedback(self, val):
        self.H._register_feedback({"wiener_process": val}, "stochastic solver")
        for c_op in self.c_ops:
            c_op._register_feedback(
                {"WienerFeedback": val}, "stochastic solver"
            )
        for sc_op in self.sc_ops:
            sc_op._register_feedback(
                {"WienerFeedback": val}, "stochastic solver"
            )


def smesolve(
    H: QobjEvoLike,
    rho0: Qobj,
    tlist: ArrayLike,
    c_ops: Qobj | QobjEvo | Sequence[QobjEvoLike] = (),
    sc_ops: Qobj | QobjEvo | Sequence[QobjEvoLike] = (),
    heterodyne: bool = False,
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    ntraj: int = 500,
    options: dict[str, Any] = None,
    seeds: int | SeedSequence | Sequence[int | SeedSequence] = None,
    target_tol: float | tuple[float, float] | list[tuple[float, float]] = None,
    timeout: float = None,
    **kwargs
) -> StochasticResult:
    """
    Solve stochastic master equation.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    rho0 : :class:`.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    c_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format), optional
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        List of stochastic collapse operators.

    e_ops : :obj:`.Qobj`, callable, list or dict, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.

    args : dict, optional
        Dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    ntraj : int, default: 500
        Number of trajectories to compute.

    heterodyne : bool, default: False
        Whether to use heterodyne or homodyne detection.

    seeds : int, SeedSequence, list, optional
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seeds, one for each
        trajectory. Seeds are saved in the result and they can be reused with::

            seeds=prev_result.seeds

        When using a parallel map, the trajectories can be re-ordered.

    target_tol : {float, tuple, list}, optional
        Target tolerance of the evolution. The evolution will compute
        trajectories until the error on the expectation values is lower than
        this tolerance. The maximum number of trajectories employed is
        given by ``ntraj``. The error is computed using jackknife resampling.
        ``target_tol`` can be an absolute tolerance or a pair of absolute and
        relative tolerance, in that order. Lastly, it can be a list of pairs of
        ``(atol, rtol)`` for each e_ops.

    timeout : float, optional
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed. Overwrite the option of the same name.

    options : dict, optional
        Dictionary of options for the solver.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | store_measurement: str, {'start', 'middle', 'end', ''}
          | Whether and how to store the measurement for each trajectories.
            'start', 'middle', 'end' indicate when in the interval the
            expectation value of the ``m_ops`` is taken.
        - | keep_runs_results : bool
          | Whether to store results from all trajectories or just store the
            averages.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors. Only normalize
            the state if the initial state is already normalized.
        - | progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          | How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.
        - | progress_kwargs : dict
          | kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - | method : str
          | Which stochastic differential equation integration method to use.
            Main ones are {"euler", "rouchon", "platen", "taylor1.5_imp"}
        - | map : str {"serial", "parallel", "loky", "mpi"}
          | How to run the trajectories. "parallel" uses the multiprocessing
            module to run in parallel while "loky" and "mpi" use the "loky" and
            "mpi4py" modules to do so.
        - | num_cpus : NoneType, int
          | Number of cpus to use when running in parallel. ``None`` detect the
            number of available cpus.
        - | dt : float
          | The finite steps lenght for the Stochastic integration method.
            Default change depending on the integrator.

        Additional options are listed under
        `options <./classes.html#qutip.solver.stochastic.SMESolver.options>`__.
        More options may be available depending on the selected
        differential equation integration method, see
        `SIntegrator <./classes.html#classes-sode>`_.

    Returns
    -------

    output: :class:`.Result`
        An instance of the class :class:`.Result`.
    """
    options = _solver_deprecation(kwargs, options, "stoc")
    H = QobjEvo(H, args=args, tlist=tlist)
    if not isinstance(sc_ops, Sequence):
        sc_ops = [sc_ops]
    if not isinstance(c_ops, Sequence):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]
    sc_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in sc_ops]
    sol = SMESolver(
        H, sc_ops, c_ops=c_ops, options=options, heterodyne=heterodyne
    )
    return sol.run(
        rho0, tlist, ntraj, e_ops=e_ops,
        seeds=seeds, target_tol=target_tol, timeout=timeout,
    )


def ssesolve(
    H: QobjEvoLike,
    psi0: Qobj,
    tlist: ArrayLike,
    sc_ops: QobjEvoLike | Sequence[QobjEvoLike] = (),
    heterodyne: bool = False,
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    args: dict[str, Any] = None,
    ntraj: int = 500,
    options: dict[str, Any] = None,
    seeds: int | SeedSequence | Sequence[int | SeedSequence] = None,
    target_tol: float | tuple[float, float] | list[tuple[float, float]] = None,
    timeout: float = None,
    **kwargs
) -> StochasticResult:
    """
    Solve stochastic Schrodinger equation.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    psi0 : :class:`.Qobj`
        Initial state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    sc_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        List of stochastic collapse operators.

    e_ops : :obj:`.Qobj`, callable, list or dict, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.

    args : dict, optional
        Dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    ntraj : int, default: 500
        Number of trajectories to compute.

    heterodyne : bool, default: False
        Whether to use heterodyne or homodyne detection.

    seeds : int, SeedSequence, list, optional
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seeds, one for each
        trajectory. Seeds are saved in the result and they can be reused with::

            seeds=prev_result.seeds

    target_tol : {float, tuple, list}, optional
        Target tolerance of the evolution. The evolution will compute
        trajectories until the error on the expectation values is lower than
        this tolerance. The maximum number of trajectories employed is
        given by ``ntraj``. The error is computed using jackknife resampling.
        ``target_tol`` can be an absolute tolerance or a pair of absolute and
        relative tolerance, in that order. Lastly, it can be a list of pairs of
        (atol, rtol) for each e_ops.

    timeout : float, optional
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed. Overwrite the option of the same name.

    options : dict, optional
        Dictionary of options for the solver.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | store_measurement: str, {'start', 'middle', 'end', ''}
          | Whether and how to store the measurement for each trajectories.
            'start', 'middle', 'end' indicate when in the interval the
            expectation value of the ``m_ops`` is taken.
        - | keep_runs_results : bool
          | Whether to store results from all trajectories or just store the
            averages.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors. Only normalize
            the state if the initial state is already normalized.
        - | progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          | How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.
        - | progress_kwargs : dict
          | kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - | method : str
          | Which stochastic differential equation integration method to use.
            Main ones are {"euler", "rouchon", "platen", "taylor1.5_imp"}
        - | map : str {"serial", "parallel", "loky", "mpi"}
          | How to run the trajectories. "parallel" uses the multiprocessing
            module to run in parallel while "loky" and "mpi" use the "loky" and
            "mpi4py" modules to do so.
        - | num_cpus : NoneType, int
          | Number of cpus to use when running in parallel. ``None`` detect the
            number of available cpus.
        - | dt : float
          | The finite steps lenght for the Stochastic integration method.
            Default change depending on the integrator.

        Additional options are listed under
        `options <./classes.html#qutip.solver.stochastic.SSESolver.options>`__.
        More options may be available depending on the selected
        differential equation integration method, see
        `SIntegrator <./classes.html#classes-sode>`_.

    Returns
    -------

    output: :class:`.Result`
        An instance of the class :class:`.Result`.
    """
    options = _solver_deprecation(kwargs, options, "stoc")
    H = QobjEvo(H, args=args, tlist=tlist)
    if not isinstance(sc_ops, Sequence):
        sc_ops = [sc_ops]
    sc_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in sc_ops]
    sol = SSESolver(H, sc_ops, options=options, heterodyne=heterodyne)
    return sol.run(
        psi0, tlist, ntraj, e_ops=e_ops,
        seeds=seeds, target_tol=target_tol, timeout=timeout,
    )


class StochasticSolver(MultiTrajSolver):
    """
    Generic stochastic solver.
    """

    name = "StochasticSolver"
    _avail_integrators = {}
    _open = None

    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "map": "serial",
        "mpi_options": {},
        "num_cpus": None,
        "bitgenerator": None,
        "method": "platen",
        "store_measurement": "",
    }

    def _resultclass(self, e_ops, options, solver, stats):
        return StochasticResult(
            e_ops,
            options,
            solver=solver,
            stats=stats,
            heterodyne=self.heterodyne,
        )

    def _trajectory_resultclass(self, e_ops, options):
        return StochasticTrajResult(
            e_ops,
            options,
            m_ops=self.m_ops,
            dw_factor=self.dW_factors,
            heterodyne=self.heterodyne,
        )

    def _initialize_stats(self):
        stats = super()._initialize_stats()
        if self._open:
            stats["solver"] = "Stochastic Master Equation Evolution"
        else:
            stats["solver"] = "Stochastic Schrodinger Equation Evolution"
        return stats

    def __init__(
        self,
        H: Qobj | QobjEvo,
        sc_ops: Sequence[Qobj | QobjEvo],
        heterodyne: bool,
        *,
        c_ops: Sequence[Qobj | QobjEvo] = (),
        options: dict[str, Any] = None,
    ):
        self._heterodyne = heterodyne
        if self.name == "ssesolve" and c_ops:
            raise ValueError("c_ops are not supported by ssesolve.")

        rhs = _StochasticRHS(self._open, H, sc_ops, c_ops, heterodyne)
        super().__init__(rhs, options=options)

        if heterodyne:
            self._m_ops = []
            for op in sc_ops:
                self._m_ops += [op + op.dag(), -1j * (op - op.dag())]
            self._dW_factors = np.ones(len(sc_ops) * 2) * 2**0.5
        else:
            self._m_ops = [op + op.dag() for op in sc_ops]
            self._dW_factors = np.ones(len(sc_ops))

    @property
    def heterodyne(self) -> bool:
        return self._heterodyne

    @property
    def m_ops(self) -> list[QobjEvo | Qobj]:
        return self._m_ops

    @m_ops.setter
    def m_ops(self, new_m_ops: list[QobjEvo | Qobj]):
        """
        Measurements operators.

        Default are:

            m_ops = sc_ops + sc_ops.dag()

        for homodyne detection, and

            m_ops = sc_ops + sc_ops.dag(), -1j*(sc_ops - sc_ops.dag())

        for heterodyne detection.

        Measurements opput is computed as:

            expect(m_ops_i, state(t)) + dW_i / dt * dW_factors

        Where ``dW`` follows a gaussian distribution with norm 0 and derivation
        of ``dt**0.5``. ``dt`` is the time difference between step in the
        ``tlist``.

        ``m_ops`` can be overwritten, but the number of operators must be
        constant.
        """
        if len(new_m_ops) != len(self.m_ops):
            if self.heterodyne:
                raise ValueError(
                    f"2 `m_ops` per `sc_ops`, {len(self.rhs.sc_ops)} operators"
                    " are expected for heterodyne measurement."
                )
            else:
                raise ValueError(
                    f"{len(self.rhs.sc_ops)} measurements "
                    "operators are expected."
                )
        if not all(
            isinstance(op, Qobj) and op.dims == self.rhs.sc_ops[0].dims
            for op in new_m_ops
        ):
            raise ValueError(
                "m_ops must be Qobj with the same dimensions"
                " as the Hamiltonian"
            )
        self._m_ops = new_m_ops

    @property
    def dW_factors(self) -> np.typing.NDArray[float]:
        return self._dW_factors

    @dW_factors.setter
    def dW_factors(self, new_dW_factors: np.typing.NDArray[float]):
        """
        Scaling of the noise on the measurements.
        Default are ``1`` for homodyne and ``sqrt(1/2)`` for heterodyne.
        ``dW_factors`` must be a list of the same length as ``m_ops``.
        """
        if len(new_dW_factors) != len(self._dW_factors):
            if self.heterodyne:
                raise ValueError(
                    f"2 `dW_factors` per `sc_ops`, {len(self.rhs.sc_ops)} "
                    "values are expected for heterodyne measurement."
                )
            else:
                raise ValueError(
                    f"{len(self.rhs.sc_ops)} dW_factors are expected."
                )
        self._dW_factors = new_dW_factors

    def _integrate_one_traj(self, seed, tlist, result):
        for t, state, noise in self._integrator.run(tlist):
            result.add(t, self._restore_state(state, copy=False), noise)
        return seed, result

    def run_from_experiment(
        self,
        state: Qobj,
        tlist: ArrayLike,
        noise: Sequence[float],
        *,
        args: dict[str, Any] = None,
        e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
        measurement: bool = False,
    ):
        """
        Run a single trajectory from a given state and noise.

        Parameters
        ----------
        state : Qobj
            Initial state of the system.

        tlist : array_like
            List of times for which to evaluate the state. The tlist must
            increase uniformly.

        noise : array_like
            Noise for each time step and each stochastic collapse operators.
            For homodyne detection, ``noise[i, t_idx]`` is the Wiener
            increments between ``tlist[t_idx]`` and ``tlist[t_idx+1]`` for the
            i-th sc_ops.
            For heterodyne detection, an extra dimension is added for the pair
            of measurement: ``noise[i, j, t_idx]``with ``j`` in ``{0,1}``.

        args : dict, optional
            Arguments to pass to the Hamiltonian and collapse operators.

        e_ops : :obj:`.Qobj`, callable, list or dict, optional
            Single operator, or list or dict of operators, for which to
            evaluate expectation values. Operator can be Qobj, QobjEvo or
            callables with the signature `f(t: float, state: Qobj) -> Any`.

        measurement : bool, default : False
            Whether the passed noise is the Wiener increments ``dW`` (gaussian
            noise with standard derivation of dt**0.5), or the measurement.

            Homodyne measurement is::

              noise[i][t] = dW/dt + expect(sc_ops[i] + sc_ops[i].dag, state[t])

            Heterodyne measurement is::

              noise[i][0][t] = dW/dt * 2**0.5
                + expect(sc_ops[i] + sc_ops[i].dag, state[t])

              noise[i][1][t] = dW/dt * 2**0.5
                -1j * expect(sc_ops[i] - sc_ops[i].dag, state[t])

            Note that this function expects the expectation values to be taken
            at the start of the time step, corresponding to the "start" setting
            for the "store_measurements" option.

            Only available for limited integration methods.

        Returns
        -------
        result : StochasticTrajResult
            Result of the trajectory.

        Notes
        -----
        Only default values of `m_ops` and `dW_factors` are supported.
        """
        start_time = time()
        self._argument(args)
        stats = self._initialize_stats()
        dt = tlist[1] - tlist[0]
        if not np.allclose(dt, np.diff(tlist)):
            raise ValueError("tlist must be evenly spaced.")
        generator = PreSetWiener(
            noise, tlist, len(self.rhs.sc_ops), self.heterodyne, measurement
        )
        state0 = self._prepare_state(state)
        try:
            old_dt = None
            if "dt" in self._integrator.options:
                old_dt = self._integrator.options["dt"]
                self._integrator.options["dt"] = dt
            mid_time = time()
            result = self._initialize_run_one_traj(
                None, state0, tlist, e_ops, generator=generator
            )
            _, result = self._integrate_one_traj(None, tlist, result)
        except Exception as err:
            if old_dt is not None:
                self._integrator.options["dt"] = old_dt
            raise

        stats['preparation time'] += mid_time - start_time
        stats['run time'] = time() - mid_time
        result.stats.update(stats)
        return result

    @overload
    def step(
        self, t: float,
        *,
        args: dict[str, Any],
        copy: bool,
        wiener_increment: Literal[False],
    ) -> Qobj: ...

    @overload
    def step(
        self, t: float,
        *,
        args: dict[str, Any],
        copy: bool,
        wiener_increment: Literal[True],
    ) -> tuple[Qobj, np.typing.NDArray[float]]: ...

    def step(self, t, *, args=None, copy=True, wiener_increment=False):
        """
        Evolve the state to ``t`` and return the state as a :obj:`.Qobj`.

        Parameters
        ----------
        t : double
            Time to evolve to, must be higher than the last call.

        args : dict, optional
            Update the ``args`` of the system.
            The change is effective from the beginning of the interval.
            Changing ``args`` can slow the evolution.

        copy : bool, default: True
            Whether to return a copy of the data or the data in the ODE solver.

        wiener_increment: bool, default: False
            Whether to return ``dW`` in addition to the state.
        """
        if not self._integrator._is_set:
            raise RuntimeError("The `start` method must called first.")
        self._argument(args)
        _, state, dW = self._integrator.integrate(t, copy=False)
        state = self._restore_state(state, copy=copy)
        if wiener_increment:
            if self.heterodyne:
                dW = dW.reshape(-1, 2)
            return state, dW
        return state

    @classmethod
    def avail_integrators(cls):
        if cls is StochasticSolver:
            return cls._avail_integrators.copy()
        return {
            **StochasticSolver.avail_integrators(),
            **cls._avail_integrators,
        }

    @property
    def options(self) -> dict[str, Any]:
        """
        Options for stochastic solver:

        store_final_state: bool, default: False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: None, bool, default: None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        store_measurement: str, {'start', 'middle', 'end', ''}, default: ""
            Whether and how to store the measurement for each trajectories.
            'start', 'middle', 'end' indicate when in the interval the
            expectation value of the ``m_ops`` is taken.
            Storing measurements will also store the wiener process, or
            brownian noise for each trajectories.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}, default: "text"
            How to present the solver progress. 'tqdm' uses the python module
            of the same name and raise an error if not installed. Empty string
            or False will disable the bar.

        progress_kwargs: dict, default: {"chunk_size":10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        keep_runs_results: bool, default: False
          Whether to store results from all trajectories or just store the
          averages.

        normalize_output: bool
            Normalize output state to hide ODE numerical errors.

        method: str, default: "platen"
            Which differential equation integration method to use.

        map: str {"serial", "parallel", "loky", "mpi"}, default: "serial"
            How to run the trajectories. "parallel" uses the multiprocessing
            module to run in parallel while "loky" and "mpi" use the "loky" and
            "mpi4py" modules to do so.

        mpi_options: dict, default: {}
            Only applies if map is "mpi". This dictionary will be passed as
            keyword arguments to the `mpi4py.futures.MPIPoolExecutor`
            constructor. Note that the `max_workers` argument is provided
            separately through the `num_cpus` option.

        num_cpus: None, int, default: None
            Number of cpus to use when running in parallel. ``None`` detect the
            number of available cpus.

        bitgenerator: {None, "MT19937", "PCG64DXSM", ...}, default: None
            Which of numpy.random's bitgenerator to use. With ``None``, your
            numpy version's default is used.
        """
        return self._options

    @options.setter
    def options(self, new_options: dict[str, Any]):
        MultiTrajSolver.options.fset(self, new_options)

    @classmethod
    def WienerFeedback(
        cls,
        default: Callable[[float], np.typing.NDArray[float]] = None,
    ):
        """
        Wiener function of the trajectory argument for time dependent systems.

        When used as an args:

            ``QobjEvo([op, func], args={"W": SMESolver.WienerFeedback()})``

        The ``func`` will receive a function as ``W`` that return an array of
        wiener processes values at ``t``. The wiener process for the i-th
        sc_ops is the i-th element for homodyne detection and the (2i, 2i+1)
        pairs of process in heterodyne detection. The process is a step
        function with step of length ``options["dt"]``.

        .. note::

            WienerFeedback can't be added to a running solver when updating
            arguments between steps: ``solver.step(..., args={})``.

        Parameters
        ----------
        default : callable, optional
            Default function used outside the solver.
            When not passed, a function returning ``np.array([0])`` is used.

        """
        return _WienerFeedback(default)

    @classmethod
    def StateFeedback(
        cls,
        default: Qobj | _data.Data = None,
        raw_data: bool = False
    ):
        """
        State of the evolution to be used in a time-dependent operator.

        When used as an args:

            ``QobjEvo([op, func], args={"state": SMESolver.StateFeedback()})``

        The ``func`` will receive the density matrix as ``state`` during the
        evolution.

        .. note::

            Not supported by the ``rouchon`` mehtod.

        Parameters
        ----------
        default : Qobj or qutip.core.data.Data, default : None
            Initial value to be used at setup of the system.

        raw_data : bool, default : False
            If True, the raw matrix will be passed instead of a Qobj.
            For density matrices, the matrices can be column stacked or square
            depending on the integration method.

        """
        if raw_data:
            return _DataFeedback(default, open=cls._open)
        return _QobjFeedback(default, open=cls._open)


class SMESolver(StochasticSolver):
    r"""
    Stochastic Master Equation Solver.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    sc_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        List of stochastic collapse operators.

    heterodyne : bool, default: False
        Whether to use heterodyne or homodyne detection.

    options : dict, optional
        Options for the solver, see :obj:`SMESolver.options` and
        `SIntegrator <./classes.html#classes-sode>`_ for a list of all options.
    """
    name = "smesolve"
    _avail_integrators = {}
    _open = True
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "map": "serial",
        "mpi_options": {},
        "num_cpus": None,
        "bitgenerator": None,
        "method": "platen",
        "store_measurement": "",
    }


class SSESolver(StochasticSolver):
    r"""
    Stochastic Schrodinger Equation Solver.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    c_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        List of stochastic collapse operators.

    heterodyne : bool, default: False
        Whether to use heterodyne or homodyne detection.

    options : dict, optional
        Options for the solver, see :obj:`SSESolver.options` and
        `SIntegrator <./classes.html#classes-sode>`_ for a list of all options.
    """
    name = "ssesolve"
    _avail_integrators = {}
    _open = False
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "map": "serial",
        "mpi_options": {},
        "num_cpus": None,
        "bitgenerator": None,
        "method": "platen",
        "store_measurement": "",
    }
