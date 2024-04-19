__all__ = ["smesolve", "SMESolver", "ssesolve", "SSESolver"]

from .sode.ssystem import StochasticOpenSystem, StochasticClosedSystem
from .result import MultiTrajResult, Result, ExpectOp
from .multitraj import _MultiTrajRHS, MultiTrajSolver
from .. import Qobj, QobjEvo
from ..core.dimensions import Dimensions
import numpy as np
from functools import partial
from .solver_base import _solver_deprecation
from ._feedback import _QobjFeedback, _DataFeedback, _WienerFeedback


class StochasticTrajResult(Result):
    def _post_init(self, m_ops=(), dw_factor=(), heterodyne=False):
        super()._post_init()
        self.W = []
        self.m_ops = []
        self.m_expect = []
        self.dW_factor = dw_factor
        self.heterodyne = heterodyne
        for op in m_ops:
            f = self._e_op_func(op)
            self.W.append([0.0])
            self.m_expect.append([])
            self.m_ops.append(ExpectOp(op, f, self.m_expect[-1].append))
            self.add_processor(self.m_ops[-1]._store)

    def add(self, t, state, noise=None):
        super().add(t, state)
        if noise is not None and self.options["store_measurement"]:
            for i, dW in enumerate(noise):
                self.W[i].append(self.W[i][-1] + dW)

    @property
    def wiener_process(self):
        """
        Wiener processes for each stochastic collapse operators.

        The output shape is
            (len(sc_ops), len(tlist))
        for homodyne detection, and
            (len(sc_ops), 2, len(tlist))
        for heterodyne detection.
        """
        W = np.array(self.W)
        if self.heterodyne:
            W = W.reshape(-1, 2, W.shape[1])
        return W

    @property
    def dW(self):
        """
        Wiener increment for each stochastic collapse operators.

        The output shape is
            (len(sc_ops), len(tlist)-1)
        for homodyne detection, and
            (len(sc_ops), 2, len(tlist)-1)
        for heterodyne detection.
        """
        dw = np.diff(self.W, axis=1)
        if self.heterodyne:
            dw = dw.reshape(-1, 2, dw.shape[1])
        return dw

    @property
    def measurement(self):
        """
        Measurements for each stochastic collapse operators.

        The output shape is
            (len(sc_ops), len(tlist)-1)
        for homodyne detection, and
            (len(sc_ops), 2, len(tlist)-1)
        for heterodyne detection.
        """
        dts = np.diff(self.times)
        m_expect = np.array(self.m_expect)[:, 1:]
        noise = np.einsum(
            "i,ij,j->ij", self.dW_factor, np.diff(self.W, axis=1), (1 / dts)
        )
        if self.heterodyne:
            m_expect = m_expect.reshape(-1, 2, m_expect.shape[1])
            noise = noise.reshape(-1, 2, noise.shape[1])
        return m_expect + noise


class StochasticResult(MultiTrajResult):
    def _post_init(self):
        super()._post_init()

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

    def _reduce_attr(self, trajectory, attr):
        """
        Add a result attribute to a list when the trajectories are not stored.
        """
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
    def measurement(self):
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
    def dW(self):
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
    def wiener_process(self):
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
    H, rho0, tlist, c_ops=(), sc_ops=(), heterodyne=False, *,
    e_ops=(), args={}, ntraj=500, options=None,
    seeds=None, target_tol=None, timeout=None, **kwargs
):
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

    e_ops : : :class:`.qobj`, callable, or list, optional
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`.expect` for more detail of operator expectation.

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
        - | store_measurement: bool
          | Whether to store the measurement and wiener process for each
            trajectories.
        - | keep_runs_results : bool
          | Whether to store results from all trajectories or just store the
            averages.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors.
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
    H, psi0, tlist, sc_ops=(), heterodyne=False, *,
    e_ops=(), args={}, ntraj=500, options=None,
    seeds=None, target_tol=None, timeout=None, **kwargs
):
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

    e_ops : :class:`.qobj`, callable, or list, optional
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

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
        - | store_measurement: bool
            Whether to store the measurement and wiener process, or brownian
            noise for each trajectories.
        - | keep_runs_results : bool
          | Whether to store results from all trajectories or just store the
            averages.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors.
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
    _resultclass = StochasticResult
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
        "store_measurement": False,
    }

    def _trajectory_resultclass(self, e_ops, options):
        return StochasticTrajResult(
            e_ops,
            options,
            m_ops=self.m_ops,
            dw_factor=self.dW_factors,
            heterodyne=self.heterodyne,
        )

    def __init__(self, H, sc_ops, heterodyne, *, c_ops=(), options=None):
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
    def heterodyne(self):
        return self._heterodyne

    @property
    def m_ops(self):
        return self._m_ops

    @m_ops.setter
    def m_ops(self, new_m_ops):
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
    def dW_factors(self):
        return self._dW_factors

    @dW_factors.setter
    def dW_factors(self, new_dW_factors):
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

    @classmethod
    def avail_integrators(cls):
        if cls is StochasticSolver:
            return cls._avail_integrators.copy()
        return {
            **StochasticSolver.avail_integrators(),
            **cls._avail_integrators,
        }

    @property
    def options(self):
        """
        Options for stochastic solver:

        store_final_state: bool, default: False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: None, bool, default: None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        store_measurement: bool, default: False
            Whether to store the measurement for each trajectories.
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
    def options(self, new_options):
        MultiTrajSolver.options.fset(self, new_options)

    @classmethod
    def WienerFeedback(cls, default=None):
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
    def StateFeedback(cls, default=None, raw_data=False):
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
        "store_measurement": False,
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
        "store_measurement": False,
    }
