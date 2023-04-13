__all__ = ["smesolve", "SMESolver", "ssesolve", "SSESolver"]

from .sode.ssystem import *
from .result import MultiTrajResult, Result, ExpectOp
from .multitraj import MultiTrajSolver
from .. import Qobj, QobjEvo, liouvillian, lindblad_dissipator
import numpy as np
from collections.abc import Iterable
from functools import partial


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

    def add(self, t, state, noise):
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


class _StochasticRHS:
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
            raise TypeError("The Hamiltonian must be am operator")
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

        if heterodyne:
            sc_ops = []
            for c_op in self.sc_ops:
                sc_ops.append(c_op / np.sqrt(2))
                sc_ops.append(c_op * (-1j / np.sqrt(2)))
            self.sc_ops = sc_ops

        if self.issuper and not self.H.issuper:
            self.dims = [self.H.dims, self.H.dims]
        else:
            self.dims = self.H.dims

    def __call__(self, options):
        if self.issuper:
            return StochasticOpenSystem(
                self.H, self.sc_ops, self.c_ops, options.get("derr_dt", 1e-6)
            )
        else:
            return StochasticClosedSystem(self.H, self.sc_ops)


def smesolve(
    H, rho0, tlist, c_ops=(), sc_ops=(), heterodyne=False, *,
    e_ops=(), args={}, ntraj=500, options=None,
    seeds=None, target_tol=None, timeout=None,
):
    """
    Solve stochastic master equation.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:class:`Qobj`, :class:`Coefficient`] or callable
        that can be made into :class:`QobjEvo` are also accepted.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    c_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        List of stochastic collapse operators.

    e_ops : : :class:`qutip.qobj`, callable, or list.
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    args : None / *dictionary*
        Dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    ntraj : int [500]
        Number of trajectories to compute.

    heterodyne : bool [False]
        Whether to use heterodyne or homodyne detection.

    seeds : int, SeedSequence, list, [optional]
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

    timeout : float [optional]
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed. Overwrite the option of the same name.

    options : None / dict
        Dictionary of options for the solver.

        - store_final_state : bool, [False]
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, None, [None]
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - store_measurement: bool, [False]
          Whether to store the measurement and wiener process for each
          trajectories.
        - keep_runs_results : bool, [False]
          Whether to store results from all trajectories or just store the
          averages.
        - normalize_output : bool, [False]
          Normalize output state to hide ODE numerical errors.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}, ["text"]
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict, [{"chunk_size": 10}]
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - method : str, ["rouchon"]
          Which stochastic differential equation integration method to use.
          Main ones are {"euler", "rouchon", "platen", "taylor1.5_imp"}
        - map : str {"serial", "parallel", "loky"}, ["serial"]
          How to run the trajectories. "parallel" uses concurent module to run
          in parallel while "loky" use the module of the same name to do so.
        - job_timeout : NoneType, int, [None]
          Maximum time to compute one trajectory.
        - num_cpus : NoneType, int, [None]
          Number of cpus to use when running in parallel. ``None`` detect the
          number of available cpus.
        - dt : float [0.001 ~ 0.0001]
          The finite steps lenght for the Stochastic integration method.
          Default change depending on the integrator.

        Other options could be supported depending on the integration method,
        see `SIntegrator <./classes.html#classes-sode>`_.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]
    sc_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in sc_ops]
    sol = SMESolver(
        H, sc_ops, c_ops=c_ops, options=options, heterodyne=heterodyne
    )
    return sol.run(
        rho0, tlist, ntraj, e_ops=e_ops,
        seed=seeds, target_tol=target_tol, timeout=timeout,
    )


def ssesolve(
    H, psi0, tlist, sc_ops=(), heterodyne=False, *,
    e_ops=(), args={}, ntraj=500, options=None,
    seeds=None, target_tol=None, timeout=None,
):
    """
    Solve stochastic Schrodinger equation.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:class:`Qobj`, :class:`Coefficient`] or callable
        that can be made into :class:`QobjEvo` are also accepted.

    psi0 : :class:`qutip.Qobj`
        Initial state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    sc_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        List of stochastic collapse operators.

    e_ops : :class:`qutip.qobj`, callable, or list.
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    args : None / *dictionary*
        Dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    ntraj : int [500]
        Number of trajectories to compute.

    heterodyne : bool [False]
        Whether to use heterodyne or homodyne detection.

    seeds : int, SeedSequence, list, [optional]
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

    timeout : float [optional]
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed. Overwrite the option of the same name.

    options : None / dict
        Dictionary of options for the solver.

        - store_final_state : bool, [False]
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, None, [None]
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - store_measurement: bool, [False]
          Whether to store the measurement and wiener process, or brownian
          noise for each trajectories.
        - keep_runs_results : bool, [False]
          Whether to store results from all trajectories or just store the
          averages.
        - normalize_output : bool, [False]
          Normalize output state to hide ODE numerical errors.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}, ["text"]
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict, [{"chunk_size": 10}]
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - method : str, ["rouchon"]
          Which stochastic differential equation integration method to use.
          Main ones are {"euler", "rouchon", "platen", "taylor1.5_imp"}
        - map : str {"serial", "parallel", "loky"}, ["serial"]
          How to run the trajectories. "parallel" uses concurent module to run
          in parallel while "loky" use the module of the same name to do so.
        - job_timeout : NoneType, int, [None]
          Maximum time to compute one trajectory.
        - num_cpus : NoneType, int, [None]
          Number of cpus to use when running in parallel. ``None`` detect the
          number of available cpus.
        - dt : float [0.001 ~ 0.0001]
          The finite steps lenght for the Stochastic integration method.
          Default change depending on the integrator.

        Other options could be supported depending on the integration method,
        see `SIntegrator <./classes.html#classes-sode>`_.

    Returns
    -------

    output: :class:`qutip.solver.Result`
        An instance of the class :class:`qutip.solver.Result`.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    sc_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in sc_ops]
    sol = SSESolver(H, sc_ops, options=options, heterodyne=heterodyne)
    return sol.run(
        psi0, tlist, ntraj, e_ops=e_ops,
        seed=seeds, target_tol=target_tol, timeout=timeout,
    )


class StochasticSolver(MultiTrajSolver):
    """
    Generic stochastic solver.
    """

    name = "StochasticSolver"
    resultclass = StochasticResult
    _avail_integrators = {}
    system = None
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "store_measurement": False,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "taylor1.5",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
    }

    def __init__(self, H, sc_ops, heterodyne, *, c_ops=(), options=None):
        self.options = options
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

    def _run_one_traj(self, seed, state, tlist, e_ops):
        """
        Run one trajectory and return the result.
        """
        result = StochasticTrajResult(
            e_ops,
            self.options,
            m_ops=self.m_ops,
            dw_factor=self.dW_factors,
            heterodyne=self.heterodyne,
        )
        generator = self._get_generator(seed)
        self._integrator.set_state(tlist[0], state, generator)
        state_t = self._restore_state(state, copy=False)
        result.add(tlist[0], state_t, None)
        for t in tlist[1:]:
            t, state, noise = self._integrator.integrate(t, copy=False)
            state_t = self._restore_state(state, copy=False)
            result.add(t, state_t, noise)
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

        store_final_state: bool, default=False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: bool, default=None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        store_measurement: bool, [False]
            Whether to store the measurement for each trajectories.
            Storing measurements will also store the wiener process, or
            brownian noise for each trajectories.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}, default="text"
            How to present the solver progress. 'tqdm' uses the python module
            of the same name and raise an error if not installed. Empty string
            or False will disable the bar.

        progress_kwargs: dict, default={"chunk_size":10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        keep_runs_results: bool
          Whether to store results from all trajectories or just store the
          averages.

        method: str, default="rouchon"
            Which ODE integrator methods are supported.

        map: str {"serial", "parallel", "loky"}, default="serial"
            How to run the trajectories. "parallel" uses concurent module to
            run in parallel while "loky" use the module of the same name to do
            so.

        job_timeout: None, int, default=None
            Maximum time to compute one trajectory.

        num_cpus: None, int, default=None
            Number of cpus to use when running in parallel. ``None`` detect the
            number of available cpus.

        bitgenerator: {None, "MT19937", "PCG64DXSM", ...}, default=None
            Which of numpy.random's bitgenerator to use. With ``None``, your
            numpy version's default is used.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        MultiTrajSolver.options.fset(self, new_options)


class SMESolver(StochasticSolver):
    r"""
    Stochastic Master Equation Solver.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:class:`Qobj`, :class:`Coefficient`] or callable
        that can be made into :class:`QobjEvo` are also accepted.

    sc_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        List of stochastic collapse operators.

    heterodyne : bool, [False]
        Whether to use heterodyne or homodyne detection.

    options : dict, [optional]
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
        "store_measurement": False,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "taylor1.5",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
    }


class SSESolver(StochasticSolver):
    r"""
    Stochastic Schrodinger Equation Solver.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        System Hamiltonian as a Qobj or QobjEvo for time-dependent
        Hamiltonians. List of [:class:`Qobj`, :class:`Coefficient`] or callable
        that can be made into :class:`QobjEvo` are also accepted.

    c_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of (:class:`QobjEvo`, :class:`QobjEvo` compatible format)
        List of stochastic collapse operators.

    heterodyne : bool, [False]
        Whether to use heterodyne or homodyne detection.

    options : dict, [optional]
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
        "store_measurement": False,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "platen",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
    }
