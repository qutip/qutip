__all__ = ["smesolve", "SMESolver", "ssesolve", "SSESolver", "StochasticSolver"]

from .sode.ssystem import *
from .result import MultiTrajResult, Result, ExpectOp
from .multitraj import MultiTrajSolver
from ..import Qobj, QobjEvo, liouvillian, lindblad_dissipator
import numpy as np
from collections.abc import Iterable
from functools import partial


class StochasticTrajResult(Result):
    def _post_init(self, m_ops=(), dw_factor=()):
        super()._post_init()
        self.W = []
        self.m_ops = []
        self.m_expect = []
        self.dW_factor = dw_factor
        for op in m_ops:
            f = self._e_op_func(op)
            self.W.append([0.])
            self.m_expect.append([])
            self.m_ops.append(ExpectOp(op, f, self.m_expect[-1].append))
            self.add_processor(self.m_ops[-1]._store)

    def add(self, t, state, noise):
        super().add(t, state)
        if noise is not None and (self.options['store_wiener_process'] or self.options['store_measurement']):
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
        if self.options["heterodyne"]:
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
        if self.options["heterodyne"]:
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
        noise = np.einsum("i,ij,j->ij", self.dW_factor, np.diff(self.W, axis=1), (1/dts))
        if self.options["heterodyne"]:
            m_expect = m_expect.reshape(-1, 2, m_expect.shape[1])
            noise = noise.reshape(-1, 2, noise.shape[1])
        return m_expect + noise


class StochasticResult(MultiTrajResult):
    def _post_init(self):
        super()._post_init()

        store_measurement = self.options['store_measurement']
        keep_runs = self.options['keep_runs_results']
        store_wiener_process = self.options['store_wiener_process']

        if (
            store_measurement and not store_wiener_process
        ):
            raise ValueError(
                "Keeping runs is needed to store measurements "
                "and wiener processes."
            )

        if not keep_runs and store_wiener_process:
            self.add_processor(partial(self._reduce_attr, attr="wiener_process"))
            self._wiener_process = []
            self.add_processor(partial(self._reduce_attr, attr="dW"))
            self._dW = []

        if not keep_runs and store_measurement:
            self.add_processor(partial(self._reduce_attr, attr="measurement"))
            self._measurement = []

    def _reduce_attr(self, trajectory, attr):
        getattr(self, "_" + attr).append(getattr(trajectory, attr))

    def _trajectories_attr(self, attr):
        if hasattr(self, "_" + attr):
            return getattr(self, "_" + attr)
        elif self.options['keep_runs_results']:
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


class StochasticRHS:
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
                self.H , self.sc_ops, self.c_ops, options.get("dt", 1.-6)
            )
        else:
            return StochasticClosedSystem(self.H , self.sc_ops)


def smesolve(H, rho0, tlist, c_ops=(), sc_ops=(), e_ops=(), m_ops=(),
             args={}, ntraj=500, options=None):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, or time dependent system.
        System Hamiltonian.
        Can depend on time, see StochasticSolverOptions help for format.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.
        Can depend on time, see StochasticSolverOptions help for format.

    sc_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.
        Can depend on time, see StochasticSolverOptions help for format.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    m_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    args : dict
        ...

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]
    sc_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in sc_ops]
    sol = SMESolver(H, sc_ops, c_ops=c_ops, options=options)
    if m_ops:
        sol.m_ops = m_ops
    return sol.run(rho0, tlist, ntraj, e_ops=e_ops)


def ssesolve(H, psi0, tlist, sc_ops=(), e_ops=(), m_ops=(),
             args={}, ntraj=500, options=None):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, or time dependent system.
        System Hamiltonian.
        Can depend on time, see StochasticSolverOptions help for format.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.
        Can depend on time, see StochasticSolverOptions help for format.

    sc_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.
        Can depend on time, see StochasticSolverOptions help for format.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    m_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    args : dict
        ...

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    sc_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in sc_ops]
    sol = SSESolver(H, sc_ops, options=options)
    if m_ops:
        sol.m_ops = m_ops
    return sol.run(psi0, tlist, ntraj, e_ops=e_ops)


class StochasticSolver(MultiTrajSolver):
    name = "StochasticSolver"
    resultclass = StochasticResult
    _avail_integrators = {}
    system = None
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "rouchon",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
        "heterodyne": False,
        "store_wiener_process": False,
        "store_measurement": False,
        "dw_factor": None,
    }

    def __init__(self, H, sc_ops, *, c_ops=(), options=None):
        self.options = options
        heterodyne = self.options["heterodyne"]
        if self.name == "ssesolve" and c_ops:
            raise ValueError("")

        rhs = StochasticRHS(self._open, H, sc_ops, c_ops, heterodyne)
        super().__init__(rhs, options=options)

        if self.options["store_measurement"]:
            self.options["store_wiener_process"] = True
            dW_factor = self.options["dw_factor"] or 1.

            if heterodyne:
                self._m_ops = []
                for op in sc_ops:
                    self._m_ops += [
                        op + op.dag(), -1j * (op - op.dag())
                    ]
                self._dW_factors = np.ones(len(sc_ops) * 2) * 0.5**0.5

            else:
                self._m_ops = [op + op.dag() for op in sc_ops]
                self._dW_factors = np.ones(len(sc_ops))

        else:
            self._m_ops = []
            self.dW_factors = []

    @property
    def m_ops(self):
        return self._m_ops

    @m_ops.setter
    def m_ops(self, new_m_ops):
        if not self.options["store_measurement"]:
            raise ValueError(
                "The 'store_measurement' options must be set to "
                "`True` to use m_ops."
            )

        if len(new_m_ops) != len(self.rhs.sc_ops):
            if self.options["heterodyne"]:
                raise ValueError(
                    f"2 `m_ops` per `sc_ops`, {len(self.rhs.sc_ops)} operators"
                    " are expected for heterodyne measurement."
                )
            else:
                raise ValueError(
                    f"{len(self.rhs.sc_ops)} measurements "
                    "operators are expected."
                )
        self._m_ops = new_m_ops

    @property
    def dW_factors(self):
        return self._dW_factors

    @dW_factors.setter
    def dW_factors(self, new_dW_factors):
        if not self.options["store_measurement"]:
            raise ValueError(
                "The 'dW_factors' are only used with measurements."
            )

        if len(new_dW_factors) != len(self.rhs.sc_ops):
            if self.options["heterodyne"]:
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
            e_ops, self.options, m_ops=self.m_ops, dw_factor=self.dW_factors,
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


class SMESolver(StochasticSolver):
    name = "smesolve"
    _avail_integrators = {}
    _open = True


class SSESolver(StochasticSolver):
    name = "ssesolve"
    _avail_integrators = {}
    _open = False
