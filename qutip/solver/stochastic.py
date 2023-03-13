__all__ = ["smesolve", "SMESolver", "ssesolve", "SSESolver", "StochasticSolver"]

from .sode.ssystem import *
from .result import MultiTrajResult, Result
from .multitraj import MultiTrajSolver
from ..import Qobj, QobjEvo, liouvillian, lindblad_dissipator
import numpy as np
from collections.abc import Iterable


class StochasticTrajResult(Result):
    def _post_init(self, m_ops=(), dw_factor=()):
        super()._post_init()
        self.noise = []
        self.m_ops = m_ops
        self.dW_factor = dw_factor
        self.measurements = [[] for _ in range(len(m_ops))]

    def _add_measurement(self, t, state, noise):
        expects = [m_op.expect(t, state) for m_op in self.m_ops]
        for measure, expect, dW, factor in zip(
            self.measurements, expects, noise, self.dW_factor
        ):
            measure.append(expect + dW * factor)

    def add(self, t, state, noise):
        super().add(t, state)

        if noise is not None:
            self.noise.append(noise)
            if self.options["store_measurement"]:
                dt = self.times[-1] - self.times[-2]
                self._add_measurement(t, state, noise / dt)


class StochasticResult(MultiTrajResult):
    def _reduce_measurements(self, trajectory):
        # Since measurements of each trajectories is kept, we keep only the
        # array to save memory.
        trajectory.measurements = np.array(trajectory.measurements)
        if self.options["heterodyne"]:
            shape = trajectory.measurements.shape
            trajectory.measurements = (
                trajectory.measurements.reshape(-1, 2, shape[-1])
            )
        self.measurement.append(trajectory.measurements)

    def _post_init(self):
        super()._post_init()
        self.measurement = []
        if self.options['store_measurement']:
            self.add_processor(self._reduce_measurements)


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

    def __call__(self):
        if self.issuper:
            return StochasticOpenSystem(self.H , self.sc_ops, self.c_ops)
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
    sol = SMESolver(H, sc_ops, c_ops=c_ops, options=options, m_ops=m_ops)
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
    sol = SSESolver(H, sc_ops, options=options, m_ops=m_ops)
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
        "store_measurement": False,
        "dw_factor": None,
    }

    def __init__(self, H, sc_ops, *, c_ops=(), options=None, m_ops=()):
        self.options = options
        heterodyne = self.options["heterodyne"]
        if self.name == "ssesolve" and c_ops:
            raise ValueError("")

        rhs = StochasticRHS(self._open, H, sc_ops, c_ops, heterodyne)
        super().__init__(rhs, options=options)

        if self.options["store_measurement"]:
            n_m_ops = len(sc_ops) * (1 + int(heterodyne))
            dW_factor = self.options["dw_factor"] or 1.

            if len(m_ops) == n_m_ops:
                self.m_ops = m_ops
            elif heterodyne:
                self.m_ops = []
                for op in sc_ops:
                    self.m_ops += [
                        op + op.dag(), -1j * (op - op.dag())
                    ]
                dW_factor = dW_factor * 0.5**0.5
            else:
                self.m_ops = [op + op.dag() for op in sc_ops]

            if not isinstance(dW_factor, Iterable):
                dW_factor = [dW_factor] * n_m_ops

            if len(dW_factor) == len(sc_ops) and heterodyne:
                dW_factor = [ i for i in dW_factor for _ in range(2) ]

            if len(dW_factor) != n_m_ops:
                raise ValueError("Bad dW_factor option")
            self.dW_factors = dW_factor

        else:
            self.m_ops = []
            self.dW_factors = []

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
