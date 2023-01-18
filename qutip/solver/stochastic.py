__all__ = ["smesolve", "SMESolver"]

from .sode.ssystem import *
from .result import MultiTrajResult, Result
from .multitraj import MultiTrajSolver
from ..import Qobj, QobjEvo, liouvillian, lindblad_dissipator
import numpy as np


class StochasticTrajResult(Result):
    def _post_init(self, m_ops=()):
        super()._post_init()
        self.noise = []
        self.m_ops = m_ops
        self.measurements = [[] for _ in range(len(m_ops))]

    def _add_measurement(self, t, state, noise):
        expects = [m_op.expect(t, state) for m_op in self.m_ops]
        noises = np.sum(noise, axis=0)
        for measure, expect, dW in zip(self.measurements, expects, noises):
            measure.append(expect + dW)

    def add(self, t, state, noise):
        super().add(t, state)
        if noise is not None:
            self._add_measurement(t, state, noise)
        self.noise.append(noise)


class StochasticResult(MultiTrajResult):
    @property
    def measurement(self):
        return []


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
    L = QobjEvo(H, args=args)
    c_ops = [QobjEvo(c_op, args=args) for c_op in c_ops]
    sc_ops = [QobjEvo(c_op, args=args) for c_op in sc_ops]
    L = liouvillian(H, c_ops)
    sol = SMESolver(L, sc_ops, options=options or {}, m_ops=m_ops)
    return sol.run(rho0, tlist, ntraj, e_ops=e_ops)


class StochasticSolver(MultiTrajSolver):
    name = "generic stochastic"
    resultclass = StochasticResult
    _avail_integrators = {}
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "euler",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
    }

    def _run_one_traj(self, seed, state, tlist, e_ops):
        """
        Run one trajectory and return the result.
        """
        result = StochasticTrajResult(e_ops, self.options, m_ops=self.m_ops)
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
    resultclass = StochasticResult
    _avail_integrators = {}
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "platen",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
        "heterodyne": False,
    }

    def __init__(self, H, sc_ops, *, options=None, m_ops=()):
        self._options = self.solver_options.copy()
        self.options = options
        if isinstance(sc_ops, (Qobj, QobjEvo)):
            sc_ops = [sc_ops]
        sc_ops = [QobjEvo(c_op) for c_op in sc_ops]
        if not H.issuper:
            L = liouvillian(H, sc_ops)
        else:
            if any(not c_op.isoper for c_op in sc_ops):
                raise TypeError("sc_ops must be operators")
            L = H + sum(lindblad_dissipator(c_op) for c_op in sc_ops)
        rhs = StochasticOpenSystem(L, sc_ops, self.options["heterodyne"])
        super().__init__(rhs, options=options)


        if len(m_ops) == rhs.num_collapse:
            self.m_ops = m_ops
        elif self.options["heterodyne"]:
            self.m_ops = []
            for sc_op in sc_ops:
                self.m_ops += [
                    sc_op + sc_op.dag(), -1j * (sc_op - sc_op.dag())
                ]
        else:
            self.m_ops = [sc_op + sc_op.dag() for sc_op in sc_ops]
