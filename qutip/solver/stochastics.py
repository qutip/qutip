__all__ = ["smesolve", "SMESolver"]

from .integrator import Integrator
from .result import MultiTrajResult, Result
from .multitraj import MultiTrajSolver


class SIntegrator(Integrator):
    def set_state(self, t, state0, generator):
        """
        Set the state of the SODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.

        generator : numpy.random.generator
            Random number generator.
        """
        raise NotImplementedError


    def integrate(self, t, copy=True):
        """
        Evolve to t.

        Before calling `integrate` for the first time, the initial state should
        be set with `set_state`.

        Parameters
        ----------
        t : float
            Time to integrate to, should be larger than the previous time.

        copy : bool [True]
            Whether to return a copy of the state or the state itself.

        Returns
        -------
        (t, state, noise) : (float, qutip.Data, np.ndarray)
            The state of the solver at ``t``.
        """
        raise NotImplementedError


    def mcstep(self, t, copy=True):
        raise NotImplementedError


class StochasticTrajResult(Result):
    def _post_init(self, m_ops=()):
        super()._post_init()
        self.noise = []
        self.m_ops = m_ops
        self.measurements = [[] for _ in range(len(m_ops))]
        self.add_processor(self._add_measurement)

    def _add_measurement(self, t, state, noise):
        expects = [m_op.expect(t, state) for m_op in self.m_ops]
        noises = np.sum(noise, axis=0)
        for measure, expect, dW in zip(self.measurements, expects, noises):
            measure.append(expect + dW)

    def add(t, state, noise):
        super().add(t, state)
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
    sol = SMESolver(L, sc_ops, options=options)
    return sol.run(rho0, tlist, e_ops, m_ops, ntraj)


class SMESolver(MultiTrajSolver):
    name = "generic multi trajectory"
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

    def __init__(self, H, sc_ops, *, options=None):
        if isinstance(sc_ops, (Qobj, QobjEvo)):
            sc_ops = [sc_ops]
        sc_ops = [QobjEvo(c_op) for c_op in sc_ops]
        if not H.issuper:
            L = liouvillian(H, sc_ops)
        else:
            L = H + sum(lindblad_dissipator(c_op) for c_op in sc_ops)
        rhs = StochasticOpenSystem(L, sc_ops)
        super().__init__(rhs, options=options)

        if self.options["heterodyne"]:
            self.m_ops += [
                sc_op + sc_op.dag(), -1j * (sc_op - sc_op.dag())
                for sc_op in sc_ops
            ]
        else:
            self.m_ops = [sc_op + sc_op.dag() for sc_op in sc_ops]

    def _run_one_traj(self, seed, state, tlist, e_ops):
        """
        Run one trajectory and return the result.
        """
        result = StochasticTrajResult(e_ops, self.options, m_ops=self.m_ops)
        generator = self._get_generator(seed)
        self._integrator.set_state(tlist[0], state, generator)
        result.add(tlist[0], self._restore_state(state, copy=False), None)
        for t in tlist[1:]:
            t, state, noise = self._integrator.integrate(t, copy=False)
            result.add(t, self._restore_state(state, copy=False), noise)
        return seed, result
