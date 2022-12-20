__all__ = ["smesolve", "SMESolver"]






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




def smesolve(H, rho0, times, c_ops=(), sc_ops=(), e_ops=(), m_ops=(),
             args={}, options=None):
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

    times : *list* / *array*
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

    args : dict
        ...

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.

    """









class SMESolver(MultiTrajSolver):
    pass
