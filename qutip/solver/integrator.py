""" `Integrator`: ODE solver wrapper to use in qutip's Solver """


__all__ = ['Integrator', 'IntegratorException', 'add_integrator',
           'sesolve_integrators', 'mesolve_integrators', 'mcsolve_integrators']


class IntegratorException(Exception):
    """Error from the ODE solver being unable to integrate with the given
    parameters.

    Exemple
    -------
    - The solver cannot reach the desired tolerance within the maximum number
    of steps.
    - The step needed to be within desired tolerance is too small.
    """
    pass


class Integrator:
    """
    A wrapper around ODE solvers.
    It ensures a common interface for Solver usage.
    It takes and return states as :class:`qutip.core.data.Data`, it may return
    a different data-type than the input type.

    Parameters
    ----------
    sytem: qutip.QobjEvo
        Quantum system in which states evolve.

    options: qutip.SolverOdeOptions
        Options for the solver.
    """
    # Used options in qutip.SolverOdeOptions
    used_options = []
    # Can evolve time dependent system
    support_time_dependant = None
    # Use the QobjEvo's matmul_data method as the driving function
    use_QobjEvo_matmul = None
    name = ""

    def __init__(self, system, options):
        self.system = system
        self.options = {key: options[key]
                        for key in self.used_options
                        if key in options}
        self._is_set = False  # get_state can be used and return a valid state.
        self._prepare()

    def _prepare(self):
        """
        Initialize the solver
        It should also set the name of the solver to be displayed in Result.
        """
        raise NotImplementedError

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.

        .. note:
            It should set the flags `_is_set` to True.
        """
        raise NotImplementedError

    def integrate(self, t, step=False, copy=True):
        """
        Evolve to t.

        If `step` advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time with subsequent calls.

        Before calling `integrate` for the first time, the initial step should
        be set with `set_state`.

        Parameters
        ----------
        t : float
            Time to integrate to, should be larger than the previous time. If
            the last integrate call was use with ``step=True``, the time can be
            between the time at the start of the last call and now.

        step : bool [False]
            When True, integrate for a one internal step of the ODE solver.

        copy : bool [True]
            Whether to return a copy of the state or the state itself.

        Return
        ------
        (t, state) : (float, qutip.Data)
            The state of the solver at ``t``. The returned this can differ from
            the input time only when ``step=True``.
        """
        raise NotImplementedError

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair (t, state).

        Parameters
        ----------
        copy : bool (True)
            Whether to return the data stored in the Integrator or a copy.

        Return
        ------
        (t, state) : (float, qutip.Data)
            The state of the solver at ``t``.
        """
        raise NotImplementedError

    def run(self, tlist):
        """
        Integrate the system yielding the state for each times in tlist.

        Parameters
        ----------
        tlist : *list* / *array*
            List of times to yield the state.

        Yield
        -----
        (t, state) : (float, qutip.Data)
            The state of the solver at each ``t`` of tlist.
        """
        for t in tlist[1:]:
            yield self.integrate(t, False)

    def reset(self):
        """Reset internal state of the ODE solver."""
        if self._is_set:
            self.set_state(*self.get_state())

    def arguments(self, args):
        """
        Change the argument of the system.
        Reset the ODE solver to ensure numerical validity.

        Parameters
        ----------
        args : dict
            New arguments
        """
        self.system.arguments(args)
        self.reset()


# TODO: These integrator set will be part of the solver classes
# So sesolve will know which integrator it support etc.
# `mesolve` and `sesolve` will use the same integrators, but some integrations
# methods can not work for `mcsolve`.
sesolve_integrators = {}
mesolve_integrators = {}
mcsolve_integrators = {}


def add_integrator(integrator, keys, integrator_set, options_class=None):
    """
    Register an integrator.

    Parameters
    ----------
    integrator : Integrator
        The ODE solver to register.

    keys : list of str
        Values of the method options that refer to this integrator.

    integrator_set : dict
        Group of integrators to which register the integrator.

    options_class : SolverOdeOptions
        If given, will add the ODE options supported by this integrator to
        those supported by the options_class. ie. If the integrator use a
        `stiff` options that Qutip's `SolverOdeOptions` does not have, it will
        add it support.
    """
    # TODO: Insert in Solvers as a classmethod.
    if not issubclass(integrator, Integrator):
        raise TypeError(f"The integrator {integrator} must be a subclass"
                        " of `qutip.solver.Integrator`")
    if not isinstance(keys, list):
        keys = [keys]
    for key in keys:
        integrator_set[key] = integrator
    if integrator.used_options and options_class:
        for opt in integrator.used_options:
            options_class.extra_options.add(opt)
