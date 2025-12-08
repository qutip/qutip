""" `Integrator`: ODE solver wrapper to use in qutip's Solver """
import numpy as np

__all__ = ['Integrator', 'IntegratorException']


class IntegratorException(Exception):
    """Error from the ODE solver being unable to integrate with the given
    parameters.

    Example
    -------
    - The solver cannot reach the desired tolerance within the maximum number
    of steps.
    - The step needed to be within desired tolerance is too small.
    """


class Integrator:
    """
    A wrapper around ODE solvers.
    It ensures a common interface for Solver usage.
    It takes and return states as :class:`.Data`, it may return
    a different data-type than the input type.

    Parameters
    ----------
    system: qutip.QobjEvo
        Quantum system in which states evolve.

    options: dict
        Options for the integrator.

    Class Attributes
    ----------------
    name : str
        The name of the integrator.

    supports_blackbox : bool
        If True, then the integrator calls only ``system.matmul``,
        ``system.matmul_data``, ``system.expect``, ``system.expect_data`` and
        ``isconstant``, ``isoper`` or ``issuper``. This allows the solver using
        the integrator to modify the system in creative ways. In particular,
        the solver may modify the system depending on *both* the time ``t``
        *and* the current ``state`` the system is being applied to.

        If the integrator calls any other methods, set to False.

    supports_time_dependent : bool
        If True, then the integrator supports time dependent systems. If False,
        ``supports_blackbox`` should usually be ``False`` too.

    integrator_options : dict
        A dictionary of options used by the integrator and their default
        values. Once initiated, ``self.options`` will be a dict with the same
        keys, not the full options object passed to the solver. Options' keys
        included here will be supported by the :cls:SolverOdeOptions.
    """
    # Dict of used options and their default values
    integrator_options = {}
    _options = None
    # Can evolve time dependent system
    support_time_dependant = None
    # Whether the integrator used the system QobjEvo as a blackbox
    supports_blackbox = None
    # The name of the integrator
    name = None
    method = ""

    def __init__(self, system, options):
        self.system = system
        self._is_set = False  # get_state can be used and return a valid state.
        self._back = (np.inf, None)
        self._options = self.integrator_options.copy()
        self.options = options
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

    def integrate(self, t, copy=True):
        """
        Evolve to t.

        Before calling `integrate` for the first time, the initial state should
        be set with `set_state`.

        Parameters
        ----------
        t : float
            Time to integrate to, should be larger than the previous time.

        copy : bool, default: True
            Whether to return a copy of the state or the state itself.

        Returns
        -------
        (t, state) : (float, qutip.Data)
            The state of the solver at ``t``.
        """
        raise NotImplementedError

    def mcstep(self, t, copy=True):
        """
        Evolve toward the time ``t``.

        If ``t`` is larger than the present state's ``t``, advance the internal
        state toward ``t``. If  ``t`` is smaller than the present ``t``, but
        larger than the previous one, it does an interpolation step and returns
        the state at that time. When advancing the state, it may return it at a
        time between present time and the asked ``t`` if more efficent for
        subsequent interpolation step.

        Before calling `mcstep` for the first time, the initial state should
        be set with `set_state`.

        Parameters
        ----------
        t : float
            Time to integrate to, should be larger than the previous time. If
            the last integrate call was use with ``step=True``, the time can be
            between the time at the start of the last call and now.

        copy : bool, default: True
            Whether to return a copy of the state or the state itself.

        Returns
        -------
        (t, state) : (float, qutip.Data)
            The state of the solver at ``t``. The returned time ``t`` can
            differ from the input time only when ``step=True``.

        .. note:
            The default implementation may be overridden by integrators that
            can provide a more efficient one.
        """
        t_last, state = self.get_state()
        if t > t_last:
            self._back = t_last, state
        elif t > self._back[0]:
            self.set_state(*self._back)
        else:
            raise IntegratorException(
                "`t` is outside the integration range: "
                f"{self._back[0]}..{t_last}."
            )
        return self.integrate(t, copy)

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair (t, state).

        Parameters
        ----------
        copy : bool, default: True
            Whether to return the data stored in the Integrator or a copy.

        Returns
        -------
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

        Yields
        ------
        (t, state) : (float, qutip.Data)
            The state of the solver at each ``t`` of tlist.
        """
        for t in tlist[1:]:
            yield self.integrate(t, False)

    def reset(self, hard=False):
        """Reset internal state of the ODE solver."""
        if self._is_set:
            state = self.get_state()
        if hard:
            self._prepare()
        if self._is_set:
            self.set_state(*state)

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

    @property
    def options(self):
        # Options should be overwritten by each integrators.
        return self._options

    @options.setter
    def options(self, new_options):
        # This does not apply the new options.
        self._options = {
            **self._options,
            **{
               key: new_options[key]
               for key in self.integrator_options.keys()
               if key in new_options and new_options[key] is not None
            }
        }
