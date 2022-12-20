

from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data

cdef class StochasticIntegrator:
    cdef:
        StochasticSystem system
        double dt
        double t
        bool support_time_dependant
        bool supports_blackbox

    def __init__(self, system, dt, options):
        self.system = system
        self.dt = dt
        self.options = options

    def set_state(self, t, state0, seed):
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

          copy : bool [True]
              Whether to return a copy of the state or the state itself.

          Returns
          -------
          (t, state) : (float, qutip.Data)
              The state of the solver at ``t``.
          """
          raise NotImplementedError

      def get_state(self, copy=True):
          """
          Obtain the state of the solver as a pair (t, state).

          Parameters
          ----------
          copy : bool (True)
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
