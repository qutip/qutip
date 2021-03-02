from ..evolver import Evolver, evolver_collection, EvolverException
from .explicit_rk import Explicit_RungeKutta
import numpy as np
from qutip import data as _data

__all__ = ['EvolverVern', 'EvolverDiag']


class EvolverVern(Evolver):
    """
    Evolver wrapping Qutip's implementation of Verner 'most efficient'
    Runge-Kutta method for solver ODE.
    http://people.math.sfu.ca/~jverner/
    """
    description = "Qutip implementation of Verner's most efficient Runge-Kutta"
    used_options = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                    'min_step', 'interpolate', 'method']

    def prepare(self):
        """
        Initialize the solver
        """
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver = Explicit_RungeKutta(self.system, **opt)
        self.name = "qutip " + self.options.ode['method']

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        state = self._ode_solver.y
        return self._ode_solver.t, state.copy() if copy else state

    def set_state(self, t, state):
        """
        Set the state of the ODE solver.
        """
        self._ode_solver.set_initial_value(state, t)

    def step(self, t, copy=True):
        """
        Evolve to t, must be `prepare` before.
        return the pair (t, state).
        """
        self._ode_solver.integrate(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def one_step(self, t, copy=True):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        self._ode_solver.integrate(t, step=True)
        self._check_failed_integration()
        return self.get_state(copy)

    def backstep(self, t, copy=True):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        self._ode_solver.integrate(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'Too much work done in one call. Try to increase the nsteps '
                'parameter or increasing the tolerance.',
            -2: 'Step size becomes too small. Try increasing tolerance',
            -3: 'Etep outside available range.',
        }
        raise EvolverException(messages[self._ode_solver.status])


vernlimits = {
    "base": True,
    "backstep": True,
    "update_args": True,
    "feedback": True,
    "time_dependent": True,
}


evolver_collection.add(EvolverVern, methods=['vern7', 'vern9'],
                       limits=vernlimits, _test=False)


class EvolverDiag(Evolver):
    """
    Evolver solving the ODE by diagonalizing the system.
    """
    description = "Diagonalize a constant system"
    used_options = []

    def __init__(self, system, options, args, feedback_args):
        if not system.const:
            raise ValueError("Hamiltonian system must be constant to use "
                             "diagonalized method")
        self.system = system
        self.options = options
        self._dt = 0.
        self._expH = None
        self._stats = {}
        self.prepare()

    def prepare(self):
        """
        Initialize the solver
        """
        self.diag, self.U = self.system(0).eigenstates()
        self.diag = self.diag.reshape((-1,1))
        self.U = np.hstack([eket.full() for eket in self.U])
        self.Uinv = np.linalg.inv(self.U)
        self.name = "qutip diagonalized"

    def step(self, t, copy=True):
        """ Evolve to t, must be `set` before. """
        dt = t - self._t
        if dt == 0:
            return self.get_state()
        elif self._dt != dt:
            self._expH = np.exp(self.diag * dt)
            self._dt = dt
        self._y *= self._expH
        self._t = t
        return self.get_state(copy)

    def one_step(self, t, copy=True):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        return self.step(t, copy=True)

    def backstep(self, t, copy=True):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        return self.step(t, copy=True)

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        y = self.U @ self._y
        return self._t, _data.dense.Dense(y, copy=copy)

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.
        """
        self._t = t
        self._y = (self.Uinv @ state0.to_array())


diaglimits = {
    "base": False,
    "backstep": True,
    "update_args": False,
    "feedback": False,
    "time_dependent": False,
}


evolver_collection.add(EvolverDiag, methods=['diag'],
                       limits=diaglimits, _test=False)
