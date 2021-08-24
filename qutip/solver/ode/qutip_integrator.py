from ..integrator import (IntegratorException, Integrator, sesolve_integrators,
                          mesolve_integrators, mcsolve_integrators,
                          add_integrator)
from .explicit_rk import Explicit_RungeKutta
from ..options import SolverOdeOptions
import numpy as np
from qutip import data as _data


__all__ = ['IntegratorVern', 'IntegratorDiag']


class IntegratorVern(Integrator):
    """
    Qutip's implementation of Verner 'most efficient' Runge-Kutta method
    of order 7 and 9. It's a Runge-Kutta method with variable steps and dense
    output.
    Use qutip's Data object for the state, allowing sparse or gpu states.
    [http://people.math.sfu.ca/~jverner/]
    """
    used_options = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                    'min_step', 'interpolate', 'method']
    support_time_dependant = True
    use_QobjEvo_matmul = True

    def _prepare(self):
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

    def integrate(self, t, step=False, copy=True):
        """
        Evolve to t, must be `prepare` before.
        return the pair (t, state).
        """
        self._ode_solver.integrate(t, step=step)
        self._check_failed_integration()
        return self.get_state(copy)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'Too much work done in one call. Try to increase the nsteps '
                'parameter or increasing the tolerance.',
            -2: 'Step size becomes too small. Try increasing tolerance',
            -3: 'Step outside available range.',
        }
        raise IntegratorException(messages[self._ode_solver.status])


class IntegratorDiag(Integrator):
    """
    Integrator solving the ODE by diagonalizing the system and solving
    analytically. It can only solve constant system and has a long preparation
    time, but the integration is very fast.
    """

    used_options = []
    support_time_dependant = False
    use_QobjEvo_matmul = False

    def __init__(self, system, options):
        if not system.isconstant:
            raise ValueError("Hamiltonian system must be constant to use "
                             "diagonalized method")
        self.system = system
        self.options = options
        self._dt = 0.
        self._expH = None
        self._prepare()

    def _prepare(self):
        """
        Initialize the solver
        """
        self.diag, self.U = self.system(0).eigenstates()
        self.diag = self.diag.reshape((-1,1))
        self.U = np.hstack([eket.full() for eket in self.U])
        self.Uinv = np.linalg.inv(self.U)
        self.name = "qutip diagonalized"

    def integrate(self, t, step=False, copy=True):
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


sets = [sesolve_integrators, mesolve_integrators, mcsolve_integrators]
for integrator_set in sets:
    add_integrator(IntegratorVern, ['vern7', 'vern9'], integrator_set, SolverOdeOptions)
    add_integrator(IntegratorDiag, 'diag', integrator_set, SolverOdeOptions)
