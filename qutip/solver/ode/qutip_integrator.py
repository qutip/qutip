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
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 1000,
        'first_step': 0,
        'max_step': 0,
        'min_step': 0,
        'interpolate': True,
        'method': 'vern7'
    }
    support_time_dependant = True
    supports_blackbox = True

    def _prepare(self):
        self._ode_solver = Explicit_RungeKutta(self.system, **self.options)
        self.name = "qutip " + self.options['method']

    def get_state(self, copy=True):
        state = self._ode_solver.y
        return self._ode_solver.t, state.copy() if copy else state

    def set_state(self, t, state):
        self._ode_solver.set_initial_value(state, t)

    def integrate(self, t, copy=True):
        self._ode_solver.integrate(t, step=False)
        self._check_failed_integration()
        return self.get_state(copy)

    def mcstep(self, t, copy=True):
        self._ode_solver.integrate(t, step=True)
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

    integrator_options = {}
    support_time_dependant = False
    supports_blackbox = False

    def __init__(self, system, options):
        if not system.isconstant:
            raise ValueError("Hamiltonian system must be constant to use "
                             "diagonalized method")
        self.system = system
        self._dt = 0.
        self._expH = None
        self._prepare()

    def _prepare(self):
        self.diag, self.U = _data.eigs(self.system(0).data, False)
        self.diag = self.diag.reshape((-1,1))
        self.Uinv = _data.inv(self.U)
        self.name = "qutip diagonalized"

    def integrate(self, t, copy=True):
        dt = t - self._t
        if dt == 0:
            return self.get_state()
        elif self._dt != dt:
            self._expH = np.exp(self.diag * dt)
            self._dt = dt
        self._y *= self._expH
        self._t = t
        return self.get_state(copy)

    def mcstep(self, t, copy=True):
        return self.integrate(t, copy=copy)

    def get_state(self, copy=True):
        return self._t, _data.matmul(self.U, _data.dense.Dense(self._y))

    def set_state(self, t, state0):
        self._t = t
        self._y = _data.matmul(self.Uinv, state0).to_array()


sets = [sesolve_integrators, mesolve_integrators, mcsolve_integrators]
for integrator_set in sets:
    add_integrator(IntegratorVern, ['vern7', 'vern9'], integrator_set, SolverOdeOptions)
    add_integrator(IntegratorDiag, 'diag', integrator_set, SolverOdeOptions)
