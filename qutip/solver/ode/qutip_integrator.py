from ..integrator import IntegratorException, Integrator
from ..solver_base import Solver
from .explicit_rk import Explicit_RungeKutta
from ..options import SolverOdeOptions
import numpy as np
from qutip import data as _data


__all__ = ['IntegratorVern', 'IntegratorDiag']


class IntegratorVern(Integrator):
    """
    QuTiP's implementation of Verner's "most efficient" Runge-Kutta method
    of order 7 and 9. These are Runge-Kutta methods with variable steps and dense
    output.

    The implementation uses QuTiP's Data objects for the state, allowing sparse, GPU or other
    data layer objects to be used efficiently by the solver in their native formats.

    See http://people.math.sfu.ca/~jverner/ for a detailed description of the methods.
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
        raise IntegratorException(self._ode_solver.status_message())


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
        self.diag = self.diag.reshape((-1, 1))
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


Solver.add_integrator(IntegratorVern, ['vern7', 'vern9'])
Solver.add_integrator(IntegratorDiag, 'diag')
