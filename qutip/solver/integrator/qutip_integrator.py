from ..integrator import IntegratorException, Integrator
from ..solver_base import Solver
from .explicit_rk import Explicit_RungeKutta
import numpy as np
from qutip import data as _data
from .verner7efficient import vern7_coeff
from .verner9efficient import vern9_coeff
from .tsit5 import tsit5_coeff

__all__ = [
    'IntegratorVern7', 'IntegratorVern9', 'IntegratorTsit5',
    'IntegratorDiag'
]


# Butcher tableau for simple method
# Never used directly: made available for debugging.
euler_coeff = {
    'order': 1,
    'a': np.array([[0.]], dtype=np.float64),
    'b': np.array([1.], dtype=np.float64),
    'c': np.array([0.], dtype=np.float64)
}

rk4_coeff = {
    'order': 4,
    'a': np.array([[0., 0., 0., 0.],
                   [.5, 0., 0., 0.],
                   [0., .5, 0., 0.],
                   [0., 0., 1., 0.]], dtype=np.float64),
    'b': np.array([1/6, 1/3, 1/3, 1/6], dtype=np.float64),
    'c': np.array([0., 0.5, 0.5, 1.0], dtype=np.float64)
}


class IntegratorVern7(Integrator):
    """
    QuTiP's implementation of Verner's "most efficient" Runge-Kutta method
    of order 7. These are Runge-Kutta methods with variable steps and dense
    output.

    The implementation uses QuTiP's Data objects for the state, allowing
    sparse, GPU or other data layer objects to be used efficiently by the
    solver in their native formats.

    See https://www.sfu.ca/~jverner/ for a detailed description of the
    methods.

    Usable with ``method="vern7"``
    """
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 1000,
        'first_step': 0,
        'max_step': 0,
        'min_step': 0,
        'interpolate': True,
    }
    support_time_dependant = True
    supports_blackbox = True
    method = 'vern7'
    tableau = vern7_coeff

    def _prepare(self):
        self._ode_solver = Explicit_RungeKutta(
            self.system, self.tableau,
            **self.options
        )
        self.name = self.method

    def get_state(self, copy=True):
        state = self._ode_solver.y
        return self._ode_solver.t, state.copy() if copy else state

    def set_state(self, t, state):
        self._ode_solver.set_initial_value(state.copy(), t)
        self._is_set = True

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

    @property
    def options(self):
        """
        Supported options by verner method:

        atol : float, default: 1e-8
            Absolute tolerance.

        rtol : float, default: 1e-6
            Relative tolerance.

        nsteps : int, default: 1000
            Max. number of internal steps/call.

        first_step : float, default: 0
            Size of initial step (0 = automatic).

        min_step : float, default: 0
            Minimum step size (0 = automatic).

        max_step : float, default: 0
            Maximum step size (0 = automatic)
            When using pulses, change to half the thinest pulse otherwise it
            may be skipped.

        interpolate : bool, default: True
            Whether to use interpolation step, faster most of the time.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


class IntegratorVern9(IntegratorVern7):
    """
    QuTiP's implementation of Verner's "most efficient" Runge-Kutta method
    of order 9. These are Runge-Kutta methods with variable steps and dense
    output.

    The implementation uses QuTiP's Data objects for the state, allowing
    sparse, GPU or other data layer objects to be used efficiently by the
    solver in their native formats.

    See https://www.sfu.ca/~jverner/ for a detailed description of the
    methods.

    Usable with ``method="vern9"``
    """
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 1000,
        'first_step': 0,
        'max_step': 0,
        'min_step': 0,
        'interpolate': True,
    }
    method = 'vern9'
    tableau = vern9_coeff


class IntegratorTsit5(IntegratorVern7):
    """
    QuTiP's implementation of Tsitouras's 5/4 order Runge-Kutta method.

    The implementation uses QuTiP's Data objects for the state, allowing
    sparse, GPU or other data layer objects to be used efficiently by the
    solver in their native formats.

    Runge–Kutta pairs of order 5(4) satisfying only the first column
    simplifying assumption,
    Ch. Tsitouras,
    Computers & Mathematics with Applications, Vol 62, Issue 2, 770-775
    Jan 2011

    Usable with ``method="tsit5"``
    """
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 1000,
        'first_step': 0,
        'max_step': 0,
        'min_step': 0,
        'interpolate': True,
    }
    method = 'tsit5'
    tableau = tsit5_coeff


class IntegratorDiag(Integrator):
    """
    Integrator solving the ODE by diagonalizing the system and solving
    analytically. It can only solve constant system and has a long preparation
    time, but the integration is fast.

    Usable with ``method="diag"``
    """
    integrator_options = {"eigensolver_dtype": "dense"}
    support_time_dependant = False
    supports_blackbox = False
    method = 'diag'

    def __init__(self, system, options):
        if not system.isconstant:
            raise ValueError("Hamiltonian system must be constant to use "
                             "diagonalized method")
        super().__init__(system, options)

    def _prepare(self):
        self._dt = 0.
        self._expH = None
        H0 = self.system(0).to(self.options["eigensolver_dtype"])
        self.diag, self.U = _data.eigs(H0.data, False)
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
        return self._t, _data.matmul(
            self.U, _data.dense.fast_from_numpy(self._y)
        )

    def set_state(self, t, state0):
        self._t = t
        self._y = _data.matmul(self.Uinv, state0).to_array()
        self._is_set = True

    @property
    def options(self):
        """
        Supported options by "diag" method:

        eigensolver_dtype : str, default: "dense"
            Qutip data type {"dense", "csr", etc.} to use when computing the
            eigenstates. The dense eigen solver is usually faster and more
            stable.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


Solver.add_integrator(IntegratorVern7, 'vern7')
Solver.add_integrator(IntegratorVern9, 'vern9')
Solver.add_integrator(IntegratorTsit5, 'tsit5')
Solver.add_integrator(IntegratorDiag, 'diag')
