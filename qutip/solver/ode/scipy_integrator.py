"""ODE integrator from scipy."""

__all__ = ['IntegratorScipyZvode', 'IntegratorScipyDop853',
           'IntegratorScipylsoda']

import numpy as np
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from qutip.core import data as _data
from ..options import SolverOdeOptions
from qutip.core.data.reshape import column_unstack_dense, column_stack_dense
from ..integrator import IntegratorException, Integrator
from ..solver_base import Solver
import warnings


class IntegratorScipyZvode(Integrator):
    """
    Integrator using Scipy `ode` with zvode integrator. Ordinary Differential
    Equation solver by netlib (http://www.netlib.org/odepack). Support 'Adams'
    and 'BDF' methods for non-stiff and stiff systems respectively.
    """
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 2500,
        'method': 'adams',
        'order': 12,
        'first_step': 0,
        'max_step': 0,
        'min_step': 0,
    }
    support_time_dependant = True
    supports_blackbox = True

    class _zvode(zvode):
        """Overwrite the scipy's zvode to advance to max to ``t`` with step."""
        def step(self, *args):
            itask = self.call_args[2]
            self.rwork[0] = args[4]
            self.call_args[2] = 5
            r = self.run(*args)
            self.call_args[2] = itask
            return r

    def _prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self._mul_np_vec)
        self._ode_solver.set_integrator('zvode')
        self._ode_solver._integrator = self._zvode(**self.options)
        self.name = "scipy zvode " + self.options['method']

    def _mul_np_vec(self, t, vec):
        """
        Interface between scipy which use numpy and the driver, which use data.
        """
        state = _data.dense.fast_from_numpy(vec)
        column_unstack_dense(state, self._size, inplace=True)
        out = self.system.matmul_data(t, state)
        column_stack_dense(out, inplace=True)
        return out.as_ndarray().ravel()

    def set_state(self, t, state0):
        self._is_set = True
        self._back = t
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel(),
            t
        )

    def get_state(self, copy=True):
        if not self._is_set:
            raise IntegratorException("The state is not initialted")
        self._check_failed_integration()
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.Dense(self._ode_solver._y, copy=copy),
                self._size,
                inplace=True)
        else:
            state = _data.dense.Dense(self._ode_solver._y, copy=copy)
        return t, state

    def integrate(self, t, copy=True):
        if t != self._ode_solver.t:
            self._ode_solver.integrate(t)
        return self.get_state(copy)

    def mcstep(self, t, copy=True):
        if self._ode_solver.t == t:
            pass
        elif self._ode_solver.t < t:
            self._back = self._ode_solver.t
            self._ode_solver.integrate(t, step=True)
        elif self._back <= t:
            self._ode_solver.integrate(t)
        else:
            raise IntegratorException(
                f"`t`={t} is outside the integration range: "
                f"{self._back[0]}..{self._ode_solver.t}."
            )
        return self.get_state(copy)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'Excess work done on this call. Try to increasing '
                'the nsteps parameter in the Options class',
            -2: 'Excess accuracy requested. (Tolerances too small.)',
            -3: 'Illegal input detected.',
            -4: 'Repeated error test failures. (Check all input.)',
            -5: 'Repeated convergence failures. (Perhaps bad'
                ' Jacobian supplied or wrong choice of MF or tolerances.)',
            -6: 'Error weight became zero during problem. (Solution'
                ' component i vanished, and ATOL or ATOL(i) = 0.)'
        }
        raise IntegratorException(
            messages[self._ode_solver._integrator.istate]
        )


class IntegratorScipyDop853(Integrator):
    """
    Integrator using Scipy `ode` with dop853 integrator. Eight order
    runge-kutta method by Dormand & Prince. Use fortran implementation
    from [E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary Differential
    Equations i. Nonstiff Problems. 2nd edition. Springer Series in
    Computational Mathematics, Springer-Verlag (1993)].
    """
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 2500,
        'first_step': 0,
        'max_step': 0,
        'ifactor': 6.0,
        'dfactor': 0.3,
        'beta': 0.0,
    }
    support_time_dependant = True
    supports_blackbox = True

    def _prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self._mul_np_vec)
        self._ode_solver.set_integrator('dop853', **self.options)
        self.name = "scipy ode dop853"

    def _mul_np_vec(self, t, vec):
        """
        Interface between scipy which use numpy and the driver, which use data.
        """
        state = _data.dense.fast_from_numpy(vec.view(np.complex128))
        column_unstack_dense(state, self._size, inplace=True)
        out = self.system.matmul_data(t, state)
        column_stack_dense(out, inplace=True)
        return out.as_ndarray().ravel().view(np.float64)

    def integrate(self, t, copy=True):
        if t != self._ode_solver.t:
            self._ode_solver.integrate(t)
        return self.get_state(copy)

    def mcstep(self, t, copy=True):
        if self._ode_solver.t <= t:
            # Scipy's DOP853 does not have a step function.
            # It has a safe step lengh, but can be 0 if unknown.
            dt = self._ode_solver._integrator.work[6]
            if dt:
                t = min(self._ode_solver.t + dt, t)
            self._ode_solver.integrate(t)
        else:
            # While DOP853 support changing the direction of the integration,
            # it does not do so efficiently. We do it manually.
            self._ode_solver._integrator.work[6] *= -1
            self._ode_solver.integrate(t)
            self._ode_solver._integrator.work[6] *= -1
        return self.get_state(copy)

    def get_state(self, copy=True):
        if not self._is_set:
            raise IntegratorException("The state is not initialted")
        self._check_failed_integration()
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.Dense(self._ode_solver._y.view(np.complex128),
                                  copy=copy),
                self._size,
                inplace=True)
        else:
            state = _data.dense.Dense(self._ode_solver._y.view(np.complex128),
                                      copy=copy)
        return t, state

    def set_state(self, t, state0):
        self._is_set = True
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel().view(np.float64),
            t
        )

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'input is not consistent',
            -2: 'larger nsteps is needed, Try to increase the nsteps '
                'parameter in the Options class.',
            -3: 'step size becomes too small. Try increasing tolerance',
            -4: 'problem is probably stiff (interrupted), try "bdf" '
                'method instead',
        }
        raise IntegratorException(
            messages[self._ode_solver._integrator.istate]
        )


class IntegratorScipylsoda(IntegratorScipyDop853):
    """
    Integrator using Scipy `ode` with lsoda integrator. ODE solver by netlib
    (http://www.netlib.org/odepack) Automatically choose between 'Adams' and
    'BDF' methods to solve both stiff and non-stiff systems.
    """
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 2500,
        'max_order_ns': 12,
        'max_order_s': 5,
        'first_step': 0.0,
        'max_step': 0.0,
        'min_step': 0.0,
    }
    support_time_dependant = True
    supports_blackbox = True

    def _prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self._mul_np_vec)
        self._ode_solver.set_integrator('lsoda', **self.options)
        self.name = "scipy lsoda"

    def mcstep(self, t, copy=True):
        if self._ode_solver.t == t:
            pass
        elif self._ode_solver.t < t:
            self._back = self.get_state(copy=True)
            self._one_step(t)
        elif self._back[0] <= t:
            self._backstep(t)
        else:
            raise IntegratorException(
                f"`t`={t} is outside the integration range: "
                f"{self._back[0]}..{self._ode_solver.t}."
            )
        return self.get_state(copy)

    def _one_step(self, t):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        # Here we want to advance up to t doing maximum one step.
        # We check if a new step is needed.
        t_front = self._ode_solver._integrator.rwork[12]
        t_ode = self._ode_solver.t
        if t > t_front and t_ode >= t_front:
            # The state is at t_front, do a step
            self._ode_solver.integrate(t, step=True)
        elif t > t_front:
            # The state is at a time before t_front, advance to t_front
            t = t_front
        else:
            # t is inside the already computed integration range.
            pass
        if t_front >= t:
            self._ode_solver.integrate(t)

    def _backstep(self, t):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        # zvode can integrate with time lower than the most recent time
        # but not over all the step interval.
        # About 90% of the last step interval?
        # Scipy raise a warning, not an Error.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            self.set_state(*self._back)
            self._ode_solver.integrate(t)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: "Excess work done on this call."
                "Try to increase the nsteps parameter in the Options class.",
            -2: "Excess accuracy requested (tolerances too small).",
            -3: "Illegal input detected (internal error).",
            -4: "Repeated error test failures (internal error).",
            -5: "Repeated convergence failures "
                "(perhaps bad Jacobian or tolerances).",
            -6: "Error weight became zero during problem.",
            -7: "Internal workspace insufficient to finish (internal error)."
        }
        raise IntegratorException(
            messages[self._ode_solver._integrator.istate]
        )


Solver.add_integrator(IntegratorScipyZvode, ['adams', 'bdf'])
Solver.add_integrator(IntegratorScipyDop853, 'dop853')
Solver.add_integrator(IntegratorScipylsoda, 'lsoda')