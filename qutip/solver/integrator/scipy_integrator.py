"""ODE integrator from scipy."""

__all__ = [
    'IntegratorScipyAdams',
    'IntegratorScipyBDF',
    'IntegratorScipyDop853',
    'IntegratorScipylsoda',
]

import numpy as np
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from qutip.core import data as _data
from qutip.core.data.reshape import column_unstack_dense, column_stack_dense
from ..integrator import IntegratorException, Integrator
from ..solver_base import Solver
import warnings


class IntegratorScipyAdams(Integrator):
    """
    Integrator using Scipy `ode` with zvode integrator using adams method.
    Ordinary Differential Equation solver by netlib
    (https://www.netlib.org/odepack).

    Usable with ``method="adams"``
    """
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 2500,
        'order': 12,
        'first_step': 0,
        'max_step': 0,
        'min_step': 0,
    }
    support_time_dependant = True
    supports_blackbox = True
    method = 'adams'

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
        self._ode_solver._integrator = self._zvode(
            method=self.method,
            **self.options,
        )
        self.name = "scipy zvode " + self.method

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
        self._front = t
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        if self._mat_state:
            state0 = _data.column_stack(state0)
        self._ode_solver.set_initial_value(state0.to_array().ravel(), t)

    def get_state(self, copy=True):
        if not self._is_set:
            raise IntegratorException("The state is not initialted")
        self._check_failed_integration()
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._ode_solver._y),
                self._size,
                inplace=True)
        else:
            state = _data.dense.fast_from_numpy(self._ode_solver._y)
        return t, state

    def _check_handle(self):
        """
        Do the check for concurrent use of the integrator and reset if used
        elsewhere.
        """
        integrator = self._ode_solver._integrator
        if integrator.initialized:
            if integrator.handle != integrator.__class__.active_global_handle:
                integrator.reset(len(self._ode_solver._y), False)

    def integrate(self, t, copy=True):
        self._check_handle()
        if t != self._ode_solver.t:
            self._ode_solver.integrate(t)
        return self.get_state(copy)

    def mcstep(self, t, copy=True):
        # When working with mcstep, we use the dense output feature:
        # a range in which the state at any time can be computed with
        # minimal work. We keep track of the range with _back and _front.
        self._check_handle()
        if self._ode_solver.t == t:
            # Exact same `t` as the last call, nothing to do.
            pass
        elif self._back > t:
            # `t` before the range: not supported.
            raise IntegratorException(
                f"`t`={t} is behind the integration limit: "
                f"{t} < {self._back}."
            )
        elif t <= self._front:
            # In the range, ask for the new state.
            self._ode_solver.integrate(t)
        elif self._ode_solver.t < self._front:
            # `t` after the range but last call (`t_prev`) not at the front.
            # Advancing the range would make the interval `t_prev`..`_front`
            # unreachable. Thus advance to _front.
            self._ode_solver.integrate(self._front)
        else:
            # Advance the range.
            self._back = self._front
            self._ode_solver.integrate(t, step=True)
            self._front = self._ode_solver._integrator.rwork[12]

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

    @property
    def options(self):
        """
        Supported options by zvode integrator:

        atol : float, default: 1e-8
            Absolute tolerance.

        rtol : float, default: 1e-6
            Relative tolerance.

        order : int, default: 12, 'adams' or 5, 'bdf'
            Order of integrator <=12 'adams', <=5 'bdf'

        nsteps : int, default: 2500
            Max. number of internal steps/call.

        first_step : float, default: 0
            Size of initial step (0 = automatic).

        min_step : float, default: 0
            Minimum step size (0 = automatic).

        max_step : float, default: 0
            Maximum step size (0 = automatic)
            When using pulses, change to half the thinest pulse otherwise it
            may be skipped.
    """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


class IntegratorScipyBDF(IntegratorScipyAdams):
    """
    Integrator using Scipy `ode` with zvode integrator using bdf method.
    Ordinary Differential Equation solver by netlib
    (https://www.netlib.org/odepack).

    Usable with ``method="bdf"``
    """
    method = 'bdf'
    integrator_options = {
        'atol': 1e-8,
        'rtol': 1e-6,
        'nsteps': 2500,
        'order': 5,
        'first_step': 0,
        'max_step': 0,
        'min_step': 0,
    }


class IntegratorScipyDop853(Integrator):
    """
    Integrator using Scipy `ode` with dop853 integrator. Eight order
    runge-kutta method by Dormand & Prince. Use fortran implementation
    from [E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary Differential
    Equations i. Nonstiff Problems. 2nd edition. Springer Series in
    Computational Mathematics, Springer-Verlag (1993)].

    Usable with ``method="dop853"``
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
    method = 'dop853'

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
            # It has a safe step length, but can be 0 if unknown.
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
                _data.dense.fast_from_numpy(
                    self._ode_solver._y.view(np.complex128)
                ),
                self._size,
                inplace=True)
        else:
            state = _data.dense.fast_from_numpy(
                self._ode_solver._y.view(np.complex128)
            )
            state = state.copy() if copy else state
        return t, state

    def set_state(self, t, state0):
        self._is_set = True
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        if self._mat_state:
            state0 = _data.column_stack(state0)
        self._ode_solver.set_initial_value(
            state0.to_array().ravel().view(np.float64),
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

    @property
    def options(self):
        """
        Supported options by dop853 integrator:

        atol : float, default: 1e-8
            Absolute tolerance.

        rtol : float, default: 1e-6
            Relative tolerance.

        nsteps : int, default: 2500
            Max. number of internal steps/call.

        first_step : float, default: 0
            Size of initial step (0 = automatic).

        max_step : float, default: 0
            Maximum step size (0 = automatic)

        ifactor, dfactor : float, default: 6., 0.3
            Maximum factor to increase/decrease step size by in one step

        beta : float, default: 0
            Beta parameter for stabilised step size control.

        See scipy.integrate.ode ode for more detail
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


class IntegratorScipylsoda(IntegratorScipyDop853):
    """
    Integrator using Scipy `ode` with lsoda integrator. ODE solver by netlib
    (https://www.netlib.org/odepack) Automatically choose between 'Adams' and
    'BDF' methods to solve both stiff and non-stiff systems.

    Usable with ``method="lsoda"``
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
    method = 'lsoda'

    def _prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self._mul_np_vec)
        self._ode_solver.set_integrator('lsoda', **self.options)
        self.name = "scipy lsoda"

    def _check_handle(self):
        """
        Do the check for concurrent use of the integrator and reset if used
        elsewhere.
        """
        integrator = self._ode_solver._integrator
        if integrator.initialized:
            if integrator.handle != integrator.__class__.active_global_handle:
                integrator.reset(len(self._ode_solver._y), False)

    def integrate(self, t, copy=True):
        self._check_handle()
        return super().integrate(t, copy)

    def set_state(self, t, state0):
        self._front = t
        super().set_state(t, state0)
        self._back = self.get_state()

    def mcstep(self, t, copy=True):
        # This solver support dense output:
        # a range in which the state at any time can be computed with
        # minimal work. We keep track of the range with _back[0] and _front.
        self._check_handle()
        if self._ode_solver.t == t:
            # Exact same `t` as the last call, nothing to do.
            pass
        elif self._back[0] <= t <= self._front:
            # In the range, ask for the new state.
            self._backstep(t)
        elif self._front < t:
            # Advance the range.
            self._one_step(t)
        else:
            raise IntegratorException(
                f"`t`={t} is behind the integration limit: "
                f"{t} < {self._back[0]}."
            )
        return self.get_state(copy)

    def _one_step(self, t):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        # Here we want to advance up to t doing maximum one step.
        # lsoda officially support step, but sometime it does more work than
        # needed, so we ask it to advance a fraction of the last step, where it
        # will advance one internal step of length allowed by the tolerance and
        # interpolate back to the asked time, effictively getting the single
        # integration step we want. The first step and abrupt changes in the
        # `rhs` can cause exceptions to this, but _backstep catch those cases.
        safe_delta = self._ode_solver._integrator.rwork[11]/100 + 1e-15
        t_ode = self._ode_solver.t

        if t > self._front and t_ode >= self._front:
            # The state is at self._front, do a step
            self._back = self.get_state()
            self._ode_solver.integrate(min(self._front + safe_delta, t))
            self._front = self._ode_solver._integrator.rwork[12]
            # We asked for a fraction of a step, now complete it.
            self._ode_solver.integrate(min(self._front, t))
        elif t > self._front:
            # The state is at a time before t_front, advance to t_front
            self._ode_solver.integrate(self._front)

    def _backstep(self, t):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        delta = self._ode_solver._integrator.rwork[11]
        if t == self._ode_solver.t:
            pass
        elif self._front - delta <= t:
            self._ode_solver.integrate(t)
        else:
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

    @property
    def options(self):
        """
        Supported options by lsoda integrator:

        atol : float, default: 1e-8
            Absolute tolerance.

        rtol : float, default: 1e-6
            Relative tolerance.

        nsteps : int, default: 2500
            Max. number of internal steps/call.

        max_order_ns : int, default: 12
            Maximum order used in the nonstiff case (<= 12).

        max_order_s : int, default: 5
            Maximum order used in the stiff case (<= 5).

        first_step : float, default: 0
            Size of initial step (0 = automatic).

        max_step : float, default: 0
            Maximum step size (0 = automatic)
            When using pulses, change to half the thinest pulse otherwise it
            may be skipped.

        min_step : float, default: 0
            Minimum step size (0 = automatic)
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


Solver.add_integrator(IntegratorScipyBDF, 'bdf')
Solver.add_integrator(IntegratorScipyAdams, 'adams')
Solver.add_integrator(IntegratorScipyDop853, 'dop853')
Solver.add_integrator(IntegratorScipylsoda, 'lsoda')
