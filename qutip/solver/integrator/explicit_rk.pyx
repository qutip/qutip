#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False
"""
Provide a cython implimentation for a general Explicit runge-Kutta method.
"""
from qutip.core.data cimport Data, Dense, CSR, dense
from qutip.core.data.add cimport iadd_dense
from qutip.core.data.add import add
from qutip.core.data.mul import imul_data
from qutip.core.data.tidyup import tidyup_csr
from qutip.core.data.norm import frobenius_data
from .verner7efficient import vern7_coeff
from .verner9efficient import vern9_coeff
from cpython.exc cimport PyErr_CheckSignals
cimport cython
import numpy as np


__all__ = ["Explicit_RungeKutta"]


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


cdef Data copy_to(Data in_, Data out):
    # Copy while reusing allocated buffer if possible.
    # Does not check the shape, etc.
    cdef size_t ptr
    if type(in_) is Dense:
        for ptr in range(in_.shape[0] * in_.shape[1]):
            (<Dense> out).data[ptr] = (<Dense> in_).data[ptr]
        return out
    else:
        return in_.copy()


cdef Data iadd_data(Data left, Data right, double complex factor):
    # left += right * factor
    # reusing `left' allocated buffer if possible.
    # TODO: when/if iadd_csr is added: move to data/add.pyx.
    if factor == 0:
        return left
    if type(left) is Dense:
        iadd_dense(left, right, factor)
        return left
    else:
        return add(left, right, factor)


cdef class Explicit_RungeKutta:
    """
    Qutip implementation of Runge Kutta ODE.
    Works in :class:`.Data` allowing solving using sparse and gpu data.

    Parameters
    ----------
    qevo : QobjEvo
        The system to integrate: ```dX = qevo.matmul_data(t, X)``

    rtol, atol : double
        Relative and absolute tolerance of the integration error respectively.

    nsteps : int
        Maximum number of steps done during one call of integrate.

    first_step : double
        Lenght in ``t`` of the first step. If ``0``, an appropriate step will
        be determined automaticaly.

    min_step, max_step : double
        Bounds of the step lenght in ``t``. ``0`` means no bounds.

    interpolate : bool
        Whether to do longer step and interpolate back when the method support
        it.
        Example:
            At ``t=1``, the integrator is asked to integrate to ``t=2`` but can
            safely integrate up to ``t=3``. If ``interpolate=True``, it will
            integrate up to ``t=3`` and then use interpolation to get the state
            at ``t=2``. With ``interpolate=False``, it will integrate to
            ``t=2``.

    method : ['euler', 'rk4', 'vern7', 'vern9']
        The integration method to use.
        * It also accept a tuple of runge-kutta coefficients.
        (See `Explicit_RungeKutta._init_coeff`)
    """
    def __init__(self, QobjEvo qevo, double rtol=1e-6, double atol=1e-8,
                 int nsteps=1000, double first_step=0, double min_step=0,
                 double max_step=0, bint interpolate=True, method="euler"):
        # Function to integrate.
        self.qevo = qevo
        # tolerances
        self.atol = atol
        self.rtol = rtol
        self._dt_safe = atol
        # Number of step to do before giving up
        self.max_numsteps = nsteps

        # contrain on the step times
        self.first_step = first_step
        self.min_step = min_step or 1e-15
        self.max_step = max_step

        # If True, will do the steps longer that the target and use
        # interpolation to get the state at the wanted time.
        self.interpolate = interpolate

        self.k = []
        if isinstance(method, dict):
            self._init_coeff(**method)
        elif "vern7" == method:
            self._init_coeff(**vern7_coeff)
        elif "vern9" == method:
            self._init_coeff(**vern9_coeff)
        elif "rk4" == method:
            self._init_coeff(**rk4_coeff)
        else:
            self._init_coeff(**euler_coeff)
        self.method = method
        self._y_prev = None

    def _init_coeff(self, order, a, b, c, e=None, bi=None):
        """
        Read the Butcher tableau.

        Parameters
        ----------
        order : int
            Order of the method

        a, b, c : numpy.ndarray
            The Butcher tableau:
            c_0 | a_00 a_01
            c_1 | a_10 a_11
            ---------------
                  b_0  b_1
            The ``c`` vector should start with `0` and the first line of ``a``
            is also expected to be zeros.
            The ``a`` and ``c`` part of the tableau can be larger than the
            number of step if dense output is available.

        e : numpy.ndarray (optional)
            The error coefficients. If given, adaptative step length will be
            used to stay within the given tolerance.
            In some notation 2 sets of b are given, then ``e_i = b_i - b*_i``

        bi : numpy.ndarray (optional)
            The coefficients for the dense output.
            The dense output is computed as:
                ``k[i] * bi[i,j] * ((t-t0) / (t'-t0))**j``
            with ``k[i]`` the derivative computed from ``a`` and ``c``.
        """
        self.order = order
        self.rk_step = b.shape[0]
        self.rk_extra_step = c.shape[0]
        if (
            self.rk_step > self.rk_extra_step or
            a.shape[1] != self.rk_extra_step or
            a.shape[0] != self.rk_extra_step
        ):
            raise ValueError("Inconsistant shape between the Butcher tableau "
                             "parts.")

        self.adaptative_step = e is not None
        if self.adaptative_step and e.shape[0] != self.rk_step:
            raise ValueError("The length of the error coefficients must be the"
                             " same as the number of steps in the Butcher "
                             f"tableau. Got {e.shape[0]} but expected "
                             f"{self.rk_step}")

        self.interpolate = bi is not None and self.interpolate
        if self.interpolate:
            self.denseout_order = bi.shape[1]
            if bi.shape[0] != self.rk_extra_step:
                raise ValueError("The interpolation coefficient's shape must "
                                 "be (a.shape[0], dense output order)")

        self.a = a
        self.b = b
        self.c = c
        self.e = e
        self.bi = bi
        self.b_factor_np = np.empty(self.rk_extra_step, dtype=np.float64)
        self.b_factor = self.b_factor_np

    def __reduce__(self):
        """
        Helper for pickle to serialize the object
        """
        return (self.__class__, (
            self.qevo, self.rtol, self.atol, self.max_numsteps, self.first_step,
            self.min_step, self.max_step, self.interpolate, self.method
        ))

    cpdef void set_initial_value(self, Data y0, double t) except *:
        """
        Set the initial state and time of the integration.
        """
        self._t = t
        self._t_prev = t
        self._t_front = t
        self._dt_int = 0
        self._y = y0
        self._norm_prev = frobenius_data(self._y)
        self._norm_front = self._norm_prev

        #prepare the buffers
        for i in range(self.rk_extra_step):
            self.k.append(self._y.copy())
        self._y_temp = self._y.copy()
        self._y_front = self._y.copy()
        self._y_prev = self._y.copy()

        if not self.first_step:
            self._dt_safe = self._estimate_first_step(t, self._y)
        else:
            self._dt_safe = self.first_step

    cdef double _estimate_first_step(self, double t, Data y0) except -1:
        if not self.adaptative_step:
            return 0.

        cdef double dt1, dt2, dt, factorial = 1, t1
        cdef double norm = frobenius_data(y0), tmp_norm
        cdef double tol = self.atol + norm * self.rtol
        cdef int i
        self.k[0] = imul_data(<Data> self.k[0], 0)
        self.k[0] = self.qevo.matmul_data(t, y0, <Data> self.k[0])

        # Ok approximation for linear system. But not in a general case.
        if norm <= self.atol:
            norm = 1
        tmp_norm = frobenius_data(<Data> self.k[0])
        for i in range(1, self.order+1):
            factorial *= i
        if tmp_norm >= (self.atol * 1e-6):
            dt1 = ((tol * factorial * norm**self.order)**(1 / (self.order+1))
                   / tmp_norm)
        else:
            dt1 = (tol * factorial)**(1 / (self.order+1)) * norm * 0.5

        t1 = t + dt1 / 100
        self._y_temp = copy_to(y0, self._y_temp)
        # below dt1 / 100 is an arbitrary small fraction of dt1:
        self._y_temp = iadd_data(self._y_temp, <Data> self.k[0], dt1 / 100)
        self.k[1] = imul_data(<Data> self.k[1], 0)
        self.k[1] = self.qevo.matmul_data(t1, self._y_temp, <Data> self.k[1])
        tmp_norm = frobenius_data(<Data> self.k[1])
        if tmp_norm >= (self.atol * 1e-6):
            dt2 = ((tol * factorial * norm**self.order)**(1 / (self.order+1))
                   / tmp_norm)
        else:
            dt2 = dt1

        dt = min(dt1, dt2)
        if self.max_step:
            dt = min(self.max_step, dt)
        if self.min_step:
            dt = max(self.min_step, dt)
        return dt

    cpdef void integrate(Explicit_RungeKutta self, double t, bint step=False) except *:
        """
        Do the integration to t.
        If ``step`` is True, it will make a maximum 1 step and may not reach
        ``t``.
        """
        cdef int nsteps_left = self.max_numsteps
        cdef double err = 0

        if self._y_prev is None:
            self._status = Status.NOT_INITIATED
            return

        self._status = Status.NORMAL

        if t == self._t:
            return

        if t < self._t_prev:
            self._status = Status.OUTSIDE_RANGE
            return

        if self.interpolate and t < self._t_front:
             self._y = self._interpolate_step(t, self._y)
             self._t = t

        if step and self._t < self._t_front and t > self._t_front:
            # To ensure that the self._t ... t_out interval can be covered.
            t = self._t_front

        while self._t_front < t:
            self._y_prev = copy_to(self._y_front, self._y_prev)
            self._t_prev = self._t_front
            self._norm_prev = self._norm_front
            nsteps_left -= self._step_in_err(t, nsteps_left)
            PyErr_CheckSignals()
            if step:
                break

        if self._status < 0:
            return

        if self._t_front > t:
            self._prep_dense_out()
            self._status = Status.INTERPOLATED
            self._t = t
            self._y = self._interpolate_step(t, self._y)
        else:
            self._status = Status.AT_FRONT
            self._t = self._t_front
            self._y = copy_to(self._y_front, self._y)

    cdef int _step_in_err(self, double t, int max_step) except -1:
        """
        Do compute one step, repeating until the error is within tolerance.
        """
        cdef double error = 1
        cdef int nsteps = 0
        while error >= 1:
            dt = self._get_timestep(t)
            error = self._compute_step(dt)
            self._dt_int = dt
            self._recompute_safe_step(error, dt)

            if dt == self.min_step and error > 1:
                # The tolerance was not reached but the dt is at the minimum.
                self._status = Status.DT_UNDERFLOW
                break
            nsteps += 1
            if nsteps > max_step:
                self._status = Status.TOO_MUCH_WORK
                break
        return nsteps

    cdef double _compute_step(self, double dt) except -1:
        """
        Do compute one step with fixed ``dt``, return the error.
        Use (_t_prev, _y_prev) to create (_t_front, _y_front)
        """
        cdef int i
        for i in range(self.rk_step):
            self.k[i] = imul_data(<Data> self.k[i], 0.)

        # Compute the derivatives
        self.k[0] = self.qevo.matmul_data(self._t_prev, self._y_prev,
                                          <Data> self.k[0])
        for i in range(1, self.rk_step):
            self._y_temp = copy_to(self._y_prev, self._y_temp)
            self._y_temp = self._accumulate(self._y_temp, self.a[i,:], dt, i)
            self.k[i] = self.qevo.matmul_data(self._t_prev + self.c[i]*dt,
                                              self._y_temp, <Data> self.k[i])

        # Compute the state
        self._y_front = copy_to(self._y_prev, self._y_front)
        self._y_front = self._accumulate(self._y_front, self.b, dt,
                                         self.rk_step)
        self._t_front = self._t_prev + dt

        if type(self._y_front) is CSR:
            # issparse() test would be better.
            tidyup_csr(self._y_front, self.atol/self._y_front.shape[0], True)

        return self._error(self._y_front, dt)

    cdef double _error(self, Data y_new, double dt) except -1:
        """ Compute the normalized error. (error/tol) """
        if not self.adaptative_step:
            return 0.
        self._y_temp = imul_data(self._y_temp, 0.)
        self._y_temp = self._accumulate(self._y_temp, self.e, dt, self.rk_step)
        self._norm_front = frobenius_data(y_new)
        return frobenius_data(self._y_temp) / (self.atol +
                max(self._norm_prev, self._norm_front) * self.rtol)

    cdef void _prep_dense_out(self) except *:
        """
        Compute derivative for the interpolation step.
        """
        cdef double dt = self._dt_int

        for i in range(self.rk_step, self.rk_extra_step):
            self.k[i] = imul_data(<Data> self.k[i], 0.)
            self._y_temp = copy_to(self._y_prev, self._y_temp)
            self._y_temp = self._accumulate(self._y_temp, self.a[i,:], dt, i)
            self.k[i] = self.qevo.matmul_data(self._t_prev + self.c[i]*dt,
                                              self._y_temp, <Data> self.k[i])

    cdef Data _interpolate_step(self, double t, Data out):
        """
        Compute the state at any points between _t_prev and _t_front
        with an error of ~dt**denseout_order.
        """
        cdef:
            int i, j
            double t0 = self._t_prev
            double dt = self._dt_int
            double tau = (t - t0) / dt
        for i in range(self.rk_extra_step):
            self.b_factor[i] = 0.
            for j in range(self.denseout_order-1, -1, -1):
                self.b_factor[i] += self.bi[i,j]
                self.b_factor[i] *= tau

        out = copy_to(self._y_prev, out)
        out = self._accumulate(out, self.b_factor, dt, self.rk_extra_step)
        return out

    cdef inline Data _accumulate(self, Data target, double[:] factors,
                         double dt, int size):
        cdef int i
        for i in range(size):
            target = iadd_data(target, <Data> self.k[i], dt * factors[i])
        return target

    @cython.cdivision(True)
    cdef double _get_timestep(self, double t):
        """ Get the dt for the step. """
        cdef double dt_needed = t - self._t_front
        if not self.adaptative_step:
            return dt_needed
        if self.interpolate:
            return self._dt_safe
        if dt_needed <= self._dt_safe:
            return dt_needed
        return dt_needed / (int(dt_needed / self._dt_safe) + 1)

    cdef void _recompute_safe_step(self, double err, double dt):
        """ Get maximum safe step in function of the error."""
        cdef double factor
        if not self.adaptative_step:
            return
        elif err == 0:
            factor = 10
        else:
            factor = 0.9*err**(-1/(self.order+1))
            factor = min(10, factor)
            factor = max(0.2, factor)

        self._dt_safe = dt * factor
        if self.max_step:
            self._dt_safe = min(self.max_step, self._dt_safe)
        if self.min_step:
            self._dt_safe = max(self.min_step, self._dt_safe)

    def successful(self):
        return self._status >= 0

    @property
    def status(self):
        return self._status

    def status_message(self):
        "Status of the last step in a human readable format."
        return {
            AT_FRONT: "Internal state at the desired time.",
            INTERPOLATED: (
                "Internal state past the desired time and "
                "interpolation to step done."),
            NORMAL: 'No work done.',
            TOO_MUCH_WORK: (
                'Too much work done in one call. Try to increase '
                'the nsteps parameter or increasing the tolerance.'
            ),
            DT_UNDERFLOW:
                'Step size becomes too small. Try increasing tolerance.',
            OUTSIDE_RANGE: 'Step outside available range.',
            NOT_INITIATED: 'Not initialized.'
        }[self._status]

    @property
    def y(self):
        return self._y

    @property
    def y_prev(self):
        return self._y_prev

    @property
    def y_front(self):
        return self._y_front

    @property
    def t_front(self):
        return self._t_front

    @property
    def t_prev(self):
        return self._t_prev

    @property
    def t(self):
        return self._t

    def _debug_state(self):
        print("t, y: ", self._t,
              'None' if self._y is None else self._y.to_array())
        print("t_prev, y_prev, |y_prev|", self._t_prev,
              'None' if self._y_prev is None else self._y_prev.to_array(),
              self._norm_prev)
        print("t_front, y_front, |y_front|", self._t_front,
              'None' if self._y_front is None else self._y_front.to_array(),
              self._norm_front)
        print("y_temp",
              'None' if self._y_temp is None else self._y_temp.to_array())
        print("dt_safe: ", self._dt_safe, "dt_int: ", self._dt_int)
        for i in range(self.rk_extra_step):
            print(f'k[{i}]',
                  'None' if self.k[i] is None else self.k[i].to_array())
