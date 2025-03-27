#cython: language_level=3
from qutip.core.data cimport Data
from qutip.core.cy.qobjevo cimport QobjEvo

cpdef enum Status:
    AT_FRONT = 2
    INTERPOLATED = 1
    NORMAL = 0
    TOO_MUCH_WORK = -1
    DT_UNDERFLOW = -2
    OUTSIDE_RANGE = -3
    NOT_INITIATED = -4

cdef class Explicit_RungeKutta:
    cdef QobjEvo qevo

    # Ode state data, set in set_initial_value
    cdef list k
    cdef Data _y_temp, _y, _y_prev, _y_front
    cdef double _norm_front, _norm_prev, _dt_safe, _dt_int
    cdef double _t, _t_prev, _t_front
    cdef Status _status
    cdef dict status_messages

    # options: set in init
    cdef readonly double rtol, atol, first_step, min_step, max_step
    cdef readonly int max_numsteps
    cdef readonly bint interpolate
    cdef readonly object method

    # Runge Kutta tableau and info
    cdef int rk_step, rk_extra_step, order, denseout_order
    cdef bint adaptative_step, can_interpolate
    cdef object b_factor_np
    cdef double [:] b
    cdef double [:] b_factor
    cdef double [:] c
    cdef double [:] e
    cdef double [:, ::1] a
    cdef double [:, ::1] bi

    cpdef void integrate(Explicit_RungeKutta self, double t, bint step=*) except *

    cpdef void set_initial_value(self, Data y0, double t) except *

    cdef int _step_in_err(self, double t, int max_step) except -1

    cdef double _compute_step(self, double dt) except -1

    cdef double _error(self, Data y_new, double dt) except -1

    cdef void _prep_dense_out(self) except *

    cdef Data _interpolate_step(self, double t, Data out)

    cdef inline Data _accumulate(self, Data target, double[:] factors,
                                 double dt, int size)

    cdef double _estimate_first_step(self, double t, Data y0) except -1

    cdef double _get_timestep(self, double t)

    cdef void _recompute_safe_step(self, double err, double dt)
