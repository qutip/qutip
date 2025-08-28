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


cdef class RKStats:
    cdef int loglevel, rk_step, rk_extra_step
    cdef int num_step_total, num_step_failed, num_step_success
    cdef int num_interpolation_step, num_interpolation_preparation
    cdef int num_derr_computation

    cdef double max_success_dt, min_success_dt, avg_success_dt
    cdef double max_failed_dt, min_failed_dt, avg_failed_dt
    cdef double max_success_error, min_success_error, avg_success_error
    cdef double max_failed_error, min_failed_error, avg_failed_error
    cdef double max_success_safe_dt, min_success_safe_dt, avg_success_safe_dt
    cdef double max_failed_safe_dt, min_failed_safe_dt, avg_failed_safe_dt
    cdef dict full_step_data

    cdef void log_success_step(self, double t, double dt, double error, double safe_dt)
    cdef void log_failed_step(self, double t, double dt, double error, double safe_dt)
    cdef void log_step(self, double t, double dt, double error, double safe_dt)
    cdef void increment_interpolation_step(self)
    cdef void increment_interpolation_preparation(self)


cdef class Explicit_RungeKutta:
    cdef QobjEvo qevo

    # Ode state data, set in set_initial_value
    cdef list k
    cdef Data _y_temp, _y, _y_prev, _y_front, _k_fsal
    cdef double _dt_safe, _dt_int
    cdef double _t, _t_prev, _t_front
    cdef Status _status
    cdef dict status_messages
    cdef RKStats statistics

    # options: set in init
    cdef readonly double rtol, atol, first_step, min_step, max_step
    cdef readonly int max_numsteps
    cdef readonly bint interpolate
    cdef readonly int loglevel

    # Runge Kutta tableau and info
    cdef int rk_step, rk_extra_step, order, denseout_order
    # Whether variable step according to tolerance are supported
    cdef bint adaptative_step
    # Whether the output can be computed everywhere along the step
    cdef bint can_interpolate
    # Whether the first derivative is the same as the last one of the prevous step
    cdef bint first_same_as_last
    cdef object b_factor_np
    cdef dict butcher_tableau
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
