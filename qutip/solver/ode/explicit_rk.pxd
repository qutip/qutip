#cython: language_level=3
from qutip.core.data cimport Data
from qutip.core.cy.qobjevo cimport QobjEvo

cdef class Explicit_RungeKutta:
    cdef QobjEvo qevo

    # Ode state data, set in set_initial_value
    cdef list k
    cdef Data _y_temp, _y, _y_prev, _y_front
    cdef double norm_front, norm_tmp, _t, _t_prev, _t_front, dt_safe, dt_int
    cdef int _status

    # options: set in init
    cdef double rtol, atol, first_step, min_step, max_step
    cdef int max_numsteps
    cdef bint interpolate
    cdef str method

    # runge Kutta tableau and info, set in cinit
    cdef int rk_step, rk_extra_step,  order, denseout_order
    cdef bint adaptative_step, can_interpolate
    cdef object b_factor_np
    cdef double [:] b
    cdef double [:] b_factor
    cdef double [:] c
    cdef double [:] e
    cdef double [:,::1] a
    # dense out factors: set in cinit
    cdef double [:,::1] bi

    cpdef integrate(Explicit_RungeKutta self, double t, bint step=*)

    cpdef void set_initial_value(self, Data y0, double t)

    cdef double compute_step(self, double dt)

    cdef double error(self, Data y_new, double dt)

    cdef void prep_dense_out(self)

    cdef Data interpolate_step(self, double t, Data out)

    cdef Data accumulate(self, Data target, double[:] factors,
                         double dt, int size)

    cdef double estimate_first_step(self, double t, Data y0)

    cdef double get_timestep(self, double t)

    cdef double recompute_safe_step(self, double err, double dt)
