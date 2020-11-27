#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False
from .wrapper cimport QtOdeData, QtOdeFuncWrapper

cdef class Explicit_RungeKutta:
    cdef QtOdeFuncWrapper f

    # Ode state data, set in set_initial_value
    cdef list k
    cdef QtOdeData _y_temp, _y, _y_prev, _y_front
    cdef double norm_front, norm_tmp, t, t_prev, t_front, dt_safe, dt_int
    cdef bint failed

    # options: set in init
    cdef double rtol, atol, first_step, min_step, max_step
    cdef int max_numsteps
    cdef bint interpolate

    # runge Kutta tableau and info, set in cinit
    cdef int rk_step, rk_extra_step,  order, denseout_order
    cdef bint adaptative_step, can_interpolate
    cdef double *b
    cdef double *c
    cdef double *e
    cdef double **a
    # dense out factors: set in cinit
    cdef double **bi
    # buffer for dense out: set in cinit
    cdef double *b_factor

    cpdef integrate(Explicit_RungeKutta self, double t, bint step=*)

    cpdef void set_initial_value(self, y0, double t)

    cdef double compute_step(self, double dt, QtOdeData out)

    cdef double eigen_est(self)

    cdef double error(self, QtOdeData y_new, double dt)

    cdef void prep_dense_out(self)

    cdef void interpolate_step(self, double t, QtOdeData out)

    cdef void accumulate(self, QtOdeData target, double *factors,
                         double dt, int size)

    cdef double estimate_first_step(self, double t, QtOdeData y0)

    cdef double get_timestep(self, double t)

    cdef double recompute_safe_step(self, double err, double dt)
