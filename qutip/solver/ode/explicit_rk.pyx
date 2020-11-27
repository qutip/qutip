#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False
from .wrapper cimport QtOdeData, QtOdeFuncWrapper
from .wrapper import qtodedata
cimport cython
cimport numpy as cnp
cdef extern from *:
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_FREE(void *ptr)

cnp.import_array()


cdef class Explicit_RungeKutta:
    def __cinit__(Explicit_RungeKutta self):
        self.rk_step = 1
        self.rk_extra_step = 1
        self.order = 1
        self.denseout_order = 1
        self.can_interpolate = False
        self.adaptative_step = False

        self.b = <double*> PyDataMem_NEW(sizeof(double))
        self.b[0] = 1.
        self.c = <double*> PyDataMem_NEW_ZEROED(1, sizeof(double))
        self.e = <double*> PyDataMem_NEW_ZEROED(1, sizeof(double))
        self.b_factor = <double*> PyDataMem_NEW_ZEROED(1, sizeof(double))
        self.a = <double**> PyDataMem_NEW_ZEROED(self.rk_extra_step,
                                                 sizeof(double*))
        self.bi = <double**> PyDataMem_NEW_ZEROED(self.rk_extra_step,
                                                  sizeof(double*))
        for i in range(self.rk_extra_step):
            self.a[i] = <double*> PyDataMem_NEW_ZEROED(i, sizeof(double))
            self.bi[i] = <double*> PyDataMem_NEW_ZEROED(self.denseout_order,
                                                        sizeof(double))

    def __dealloc__(self):
        cdef int i
        PyDataMem_FREE(self.b)
        PyDataMem_FREE(self.c)
        PyDataMem_FREE(self.e)
        PyDataMem_FREE(self.b_factor)
        for i in range(self.rk_extra_step):
            PyDataMem_FREE(self.a[i])
            PyDataMem_FREE(self.bi[i])
        PyDataMem_FREE(self.a)
        PyDataMem_FREE(self.bi)

    def __init__(Explicit_RungeKutta self, QtOdeFuncWrapper f,
                 rtol=1e-6, atol=1e-8, nsteps=1000,
                 first_step=0, min_step=0, max_step=0, interpolate=True):
        self.f = f
        self.atol = atol
        self.rtol = rtol
        self.max_numsteps = nsteps
        self.first_step = first_step
        self.min_step = min_step
        self.max_step = max_step
        self.interpolate = self.can_interpolate and interpolate
        self.k = []
        self.dt_safe = atol

    cpdef integrate(Explicit_RungeKutta self, double t, bint step=False):
        cdef int nsteps = 0
        cdef double err = 0
        while self.t_front < t and nsteps < self.max_numsteps:
            nsteps += 1
            if err < 1:
                self._y_prev.copy(self._y_front)
                self.t_prev = self.t_front
            dt = self.get_timestep(t)
            err = self.compute_step(dt, self._y_front)
            self.recompute_safe_step(err, dt)
            if err < 1:
                self.dt_int = dt
                self.t_front = self.t_prev + dt
                self.norm_front = self.norm_tmp
                if step:
                    break
        if self.t_front < t - 1e-15 and not step:
            self.failed = True
        elif self.t_front > t + 1e-15:
            self.prep_dense_out()
            self.t = t
            self.interpolate_step(t, self._y)
        else:
            self.t = self.t_front
            self._y.copy(self._y_front)

    def successful(self):
        return not self.failed

    def state(self):
        print(self.t, self.t_prev, self.t_front)
        print(self.dt_int, self.dt_safe)
        print(self.y, self._y_prev.raw())
        print(self.norm_tmp, self.norm_front)
        print(self.eigen_est())

    def print_table(self):
        for i in range(self.rk_step):
            print("c",i,self.c[i])
        for i in range(self.rk_step):
            print("b",i,self.b[i])
        for i in range(self.rk_step):
            print("e",i,self.e[i])
        for i in range(self.rk_extra_step):
            for j in range(i):
                print("a",i,j,self.a[i][j])

    cpdef void set_initial_value(self, y0, double t):
        self.t = t
        self.t_prev = t
        self.t_front = t
        self.dt_int = 0
        self._y = qtodedata(y0)
        self.norm_tmp = self._y.norm()
        self.norm_front = self.norm_tmp
        self.failed = False

        #prepare_buffer
        for i in range(self.rk_extra_step):
            self.k.append(self._y.empty_like())
        self._y_temp = self._y.empty_like()
        self._y_front = self._y.empty_like()
        self._y_front.copy(self._y)
        self._y_prev = self._y.empty_like()

        if not self.first_step:
            self.dt_safe = self.estimate_first_step(t, self._y)
            print("estimated")
        else:
            self.dt_safe = self.first_step
        print(self.dt_safe)

    cdef double compute_step(self, double dt, QtOdeData out):
        cdef int i, j
        cdef double t = self.t_front

        for i in range(self.rk_step):
            (<QtOdeData> self.k[i]).zero()

        self.f.call(self.k[0], t, self._y_prev)

        for i in range(1, self.rk_step):
            self._y_temp.copy(self._y_prev)
            self.accumulate(self._y_temp, self.a[i], dt, i)
            self.f.call((<QtOdeData> self.k[i]), t + self.c[i]*dt, self._y_temp)

        out.copy(self._y_prev)
        self.accumulate(out, self.b, dt, self.rk_step)

        return self.error(out, dt)

    cdef double eigen_est(self):
        return 0

    cdef double error(self, QtOdeData y_new, double dt):
        cdef int j
        self._y_temp.zero()
        self.accumulate(self._y_temp, self.e, dt, self.rk_step)
        self.norm_tmp = y_new.norm()
        return self._y_temp.norm() / (self.atol +
                max(self.norm_tmp, self.norm_front) * self.rtol)

    cdef void prep_dense_out(self):
        cdef:
            double t = self.t_prev
            double dt = self.dt_int

        for i in range(self.rk_step, self.rk_extra_step):
            (<QtOdeData> self.k[i]).zero()
            self._y_temp.copy(self._y_prev)
            for j in range(i):
                if self.a[i][j]:
                    self._y_temp.inplace_add((<QtOdeData> self.k[j]),
                                        dt * self.a[i][j])
            self.f.call(self.k[i], t + dt * self.c[i], self._y_temp)

    cdef void interpolate_step(self, double t, QtOdeData out):
        cdef:
            int i, j, num_k = self.rk_extra_step
            double t0 = self.t_prev
            double dt = self.dt_int
            double tau = (t - t0) / dt
        for i in range(self.rk_extra_step):
            self.b_factor[i] = 0.
            for j in range(self.denseout_order-1, -1, -1):
                self.b_factor[i] += self.bi[i][j]
                self.b_factor[i] *= tau
        out.copy(self._y_prev)
        self.accumulate(out, self.b_factor, dt, self.rk_extra_step)

    cdef void accumulate(self, QtOdeData target, double *factors,
                         double dt, int size):
        cdef int i
        for i in range(size):
            if factors[i]:
                target.inplace_add((<QtOdeData> self.k[i]), dt * factors[i])

    cdef double estimate_first_step(self, double t, QtOdeData y0):
        if not self.adaptative_step:
            return 0.

        cdef double tol = self.atol + y0.norm() * self.rtol
        cdef double dt1, dt2, dt, factorial = 1
        cdef double norm = y0.norm(), tmp_norm
        cdef int i
        (<QtOdeData> self.k[0]).zero()
        self.f.call((<QtOdeData> self.k[0]), t, y0)
        # Good approximation for linear system. But not in a general case.
        if norm == 0:
            norm = 1
        tmp_norm = (<QtOdeData> self.k[0]).norm()
        for i in range(1, self.order+1):
            factorial *= i
        if tmp_norm != 0:
            dt1 = (tol*factorial*norm**self.order)**(1/(self.order+1)) / tmp_norm
        else:
            dt1 = (tol*factorial*norm**self.order)**(1/(self.order+1))
        self._y_temp.copy(y0)
        self._y_temp.inplace_add((<QtOdeData> self.k[0]), dt1 / 100)
        (<QtOdeData> self.k[1]).zero()
        self.f.call((<QtOdeData> self.k[1]), t + dt1 / 100, self._y_temp)
        (<QtOdeData> self.k[1]).inplace_add(self.k[0], -1)
        tmp_norm = (<QtOdeData> self.k[1]).norm()
        if tmp_norm != 0:
            dt2 = ((tol * factorial* norm**(self.order//2))**(1/(self.order+1)) /
                   (tmp_norm / dt1 * 100)**0.5)
        else:
            dt2 = 0.1
        dt = min(dt1, dt2)
        if self.max_step:
            dt = min(self.max_step, dt)
        if self.min_step:
            dt = max(self.min_step, dt)
        return dt

    @cython.cdivision(True)
    cdef double get_timestep(self, double t):
        if not self.adaptative_step:
            return t - self.t_front
        if self.interpolate:
            return self.dt_safe
        dt_needed = t - self.t_front
        if dt_needed <= self.dt_safe:
            return dt_needed
        return dt_needed / (int(dt_needed / self.dt_safe) + 1)

    cdef double recompute_safe_step(self, double err, double dt):
        cdef factor = 0.
        if err == 0:
            factor = 10
        factor = 0.9*err**(-1/(self.order+1))
        factor = min(10, factor)
        factor = max(0.2, factor)

        self.dt_safe = dt * factor
        if self.max_step:
            self.dt_safe = min(self.max_step, self.dt_safe)
        if self.min_step:
            self.dt_safe = max(self.min_step, self.dt_safe)

    @property
    def y(self):
        return self._y.raw()

    @property
    def y_prev(self):
        return self._y_prev.raw()

    @property
    def y_front(self):
        return self._y_front.raw()

    @property
    def t_front(self):
        return self.t_front

    @property
    def t(self):
        return self.t
