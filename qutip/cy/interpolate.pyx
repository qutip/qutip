#cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport (fabs, fmin)
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double phi(double t):

    cdef double abs_t = fabs(t)
    if abs_t <= 1:
        return 4 - 6 * abs_t**2 + 3 * abs_t**3
    elif abs_t <= 2:
        return (2-abs_t)**3
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double interp(double x, double a, double b, double[::1] c):
    cdef int n = c.shape[0] - 3
    cdef double h = (b-a) / n
    cdef int l = <int>((x-a)/h) + 1
    cdef int m = <int>(fmin(l+3, n+3))
    cdef size_t ii
    cdef double s = 0, _tmp
    cdef double pos = (x-a)/h + 2

    for ii in range(l, m+1):
        _tmp = phi(pos - ii)
        if _tmp:
            s += c[ii-1] * _tmp
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef complex zinterp(double x, double a, double b, complex[::1] c):
    cdef int n = c.shape[0] - 3
    cdef double h = (b-a) / n
    cdef int l = <int>((x-a)/h) + 1
    cdef int m = <int>(fmin(l+3, n+3))
    cdef size_t ii
    cdef complex s = 0
    cdef double _tmp, pos = (x-a)/h + 2

    for ii in range(l, m+1):
        _tmp = phi(pos - ii)
        if _tmp:
            s += c[ii-1] * _tmp
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arr_interp(double[::1] x, double a, double b, double[::1] c):
    cdef int lenx = x.shape[0]
    cdef int lenc = c.shape[0]
    cdef int n = lenc - 3
    cdef double h = (b-a) / n
    cdef size_t ii, jj
    cdef int l, m
    cdef double pos, _tmp
    cdef cnp.ndarray[double, ndim=1, mode="c"] out = np.zeros(lenx, dtype=float)

    for jj in range(lenx):
        l = <int>((x[jj]-a)/h) + 1
        m = <int>(fmin(l+3, n+3))
        pos = (x[jj]-a)/h + 2

        for ii in range(l, m+1):
            _tmp = phi(pos - ii)
            if _tmp:
                out[jj] += c[ii-1] * _tmp

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arr_zinterp(double[::1] x, double a, double b, complex[::1] c):
    cdef int lenx = x.shape[0]
    cdef int lenc = c.shape[0]
    cdef int n = lenc - 3
    cdef double h = (b-a) / n
    cdef size_t ii, jj
    cdef int l, m
    cdef double pos, _tmp
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros(lenx, dtype=complex)

    for jj in range(lenx):
        l = <int>((x[jj]-a)/h) + 1
        m = <int>(fmin(l+3, n+3))
        pos = (x[jj]-a)/h + 2

        for ii in range(l, m+1):
            _tmp = phi(pos - ii)
            if _tmp:
                out[jj] = out[jj] + c[ii-1] * _tmp

    return out
