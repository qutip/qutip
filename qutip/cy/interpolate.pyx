#cython: boundscheck = False
#cython: wraparound = False
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "c_interpolate.h" nogil:
    double cinterpolate(double x, double a, double b, double *c, int lenc)
    void carray_interpolate(double *x, double a, double b, double *c, 
                            double *out, int lenx, int lenc)


cpdef double _interpolate(double x, double a, double b, double[::1] data):
    """
    Returns the interpolated value of a function at f(x) where x is in [a,b].
    """
    return cinterpolate(x, a, b, &data[0], data.shape[0])


def _array_interpolate(double[::1] x, double a, double b, double[::1] data): 
    """
    Returns the interpolated value of a function at f(x) where x is an
    array of points in [a,b].
    """
    cdef np.ndarray[double, ndim=1, mode='c'] out = np.zeros(x.shape[0], dtype=float)
    carray_interpolate(&x[0], a, b, &data[0], &out[0], x.shape[0], data.shape[0])
    return out
