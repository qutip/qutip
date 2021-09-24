#cython: language_level=3

"""
Second version of cublicspline interpolation. (in parallel with interpolate)
- Accept non-uniformely sampled data.
- Faster but use more memory than interpolate
- No python interface, used by QobjEvo.
"""
import cython
cimport cython

import numpy as np
cimport numpy as cnp

import scipy.linalg

def _prep_cubic_spline(array, tlist):
    """ Prepare coefficients for interpalation of array.
    boudary conditions assumed: second derivative null at the extremities.

    Parameters
    ----------
      array : nd.array of double / complex
        Array to interpolate

      tlist : nd.array of double
        times or x of the array, must be inscreasing.
        The step size do not need to be constant.

    Returns
    -------
    np.array
        the second derivative at each time
    """
    n_t = len(tlist)
    M = np.zeros((3,n_t), dtype=array.dtype)
    x = np.zeros(n_t, dtype=array.dtype)
    M[1,:] = 2.
    dt_cte = True
    dt0 = tlist[1]-tlist[0]
    for i in range(1,n_t-1):
        dt1 = tlist[i]-tlist[i-1]
        dt2 = tlist[i+1]-tlist[i]
        if ((dt2 - dt0) > 10e-10):
            dt_cte = False
        M[0,i+1] =  dt1 / (dt1+dt2)
        M[2,i-1] =  dt2 / (dt1+dt2)
        x[i] = ((array[i-1] - array[i]) / dt1 - (array[i] - array[i+1]) / dt2) \
               * 6 / (dt1+dt2)
    Ms = scipy.linalg.solve_banded((1,1), M, x, True, True) / 6
    return (Ms, dt_cte)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _binary_search(double x, double[::1] t, int n):
    #Binary search for the interval
    cdef int low = 0
    cdef int high = n
    cdef int middle
    cdef int count = 0
    while low+1 != high and count < 30:
        middle = (low+high)//2
        if x < t[middle]:
            high = middle
        else:
            low = middle
        count += 1
    return low


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _spline_float_cte_second(double x,
                                     double[::1] t,
                                     double[::1] y,
                                     double[::1] M,
                                     int n_t,
                                     double dt):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[n_t-1]:
        return y[n_t-1]
    cdef double xx = (x-t[0])/dt
    cdef int p = <int>xx
    cdef double tb = (xx - p)
    cdef double te = 1 - tb
    cdef double dt2 = dt * dt
    cdef double Me = M[p+1] * dt2
    cdef double Mb = M[p] * dt2
    return te * (Mb * te * te + (y[p]   - Mb)) + \
           tb * (Me * tb * tb + (y[p+1] - Me))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _spline_float_t_second(double x,
                                   double[::1] t,
                                   double[::1] y,
                                   double[::1] M,
                                   int n_t):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[n_t-1]:
        return y[n_t-1]
    cdef int p = _binary_search(x, t, n_t)
    cdef double dt = t[p+1] - t[p]
    cdef double tb = (x - t[p]) / dt
    cdef double te = 1 - tb
    cdef double dt2 = dt * dt
    cdef double Me = M[p+1] * dt2
    cdef double Mb = M[p] * dt2
    return te * (Mb * te * te + (y[p]   - Mb)) + \
           tb * (Me * tb * tb + (y[p+1] - Me))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef complex _spline_complex_cte_second(double x,
                                        double[::1] t,
                                        complex[::1] y,
                                        complex[::1] M,
                                        int n_t,
                                        double dt):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[n_t-1]:
        return y[n_t-1]
    cdef double xx = (x-t[0])/dt
    cdef int p = <int>xx
    cdef double tb = (xx - p)
    cdef double te = 1 - tb
    cdef double dt2 = dt * dt
    cdef complex Me = M[p+1] * dt2
    cdef complex Mb = M[p] * dt2
    return te * (Mb * te * te + (y[p]   - Mb)) + \
           tb * (Me * tb * tb + (y[p+1] - Me))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef complex _spline_complex_t_second(double x,
                                      double[::1] t,
                                      complex[::1] y,
                                      complex[::1] M,
                                      int n_t):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[n_t-1]:
        return y[n_t-1]
    cdef int p = _binary_search(x, t, n_t)
    cdef double dt = t[p+1] - t[p]
    cdef double tb = (x - t[p]) / dt
    cdef double te = 1 - tb
    cdef double dt2 = dt * dt
    cdef complex Me = M[p+1] * dt2
    cdef complex Mb = M[p] * dt2
    return te * (Mb * te * te + (y[p]   - Mb)) + \
           tb * (Me * tb * tb + (y[p+1] - Me))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _step_float_cte(double x, double[::1] t, double[::1] y, int n_t):
    if x < t[0]:
        return y[0]
    elif x >= t[n_t-1]:
        return y[n_t-1]
    cdef int p = <int> ((x-t[0]) / (t[n_t-1]-t[0]) * (n_t-1))
    return y[p]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef complex _step_complex_cte(double x, double[::1] t, complex[::1] y, int n_t):
    if x < t[0]:
        return y[0]
    elif x >= t[n_t-1]:
        return y[n_t-1]
    cdef int p = <int> ((x-t[0]) / (t[n_t-1]-t[0]) * (n_t-1))
    return y[p]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _step_float_t(double x, double[::1] t, double[::1] y, int n_t):
    if x < t[0]:
        return y[0]
    elif x >= t[n_t-1]:
        return y[n_t-1]
    # TODO this can be moved out for better performance
    cdef int p = _binary_search(x, t, n_t)
    return y[p]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef complex _step_complex_t(double x, double[::1] t, complex[::1] y, int n_t):
    if x < t[0]:
        return y[0]
    elif x >= t[n_t-1]:
        return y[n_t-1]
    # TODO this can be moved out for better performance
    cdef int p = _binary_search(x, t, n_t)
    return y[p]
