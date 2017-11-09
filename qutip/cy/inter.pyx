import cython
cimport cython

import numpy as np
cimport numpy as cnp

import scipy.linalg

def prep_cubic_spline(array, tlist, poly=False):
    """ Prepare coefficients for interpalation of array.
    boudary conditions assumed: second derivative null at the extremities.

    Input:
      array:
        nd.array of double / complex
        Array to interpolate

      tlist:
        nd.array of double
        times or x of the array, must be inscreasing,
          the step size do not need to be constant.

      poly:
        bool
        decide the format of the output

    Output:
      if poly:
        (np.array of polynomial of third order [N,4], bool
        For t in the interval  tlist[i] < t < tlist[i+1]
        The interpolation function is
          poly[i,0] + poly[i,1]*t + poly[i,2]*t**2 + poly[i,3]*t**3

      if not poly:
        the second derivative at each time
        np.array [N]

    """
    N = len(tlist)
    M = np.zeros((3,N), dtype=array.dtype)
    x = np.zeros(N, dtype=array.dtype)
    M[1,:] = 2.
    dt_cte = True
    dt0 = tlist[1]-tlist[0]
    for i in range(1,N-1):
        dt1 = tlist[i]-tlist[i-1]
        dt2 = tlist[i+1]-tlist[i]
        if ((dt2 - dt0) > 10e-10):
            dt_cte = False
        M[0,i+1] =  dt1 / (dt1+dt2)
        M[2,i-1] =  dt2 / (dt1+dt2)
        x[i] = ((array[i-1] - array[i]) / dt1 - (array[i] - array[i+1]) / dt2) \
               * 6 / (dt1+dt2)
    Ms = scipy.linalg.solve_banded((1,1), M, x, True, True) / 6

    if poly:
        poly = np.zeros((N,4), dtype=array.dtype)
        for i in range(N-1):
            dt = tlist[i+1]-tlist[i]
            a1 = Ms[i]/dt
            a2 = Ms[i+1]/dt
            b1 = array[i]/dt - Ms[i]*dt
            b2 = array[i+1]/dt - Ms[i+1]*dt
            poly[i,0] = a1 * tlist[i+1] * tlist[i+1] * tlist[i+1] - \
                        a2 * tlist[i] * tlist[i] * tlist[i] + \
                        b1 * tlist[i+1] - b2 * tlist[i]
            poly[i,1] = 3 * a2 * tlist[i] * tlist[i] - \
                        3 * a1 * tlist[i+1] * tlist[i+1] - b1 + b2
            poly[i,2] = 3 * (a1 * tlist[i+1] - a2 * tlist[i])
            poly[i,3] = a2 - a1
        return (poly, dt_cte)
    else:
        return (Ms, dt_cte)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int binary_search(double x, double[::1] t, int N):
    #Binary search for the interval
    cdef int low = 0
    cdef int high = N
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
cpdef double spline_float_cte_poly(double x,
                                   double[::1] t,
                                   double[:,::1] poly,
                                   int N,
                                   double dt):
    # inbound?
    if x < t[0]:
        return poly[0,0]
    elif x > t[N-1]:
        return poly[N-2,0]+x*(poly[N-2,1]+x*(poly[N-2,2]+x*poly[N-2,3]))
    cdef int p = <int>(x/dt)
    return poly[p,0]+x*(poly[p,1]+x*(poly[p,2]+x*poly[p,3]))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double spline_float_t_poly(double x,
                                 double[::1] t,
                                 double[:,::1] poly,
                                 int N):
    # inbound?
    if x < t[0]:
        return poly[0,0]
    elif x > t[N-1]:
        return poly[N-2,0]+x*(poly[N-2,1]+x*(poly[N-2,2]+x*poly[N-2,3]))
    cdef int p = binary_search(x, t, N)
    return poly[p,0]+x*(poly[p,1]+x*(poly[p,2]+x*poly[p,3]))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double spline_float_cte_second(double x,
                                     double[::1] t,
                                     double[::1] y,
                                     double[::1] M,
                                     int N,
                                     double dt):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[N-1]:
        return y[N-1]
    cdef int p = <int>(x/dt)
    cdef double tb = (x/dt - p)
    cdef double te = 1 - tb
    cdef double dt2 = dt * dt
    cdef double Me = M[p+1] * dt2
    cdef double Mb = M[p] * dt2
    return te * (Mb * te * te + (y[p]   - Mb)) + \
           tb * (Me * tb * tb + (y[p+1] - Me))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double spline_float_t_second(double x,
                                   double[::1] t,
                                   double[::1] y,
                                   double[::1] M,
                                   int N):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[N-1]:
        return y[N-1]
    cdef int p = binary_search(x, t, N)
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
cpdef complex spline_complex_cte_poly(double x,
                                      double[::1] t,
                                      complex[:,::1] poly,
                                      int N,
                                      double dt):
    # inbound?
    if x < t[0]:
        return poly[0,0]
    elif x > t[N-1]:
        return poly[N-2,0]+x*(poly[N-2,1]+x*(poly[N-2,2]+x*poly[N-2,3]))
    cdef int p = <int>(x/dt)
    return poly[p,0]+x*(poly[p,1]+x*(poly[p,2]+x*poly[p,3]))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef complex spline_complex_t_poly(double x,
                                    double[::1] t,
                                    complex[:,::1] poly,
                                    int N):
    # inbound?
    if x < t[0]:
        return poly[0,0]
    elif x > t[N-1]:
        return poly[N-2,0]+x*(poly[N-2,1]+x*(poly[N-2,2]+x*poly[N-2,3]))
    cdef int p = binary_search(x, t, N)
    return poly[p,0]+x*(poly[p,1]+x*(poly[p,2]+x*poly[p,3]))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef complex spline_complex_cte_second(double x,
                                        double[::1] t,
                                        complex[::1] y,
                                        complex[::1] M,
                                        int N,
                                        double dt):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[N-1]:
        return y[N-1]
    cdef int p = <int>(x/dt)
    cdef double tb = (x/dt - p)
    cdef double te = 1 - tb
    cdef double dt2 = dt * dt
    cdef complex Me = M[p+1] * dt2
    cdef complex Mb = M[p] * dt2
    return te * (Mb * te * te + (y[p]   - Mb)) + \
           tb * (Me * tb * tb + (y[p+1] - Me))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef complex spline_complex_t_second(double x,
                                      double[::1] t,
                                      complex[::1] y,
                                      complex[::1] M,
                                      int N):
    # inbound?
    if x < t[0]:
        return y[0]
    elif x > t[N-1]:
        return y[N-1]
    cdef int p = binary_search(x, t, N)
    cdef double dt = t[p+1] - t[p]
    cdef double tb = (x - t[p]) / dt
    cdef double te = 1 - tb
    cdef double dt2 = dt * dt
    cdef complex Me = M[p+1] * dt2
    cdef complex Mb = M[p] * dt2
    return te * (Mb * te * te + (y[p]   - Mb)) + \
           tb * (Me * tb * tb + (y[p+1] - Me))







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef complex interpolate(double t, double* str_array_0, int N, double dt):
    # inbound?
    if t < 0.:
        return str_array_0[0]
    if t > dt*(N-1):
        return str_array_0[N-1]

    # On the boundaries, linear approximation
    # Better sheme useful?
    if t < dt:
        return str_array_0[0]*(dt-t)/dt + str_array_0[1]*t/dt
    if t > dt*(N-2):
        return str_array_0[N-2]*(dt*(N-1)-t)/dt + \
                str_array_0[N-1]*(t-dt*(N-2))/dt

    # In the middle: 4th order polynomial approximation
    cdef int ii = <int>(t/dt)
    cdef double a = (t/dt - ii)
    cdef complex approx = 0.

    approx += a * (a * (3 - a) - 2) * 0.1666666666666666 * str_array_0[ii-1]
    approx += (2 + a * (a * (a - 2) - 1)) * 0.5 * str_array_0[ii]
    approx += a * (a * (1 - a) + 2) * 0.5 * str_array_0[ii+1]
    approx += a * (a * a - 1) * 0.1666666666666666 * str_array_0[ii+2]

    return approx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef complex zinterpolate(double t, complex* str_array_0, int N, double dt):
    # inbound?
    if t < 0.:
        return str_array_0[0]
    if t > dt*(N-1):
        return str_array_0[N-1]

    # On the boundaries, linear approximation
    # Better sheme useful?
    if t < dt:
        return str_array_0[0]*(dt-t)/dt + str_array_0[1]*t/dt
    if t > dt*(N-2):
        return str_array_0[N-2]*(dt*(N-1)-t)/dt + \
                str_array_0[N-1]*(t-dt*(N-2))/dt

    # In the middle: 4th order polynomial approximation
    cdef int ii = <int>(t/dt)
    cdef double a = (t/dt - ii)
    cdef complex approx = 0.

    approx += a * (a * (3 - a) - 2) * 0.1666666666666666 * str_array_0[ii-1]
    approx += (2 + a * (a * (a - 2) - 1)) * 0.5 * str_array_0[ii]
    approx += a * (a * (1 - a) + 2) * 0.5 * str_array_0[ii+1]
    approx += a * (a * a - 1) * 0.1666666666666666 * str_array_0[ii+2]

    return approx
