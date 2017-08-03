import cython
#import numpy as np
#cimport numpy as np


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
        return str_array_0[N-2]*(dt*(N-1)-t)/dt + str_array_0[N-1]*(t-dt*(N-2))/dt

    # In the middle: 4th order polynomial approximation
    cdef int ii = <int>(t/dt)
    cdef double a = (t/dt - ii)
    cdef complex approx = 0.

    approx += (-a**3 +3*a**2 - 2*a   )/6.0*str_array_0[ii-1]
    approx += ( a**3 -2*a**2 -   a +2)*0.5*str_array_0[ii]
    approx += (-a**3 +  a**2 + 2*a   )*0.5*str_array_0[ii+1]
    approx += ( a**3         -   a   )/6.0*str_array_0[ii+2]

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
        return str_array_0[N-2]*(dt*(N-1)-t)/dt + str_array_0[N-1]*(t-dt*(N-2))/dt

    # In the middle: 4th order polynomial approximation
    cdef int ii = <int>(t/dt)
    cdef double a = (t/dt - ii)
    cdef complex approx = 0.

    approx += (-a**3 +3*a**2 - 2*a   )/6.0*str_array_0[ii-1]
    approx += ( a**3 -2*a**2 -   a +2)*0.5*str_array_0[ii]
    approx += (-a**3 +  a**2 + 2*a   )*0.5*str_array_0[ii+1]
    approx += ( a**3         -   a   )/6.0*str_array_0[ii+2]

    return approx
