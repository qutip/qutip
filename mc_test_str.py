import numpy as np
cimport numpy as np
import cython

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)





cdef class factor_obj:
    cdef complex* str_array_0
    cdef double w 

    def __init__(self, args):
        self.str_array_0 = &args["str_array_0"].data
        self.w = args['w']

    cdef void factor(double t, complex* factor):
        cdef double str_array_0 = self.str_array_0
        cdef double w = self.w
        factor[0] = interpolate(t, str_array_0, 100, 0.01)
        factor[1] = sin(w*t)
    
    def factor_ptr(double t, complex* factor):
        return PyLong_FromVoidPtr(<void *> self.factor)

cdef complex interpolate(double t, complex* str_array_0, int N, double dt):
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
    cdef np.ndarray[double, ndim=1] coeff = np.empty(4)
    cdef int i
    cdef int ii = <int>(t/dt)
    cdef double a = (t/dt - ii)
    cdef double approx = 0.


    approx += (-a**3 +3*a**2 - 2*a   )/6.0*str_array_0[ii-1]
    approx += ( a**3 -2*a**2 -   a +2)*0.5*str_array_0[ii]
    approx += (-a**3 +  a**2 + 2*a   )*0.5*str_array_0[ii+1]
    approx += ( a**3         -   a   )/6.0*str_array_0[ii+2]

    return approx




def interpolate_from_array(t,array,dt):
    N = len(array)
    ii = int(t/dt)
    a = (t/dt - ii)
    vec = np.array([array[ii], array[ii+1],
                    array[ii+2], array[ii+3]])
    return np.inner(vec,fac(a))


