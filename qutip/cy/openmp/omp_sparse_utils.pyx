#cython: language_level=3

import numpy as np
cimport numpy as cnp
cimport cython
from libcpp cimport bool
from libc.math cimport fabs
from cython.parallel cimport parallel, prange

cdef extern from "<complex>" namespace "std" nogil:
    double real(double complex x)
    double imag(double complex x)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool omp_tidyup(complex[::1] data, double atol, int nnz, int nthr):
    cdef int kk,
    cdef double re, im
    cdef bool re_flag, im_flag, out_flag = 0
    with nogil, parallel(num_threads = nthr):
        for kk in prange(nnz, schedule='static'):
            re_flag = 0
            im_flag = 0
            re = real(data[kk])
            im = imag(data[kk])
            if fabs(re) < atol:
                re = 0
                re_flag = 1
            if fabs(im) < atol:
                im = 0
                im_flag = 1
            if re_flag or im_flag:
                data[kk] = re +1j*im
            if re_flag and im_flag:
                out_flag = 1
    return out_flag
