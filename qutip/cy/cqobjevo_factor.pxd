#!python
#cython: language_level=3
cimport numpy as np

cdef np.ndarray[complex, ndim=1] zptr2array1d(complex* ptr, int N)

cdef np.ndarray[complex, ndim=2] zptr2array2d(complex* ptr, int R, int C)

cdef np.ndarray[int, ndim=1] iprt2array(int* ptr, int N)

cdef class CoeffFunc:
    cdef dict _args
    cdef int _num_ops
    cdef void _call_core(self, double t, complex* coeff)
    cdef void _dyn_args(self, double t, complex* state, int[::1] shape)

cdef class StrCoeff(CoeffFunc):
    cdef list _dyn_args_list
    cdef int _num_expect
    cdef int[2] _mat_shape
    cdef list _expect_op
    cdef complex[::1] _expect_vec
    cdef complex[::1] _vec
