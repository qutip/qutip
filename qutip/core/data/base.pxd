#cython: language_level=3

import numpy as np
cimport numpy as cnp

# If you change the typedef, change the objects in base.pyx too!
ctypedef cnp.npy_int32 idxint
cdef public object idxint_dtype
cdef int idxint_DTYPE

cdef class Data:
    cdef readonly (idxint, idxint) shape
    cpdef object to_array(self)
    cpdef double complex trace(self)
    cpdef Data adjoint(self)
    cpdef Data conj(self)
    cpdef Data transpose(self)
