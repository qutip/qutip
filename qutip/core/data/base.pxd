#cython: language_level=3

import numpy as np
cimport numpy as cnp

# If you change the typedef, change the objects in base.pyx too!
ctypedef cnp.npy_int32 idxint
cdef object idxint_dtype
cdef int idxint_DTYPE

cdef class Data:
    cdef readonly (idxint, idxint) shape
