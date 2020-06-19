#cython: language_level=3

import numpy as np
cimport numpy as cnp

ctypedef cnp.npy_int32 idxint
cdef object idxint_dtype = np.int32
cdef int idxint_DTYPE = cnp.NPY_INT32

cdef class Data:
    cdef readonly (idxint, idxint) shape
