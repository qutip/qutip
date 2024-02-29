#cython: language_level=3

cdef extern from "src/intdtype.h":
    # cython is smart enough to understand this int can be 32 or 64 bits.
    ctypedef int idxint
    cdef int _idxint_size

cdef int idxint_DTYPE

cdef class Data:
    cdef readonly (idxint, idxint) shape
    cpdef object to_array(self)
    cpdef double complex trace(self)
    cpdef Data adjoint(self)
    cpdef Data conj(self)
    cpdef Data transpose(self)
    cpdef Data copy(self)
