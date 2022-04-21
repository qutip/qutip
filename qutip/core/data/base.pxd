#cython: language_level=3

cdef class Data:
    cdef readonly (idxint, idxint) shape
    cpdef object to_array(self)
    cpdef double complex trace(self)
    cpdef Data adjoint(self)
    cpdef Data conj(self)
    cpdef Data transpose(self)
    cpdef Data copy(self)
