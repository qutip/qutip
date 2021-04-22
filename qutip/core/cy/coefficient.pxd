#cython: language_level=3
cdef class Coefficient:
    cdef dict args
    cdef double complex _call(self, double t) except *
    cpdef Coefficient copy(self)
