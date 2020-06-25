#cython: language_level=3
cdef class Coefficient:
    cdef readonly str codeString

    cdef complex _call(self, double t, dict args)
