



cdef class coeffFunc:
    cdef int N_ops
    cdef void _call_core(self, double t, complex * coeff)
