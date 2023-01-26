#cython: language_level=3
from qutip.core.data cimport Data, Dense

cdef class _StochasticSystem:
    cdef readonly int num_collapse
    cdef readonly bint issuper
    cdef readonly object dims
    cdef Data state
    cdef double t
    cdef object imp
    cdef int is_set

    cpdef Data drift(self, t, Data state)

    cpdef list diffusion(self, t, Data state)

    cpdef void set_state(self, double t, Data state)
    cpdef Data a(self)
    cpdef Data bi(self, int i)
    cpdef Data Libj(self, int i, int j)
    cpdef Data Lia(self, int i)
    cpdef Data L0bi(self, int i)
    cpdef Data LiLjbk(self, int i, int j, int k)
    cpdef Data L0a(self)
