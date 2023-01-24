#cython: language_level=3
from qutip.core.data cimport Data

cdef class _StochasticSystem:
    cdef readonly int num_collapse
    cdef readonly bint issuper
    cdef readonly object dims
    cdef Data state
    cdef double t
    cdef object imp

    cpdef Data drift(self, t, Data state)

    cpdef list diffusion(self, t, Data state)

    cdef void set_state(self, double t, Data state)
