#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Data, Dia

cpdef double one_csr(CSR matrix) except -1
cpdef double trace_csr(CSR matrix) except -1
cpdef double max_csr(CSR matrix) nogil
cpdef double frobenius_csr(CSR matrix) nogil
cpdef double l2_csr(CSR matrix) except -1 nogil

cpdef double frobenius_dense(Dense matrix) nogil
cpdef double l2_dense(Dense matrix) except -1 nogil

cpdef double one_dia(Dia matrix) except -1
cpdef double max_dia(Dia matrix) nogil
cpdef double frobenius_dia(Dia matrix) nogil
cpdef double l2_dia(Dia matrix) except -1 nogil

cpdef double frobenius_data(Data state) except -1

cdef inline int int_max(int a, int b) nogil:
    # Name collision between the ``max`` builtin and norm.max
    return b if b > a else a
