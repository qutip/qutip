#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Data, Diag

cpdef double one_csr(CSR matrix) except -1
cpdef double trace_csr(CSR matrix) except -1
cpdef double max_csr(CSR matrix) nogil
cpdef double frobenius_csr(CSR matrix) nogil
cpdef double l2_csr(CSR matrix) nogil except -1

cpdef double frobenius_dense(Dense matrix) nogil
cpdef double l2_dense(Dense matrix) nogil except -1

cpdef double one_diag(Diag matrix) except -1
cpdef double max_diag(Diag matrix) nogil
cpdef double frobenius_diag(Diag matrix) nogil
cpdef double l2_diag(Diag matrix) except -1 nogil

cpdef double frobenius_data(Data state)

cdef inline int int_max(int a, int b) nogil:
    # Name collision between the ``max`` builtin and norm.max
    return b if b > a else a
