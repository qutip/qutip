#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data.csr cimport CSR

cpdef double one_csr(CSR matrix) nogil except -1
cpdef double trace_csr(CSR matrix) except -1
cpdef double max_csr(CSR matrix) nogil
cpdef double frobenius_csr(CSR matrix) nogil
cpdef double l2_csr(CSR matrix) nogil except -1
