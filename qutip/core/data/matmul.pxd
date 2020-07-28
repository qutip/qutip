#cython: language_level=3

from qutip.core.data.csr cimport CSR

cdef void mv_csr(CSR matrix, double complex *vector, double complex *out) nogil
cpdef CSR matmul_csr(CSR left, CSR right)
