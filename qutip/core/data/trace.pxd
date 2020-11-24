#cython: language_level=3

from qutip.core.data cimport CSR, Dense, CSC

cpdef double complex trace_csr(CSR matrix) nogil except *
cpdef double complex trace_csc(CSC matrix) nogil except *
cpdef double complex trace_dense(Dense matrix) nogil except *
