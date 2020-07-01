#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef double complex trace_csr(CSR matrix)
