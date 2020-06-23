#cython: language_level=3

from qutip.core.data cimport CSR

cpdef CSR mul_csr(CSR matrix, double complex value)
cpdef CSR neg_csr(CSR matrix)
