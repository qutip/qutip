#cython: language_level=3

from qutip.core.data cimport CSR, Dense, CSC

cpdef void mul_csr_inplace(CSR matrix, double complex value)
cpdef CSR mul_csr(CSR matrix, double complex value)
cpdef CSR neg_csr(CSR matrix)

cpdef void mul_csc_inplace(CSC matrix, double complex value)
cpdef CSC mul_csc(CSC matrix, double complex value)
cpdef CSC neg_csc(CSC matrix)

cpdef void mul_dense_inplace(Dense matrix, double complex value)
cpdef Dense mul_dense(Dense matrix, double complex value)
cpdef Dense neg_dense(Dense matrix)
