#cython: language_level=3

from qutip.core.data cimport CSR, Dense, Data, Diag

cpdef CSR imul_csr(CSR matrix, double complex value)
cpdef CSR mul_csr(CSR matrix, double complex value)
cpdef CSR neg_csr(CSR matrix)

cpdef Dense imul_dense(Dense matrix, double complex value)
cpdef Dense mul_dense(Dense matrix, double complex value)
cpdef Dense neg_dense(Dense matrix)

cpdef Diag imul_diag(Diag matrix, double complex value)
cpdef Diag mul_diag(Diag matrix, double complex value)
cpdef Diag neg_diag(Diag matrix)

cpdef Data imul_data(Data matrix, double complex value)
