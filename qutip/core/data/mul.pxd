#cython: language_level=3

from qutip.core.data cimport COO, CSR, Dense, Data, Dia

cpdef COO imul_coo(COO matrix, double complex value)
cpdef COO mul_coo(COO matrix, double complex value)
cpdef COO neg_coo(COO matrix)

cpdef CSR imul_csr(CSR matrix, double complex value)
cpdef CSR mul_csr(CSR matrix, double complex value)
cpdef CSR neg_csr(CSR matrix)

cpdef Dense imul_dense(Dense matrix, double complex value)
cpdef Dense mul_dense(Dense matrix, double complex value)
cpdef Dense neg_dense(Dense matrix)

cpdef Dia imul_dia(Dia matrix, double complex value)
cpdef Dia mul_dia(Dia matrix, double complex value)
cpdef Dia neg_dia(Dia matrix)

cpdef Data imul_data(Data matrix, double complex value)
