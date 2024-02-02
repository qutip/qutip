#cython: language_level=3

from qutip.core.data cimport COO, CSR, Dense, Dia

cpdef COO pow_coo(COO matrix, unsigned long long n)

cpdef CSR pow_csr(CSR matrix, unsigned long long n)

cpdef Dense pow_dense(Dense matrix, unsigned long long n)

cpdef Dia pow_dia(Dia matrix, unsigned long long n)