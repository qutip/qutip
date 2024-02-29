#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Dia

cpdef CSR tidyup_csr(CSR matrix, double tol, bint inplace=*)
cpdef Dense tidyup_dense(Dense matrix, double tol, bint inplace=*)
cpdef Dia tidyup_dia(Dia matrix, double tol, bint inplace=*)
