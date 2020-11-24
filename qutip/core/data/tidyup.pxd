#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, CSC

cpdef CSR tidyup_csr(CSR matrix, double tol, bint inplace=*)
cpdef CSC tidyup_csc(CSC matrix, double tol, bint inplace=*)
cpdef Dense tidyup_dense(Dense matrix, double tol, bint inplace=*)
