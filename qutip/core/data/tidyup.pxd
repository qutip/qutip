#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data.csr cimport CSR

cpdef CSR tidyup_csr(CSR matrix, double tol)
