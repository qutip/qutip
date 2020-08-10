#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef CSR pow_csr(CSR matrix, unsigned long long n)
