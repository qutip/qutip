#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef bint isherm_csr(CSR matrix, double tol=*)
