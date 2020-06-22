#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef CSR adjoint_csr(CSR matrix)
cpdef CSR transpose_csr(CSR matrix)
cpdef CSR conj_csr(CSR matrix)
