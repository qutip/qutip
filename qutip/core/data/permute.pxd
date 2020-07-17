#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data.csr cimport CSR

cpdef CSR permute_csr(CSR matrix, object row_perm, object col_perm)
