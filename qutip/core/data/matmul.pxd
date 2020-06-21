#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef CSR matmul_csr(CSR left, CSR right, CSR out=*)
