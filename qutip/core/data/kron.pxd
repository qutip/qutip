#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef CSR kron_csr(CSR left, CSR right)
