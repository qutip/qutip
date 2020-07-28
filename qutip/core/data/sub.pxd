#cython: language_level=3

from qutip.core.data cimport CSR

cpdef CSR sub_csr(CSR left, CSR right)
