#cython: language_level=3

from qutip.core.data cimport CSR, Dense

cpdef CSR sub_csr(CSR left, CSR right)
cpdef Dense sub_dense(Dense left, Dense right)
