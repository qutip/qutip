#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef CSR add_csr(CSR left, CSR right, double complex scale=*)
