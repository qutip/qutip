#cython: language_level=3
#cython: boundscheck=False, wrapround=False, initializedcheck=False

from qutip.core.data cimport CSR, add_csr, Dense, add_dense

cpdef CSR sub_csr(CSR left, CSR right):
    return add_csr(left, right, -1)

cpdef Dense sub_dense(Dense left, Dense right):
    return add_dense(left, right, -1)
