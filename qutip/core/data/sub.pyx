#cython: language_level=3
#cython: boundscheck=False, wrapround=False

from qutip.core.data cimport csr, CSR, neg_csr, add_csr

cpdef CSR sub_csr(CSR left, CSR right):
    return add_csr(left, neg_csr(right))
