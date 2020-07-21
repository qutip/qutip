#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data.csr cimport CSR

cpdef double complex expect_csr(CSR op, CSR state) nogil except *
cpdef double complex expect_super_csr(CSR op, CSR state) nogil except *
