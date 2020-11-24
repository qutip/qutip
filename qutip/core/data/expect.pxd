#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, CSC, Dense

cpdef double complex expect_csr(CSR op, CSR state) nogil except *
cpdef double complex expect_super_csr(CSR op, CSR state) nogil except *

cpdef double complex expect_csr_dense(CSR op, Dense state) nogil except *
cpdef double complex expect_super_csr_dense(CSR op, Dense state) nogil except *

cpdef double complex expect_csc_dense(CSC op, Dense state) except *
cpdef double complex expect_super_csc_dense(CSC op, Dense state) except *

cpdef double complex expect_dense_dense(Dense op, Dense state) nogil except *
cpdef double complex expect_super_dense_dense(Dense op, Dense state) nogil except *
