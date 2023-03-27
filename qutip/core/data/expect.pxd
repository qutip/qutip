#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Data

cpdef double complex expect_csr(CSR op, CSR state) nogil except *
cpdef double complex expect_super_csr(CSR op, CSR state) nogil except *

cpdef double complex expect_csr_dense(CSR op, Dense state) nogil except *
cpdef double complex expect_super_csr_dense(CSR op, Dense state) nogil except *

cpdef double complex expect_dense(Dense op, Dense state) nogil except *
cpdef double complex expect_super_dense(Dense op, Dense state) nogil except *

cdef double complex expect_data_dense(Data op, Dense state) except *
cdef double complex expect_super_data_dense(Data op, Dense state) except *
