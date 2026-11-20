#cython: language_level=3

from qutip.core.data cimport CSR, Dense, Dia

cpdef bint isherm_csr(CSR matrix, double tol=*)
cpdef bint isdiag_csr(CSR matrix) nogil
cpdef bint iszero_csr(CSR matrix, double tol=*) nogil
cpdef bint iszero_dense(Dense matrix, double tol=*) nogil

cpdef bint isherm_dia(Dia matrix, double tol=*) nogil
cpdef bint isdiag_dia(Dia matrix, double tol=*) nogil
cpdef bint iszero_dia(Dia matrix, double tol=*) nogil
