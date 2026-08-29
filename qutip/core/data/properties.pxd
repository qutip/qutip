from qutip.core.data cimport CSR, Dense, Dia

cpdef bint isherm_csr(CSR matrix, double tol=*) except -1
cpdef bint isdiag_csr(CSR matrix) noexcept nogil
cpdef bint iszero_csr(CSR matrix, double tol=*) except -1 nogil

cpdef bint isherm_dense(Dense matrix, double tol=*) except -1 nogil
cpdef bint isdiag_dense(Dense matrix) noexcept nogil
cpdef bint iszero_dense(Dense matrix, double tol=*) except -1 nogil

cpdef bint isherm_dia(Dia matrix, double tol=*) except -1 nogil
cpdef bint isdiag_dia(Dia matrix, double tol=*) except -1 nogil
cpdef bint iszero_dia(Dia matrix, double tol=*) except -1 nogil
