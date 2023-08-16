#cython: language_level=3

from qutip.core.data cimport CSR, Dense, Dia

cpdef double complex trace_csr(CSR matrix) except * nogil
cpdef double complex trace_dense(Dense matrix) except * nogil
cpdef double complex trace_dia(Dia matrix) except * nogil

cpdef double complex trace_oper_ket_csr(CSR matrix) except * nogil
cpdef double complex trace_oper_ket_dense(Dense matrix) except * nogil
cpdef double complex trace_oper_ket_dia(Dia matrix) except * nogil
