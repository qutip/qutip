#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.dia cimport Dia

cdef void add_dense_eq_order_inplace(Dense left, Dense right, double complex scale)
cpdef CSR add_csr(CSR left, CSR right, double complex scale=*)
cpdef Dense add_dense(Dense left, Dense right, double complex scale=*)
cpdef Dia add_dia(Dia left, Dia right, double complex scale=*)
cpdef Dense iadd_dense(Dense left, Dense right, double complex scale=*)

cpdef CSR sub_csr(CSR left, CSR right)
cpdef Dense sub_dense(Dense left, Dense right)
cpdef Dia sub_dia(Dia left, Dia right)
