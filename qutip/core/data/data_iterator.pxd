#cython: language_level=3

from qutip.core.data cimport Data, Dense, Dia, CSR


cdef class Data_iterator:
    cdef readonly int nnz
    cdef bint transpose, conj

    cdef (int, int, double complex) next(self)


cdef class Dense_iterator(Data_iterator):
    cdef Dense oper
    cdef int position


cdef class CSR_iterator(Data_iterator):
    cdef CSR oper
    cdef int row, idx


cdef class Dia_iterator(Data_iterator):
    cdef Dia oper
    cdef int diag_N, col, offset, diag_end


cdef Data_iterator _make_iter(Data oper, bint transpose, bint conj)
