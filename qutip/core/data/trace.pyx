#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython

from qutip.core.data.base cimport Data
from qutip.core.data.csr cimport CSR

cdef void _check_shape(Data matrix) nogil except *:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("".join([
            "matrix shape ", str(matrix.shape), " is not square.",
        ]))


cpdef double complex trace_csr(CSR matrix) nogil except *:
    _check_shape(matrix)
    cdef size_t row, ptr
    cdef double complex trace = 0
    for row in range(matrix.shape[0]):
        for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
            if matrix.col_index[ptr] == row:
                trace += matrix.data[ptr]
                break
    return trace
