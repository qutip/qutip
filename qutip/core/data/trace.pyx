#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython

from qutip.core.data.csr cimport CSR


cpdef double complex trace_csr(CSR matrix):
    cdef size_t row, ptr
    cdef double complex trace = 0

    for row in range(matrix.shape[0]):
        for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
            if matrix.col_index[ptr] == row:
                trace += matrix.data[ptr]
                break
    return trace
