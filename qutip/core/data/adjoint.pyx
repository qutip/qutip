#cython: language_level=3
#cython: boundscheck=False, wraparound=False

from libc.string cimport memset

cimport cython

from qutip.core.data.base cimport idxint
from qutip.core.data.csr cimport CSR
from qutip.core.data cimport csr

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)


cpdef CSR transpose_csr(CSR matrix):
    """Transpose the CSR matrix, and return a new object."""
    cdef idxint nnz_ = csr.nnz(matrix)
    cdef CSR out = csr.empty(matrix.shape[1], matrix.shape[0], nnz_)
    cdef idxint row, col, ptr, ptr_out
    cdef idxint rows_in=matrix.shape[0], rows_out=matrix.shape[1]
    with nogil:
        memset(&out.row_index[0], 0, (rows_out + 1) * sizeof(idxint))
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr] + 1
                out.row_index[col] += 1
        for row in range(rows_out):
            out.row_index[row + 1] += out.row_index[row]
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr]
                ptr_out = out.row_index[col]
                out.data[ptr_out] = matrix.data[ptr]
                out.col_index[ptr_out] = row
                out.row_index[col] = ptr_out + 1
        for row in range(rows_out, 0, -1):
            out.row_index[row] = out.row_index[row - 1]
        out.row_index[0] = 0
    return out


cpdef CSR adjoint_csr(CSR matrix):
    """Conjugate-transpose the CSR matrix, and return a new object."""
    cdef idxint nnz_ = csr.nnz(matrix)
    cdef CSR out = csr.empty(matrix.shape[1], matrix.shape[0], nnz_)
    cdef idxint row, col, ptr, ptr_out
    cdef idxint rows_in=matrix.shape[0], rows_out=matrix.shape[1]
    with nogil:
        memset(&out.row_index[0], 0, (rows_out + 1) * sizeof(idxint))
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr] + 1
                out.row_index[col] += 1
        for row in range(rows_out):
            out.row_index[row + 1] += out.row_index[row]
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr]
                ptr_out = out.row_index[col]
                out.data[ptr_out] = conj(matrix.data[ptr])
                out.col_index[ptr_out] = row
                out.row_index[col] = ptr_out + 1
        for row in range(rows_out, 0, -1):
            out.row_index[row] = out.row_index[row - 1]
        out.row_index[0] = 0
    return out


cpdef CSR conj_csr(CSR matrix):
    """Conjugate the CSR matrix, and return a new object."""
    cdef CSR out = csr.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            out.data[ptr] = conj(matrix.data[ptr])
    return out
