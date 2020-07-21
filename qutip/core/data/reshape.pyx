#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.stdlib cimport div, div_t
from libc.string cimport memcpy, memset

cimport cython

from qutip.core.data.base cimport idxint
from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR

cpdef CSR reshape_csr(CSR matrix, idxint n_rows_out, idxint n_cols_out):
    cdef size_t ptr, row_in, row_out=0, loc, cur=0
    cdef size_t n_rows_in=matrix.shape[0], n_cols_in=matrix.shape[1]
    cdef idxint nnz = csr.nnz(matrix)
    cdef div_t res
    cdef CSR out
    if n_rows_out * n_cols_out != n_rows_in * n_cols_in:
        message = "".join([
            "cannot reshape ", str(matrix.shape), " to ",
            "(", str(n_rows_out), ", ", str(n_cols_out), ")",
        ])
        raise ValueError(message)
    if n_rows_out <= 0 or n_cols_out <= 0:
        raise ValueError("must have > 0 rows and columns")
    out = csr.empty(n_rows_out, n_cols_out, nnz)
    matrix.sort_indices()
    with nogil:
        # Since the indices are now sorted, the data arrays will be identical.
        memcpy(&out.data[0], &matrix.data[0], nnz*sizeof(double complex))
        memset(&out.row_index[0], 0, (n_rows_out + 1) * sizeof(idxint))
        for row_in in range(n_rows_in):
            for ptr in range(matrix.row_index[row_in], matrix.row_index[row_in+1]):
                loc = cur + matrix.col_index[ptr]
                # This stdlib.div method is a little bit faster when working
                # with very dense large matrices, and doesn't make a difference
                # for smaller ones.
                res = div(loc, n_cols_out)
                out.row_index[res.quot + 1] += 1
                out.col_index[ptr] = res.rem
            cur += n_cols_in
        for row_out in range(n_rows_out + 1):
            out.row_index[row_out + 1] += out.row_index[row_out]
    return out

cpdef CSR column_stack_csr(CSR matrix):
    if matrix.shape[1] == 1:
        return matrix.copy()
    return reshape_csr(matrix.transpose(), matrix.shape[0]*matrix.shape[1], 1)


cpdef CSR column_unstack_csr(CSR matrix, idxint rows):
    if matrix.shape[1] != 1:
        raise ValueError("input is not a single column")
    if matrix.shape[0] % rows:
        raise ValueError("number of rows does not divide into the shape")
    cdef idxint cols = matrix.shape[0] // rows
    return reshape_csr(matrix, cols, rows).transpose()
