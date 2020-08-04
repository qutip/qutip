#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.stdlib cimport div, div_t
from libc.string cimport memcpy, memset

cimport cython

import warnings

from qutip.core.data.base cimport idxint
from qutip.core.data cimport csr, dense, CSR, Dense, Data

cdef void _reshape_check_input(Data matrix, idxint n_rows_out, idxint n_cols_out) except *:
    if n_rows_out * n_cols_out != matrix.shape[0] * matrix.shape[1]:
        message = "".join([
            "cannot reshape ", str(matrix.shape), " to ",
            "(", str(n_rows_out), ", ", str(n_cols_out), ")",
        ])
        raise ValueError(message)
    if n_rows_out <= 0 or n_cols_out <= 0:
        raise ValueError("must have > 0 rows and columns")


cpdef CSR reshape_csr(CSR matrix, idxint n_rows_out, idxint n_cols_out):
    cdef size_t ptr, row_in, row_out=0, loc, cur=0
    cdef size_t n_rows_in=matrix.shape[0], n_cols_in=matrix.shape[1]
    cdef idxint nnz = csr.nnz(matrix)
    cdef div_t res
    cdef CSR out
    _reshape_check_input(matrix, n_rows_out, n_cols_out)
    out = csr.empty(n_rows_out, n_cols_out, nnz)
    matrix.sort_indices()
    with nogil:
        # Since the indices are now sorted, the data arrays will be identical.
        memcpy(out.data, matrix.data, nnz*sizeof(double complex))
        memset(out.row_index, 0, (n_rows_out + 1) * sizeof(idxint))
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
        for row_out in range(n_rows_out):
            out.row_index[row_out + 1] += out.row_index[row_out]
    return out


# We have to use a signed integer type because the standard library doesn't
# provide overloads for unsigned types.
cdef inline idxint _reshape_dense_reindex(idxint idx, idxint size):
    cdef div_t res = div(idx, size)
    return res.quot + res.rem

cpdef Dense reshape_dense(Dense matrix, idxint n_rows_out, idxint n_cols_out):
    _reshape_check_input(matrix, n_rows_out, n_cols_out)
    cdef Dense out
    if not matrix.fortran:
        out = matrix.copy()
        out.shape = (n_rows_out, n_cols_out)
        return out
    out = dense.zeros(n_rows_out, n_cols_out)
    cdef size_t idx_in=0, idx_out=0
    cdef size_t size = n_rows_out * n_cols_out
    # TODO: improve the algorithm here.
    cdef size_t stride = _reshape_dense_reindex(matrix.shape[1]*n_rows_out, size)
    for idx_in in range(size):
        out.data[idx_out] = matrix.data[idx_in]
        idx_out = _reshape_dense_reindex(idx_out + stride, size)
    return out


cpdef CSR column_stack_csr(CSR matrix):
    if matrix.shape[1] == 1:
        return matrix.copy()
    return reshape_csr(matrix.transpose(), matrix.shape[0]*matrix.shape[1], 1)


cpdef Dense column_stack_dense(Dense matrix, bint inplace=False):
    cdef Dense out
    if inplace and matrix.fortran:
        matrix.shape = (matrix.shape[0] * matrix.shape[1], 1)
        return matrix
    if matrix.fortran:
        out = matrix.copy()
        out.shape = (matrix.shape[0]*matrix.shape[1], 1)
        return out
    if inplace:
        warnings.warn("cannot stack columns inplace for C-ordered matrix")
    return reshape_dense(matrix.transpose(), matrix.shape[0]*matrix.shape[1], 1)


cdef void _column_unstack_check_shape(Data matrix, idxint rows) except *:
    if matrix.shape[1] != 1:
        raise ValueError("input is not a single column")
    if matrix.shape[0] % rows:
        raise ValueError("number of rows does not divide into the shape")


cpdef CSR column_unstack_csr(CSR matrix, idxint rows):
    _column_unstack_check_shape(matrix, rows)
    cdef idxint cols = matrix.shape[0] // rows
    return reshape_csr(matrix, cols, rows).transpose()

cpdef Dense column_unstack_dense(Dense matrix, idxint rows, bint inplace=False):
    _column_unstack_check_shape(matrix, rows)
    cdef idxint cols = matrix.shape[0] // rows
    if inplace and matrix.fortran:
        matrix.shape = (rows, cols)
        return matrix
    elif inplace:
        warnings.warn("cannot unstack columns inplace for C-ordered matrix")
    out = dense.empty(rows, cols, fortran=True)
    memcpy(out.data, matrix.data, rows*cols * sizeof(double complex))
    return out
