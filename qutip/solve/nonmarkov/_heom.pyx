#cython: language_level=3
#cython: boundscheck=False, wraparound=False

from libc.string cimport memcpy, memset
from qutip.core.data cimport idxint, CSR, csr

def pad_csr(CSR matrix, idxint row_scale, idxint col_scale,
            idxint insertrow=0, idxint insertcol=0):
    cdef idxint n_rows_in = matrix.shape[0]
    cdef idxint n_cols_in = matrix.shape[1]
    cdef idxint n_rows_out = n_rows_in * row_scale
    cdef idxint n_cols_out = n_cols_in * col_scale
    cdef idxint temp, ptr
    cdef size_t nnz = csr.nnz(matrix)
    cdef CSR out = csr.empty(n_rows_out, n_cols_out, nnz)

    memcpy(out.data, matrix.data, nnz * sizeof(double complex))
    if insertcol == 0:
        memcpy(out.col_index, matrix.col_index, nnz * sizeof(idxint))
    elif insertcol > 0 and insertcol < col_scale:
        temp = insertcol * n_cols_in
        for ptr in range(nnz):
            out.col_index[ptr] = matrix.col_index[ptr] + temp
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")

    if insertrow == 0:
        memcpy(out.row_index, matrix.row_index, n_rows_in * sizeof(idxint))
        temp = matrix.row_index[n_rows_in]
        for ptr in range(n_rows_in, n_rows_out + 1):
            out.row_index[ptr] = temp

    elif insertrow == row_scale - 1:
        temp = insertrow * n_rows_in
        memset(out.row_index, 0, temp * sizeof(idxint))
        memcpy(out.row_index + temp, matrix.row_index,
               (n_rows_out + 1 - temp) * sizeof(idxint))

    elif insertrow > 0 and insertrow < row_scale - 1:
        temp = insertrow * n_rows_in
        memset(out.row_index, 0, temp * sizeof(idxint))
        memcpy(out.row_index + temp, matrix.row_index, n_rows_in * sizeof(idxint))
        for ptr in range(temp + n_rows_in, n_rows_out + 1):
            out.row_index[ptr] = nnz
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    return out


from qutip.core.data import Dispatcher
import inspect as _inspect

pad = Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('rowscale', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('colscale', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('insertrow', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('insertcol', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='pad',
    module=__name__,
    inputs=('matrix',),
    out=True)
pad.add_specialisations([
    (CSR, CSR, pad_csr),
])

del Dispatcher, _inspect
