#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.string cimport memcpy, memset
from scipy.linalg cimport cython_blas as blas
cimport cython

import warnings

from qutip.core.data.base cimport idxint
from qutip.core.data cimport csr, dense, CSR, Dense, Data, Dia

__all__ = [
    'reshape', 'reshape_csr', 'reshape_dense', 'reshape_dia',
    'column_stack', 'column_stack_csr', 'column_stack_dense', 'column_stack_dia',
    'column_unstack', 'column_unstack_csr', 'column_unstack_dense', 'column_unstack_dia',
    'split_columns', 'split_columns_dense', 'split_columns_csr', 'split_columns_dia',
]


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
                out.row_index[loc // n_cols_out + 1] += 1
                out.col_index[ptr] = loc % n_cols_out
            cur += n_cols_in
        for row_out in range(n_rows_out):
            out.row_index[row_out + 1] += out.row_index[row_out]
    return out


cdef inline size_t _reshape_dense_reindex(size_t idx, size_t size):
    return (idx // size) + (idx % size)


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
    cdef size_t tmp = (<size_t> matrix.shape[1]) * (<size_t> n_rows_out)
    # TODO: improve the algorithm here.
    cdef size_t stride = _reshape_dense_reindex(tmp, size)
    for idx_in in range(size):
        out.data[idx_out] = matrix.data[idx_in]
        idx_out = _reshape_dense_reindex(idx_out + stride, size)
    return out


cpdef Dia reshape_dia(Dia matrix, idxint n_rows_out, idxint n_cols_out):
    _reshape_check_input(matrix, n_rows_out, n_cols_out)
    # Once reshaped, diagonals are no longer ligned up.
    return Dia(
        matrix.as_scipy().reshape((n_rows_out, n_cols_out)).todia(), copy=False
    )


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
    out = dense.zeros(matrix.shape[0] * matrix.shape[1], 1)
    cdef idxint col
    cdef int ONE=1
    for col in range(matrix.shape[1]):
        blas.zcopy(
            &matrix.shape[0],
            &matrix.data[col], &matrix.shape[1],
            &out.data[col * matrix.shape[0]], &ONE
        )
    return out


cpdef Dia column_stack_dia(Dia matrix):
    if matrix.shape[1] == 1:
        return matrix.copy()
    return reshape_dia(matrix.transpose(), matrix.shape[0]*matrix.shape[1], 1)


cdef void _column_unstack_check_shape(Data matrix, idxint rows) except *:
    if matrix.shape[1] != 1:
        raise ValueError("input is not a single column")
    if rows < 1:
        raise ValueError("rows must be a positive integer")
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


cpdef Dia column_unstack_dia(Dia matrix, idxint rows):
    _column_unstack_check_shape(matrix, rows)
    cdef idxint cols = matrix.shape[0] // rows
    return reshape_dia(matrix, cols, rows).transpose()


cpdef list split_columns_dense(Dense matrix, copy=True):
    return [Dense(matrix.as_ndarray()[:, k], copy=copy)
            for k in range(matrix.shape[1])]


cpdef list split_columns_csr(CSR matrix, copy=True):
    return [CSR(matrix.as_scipy()[:, k], copy=copy)
            for k in range(matrix.shape[1])]


cpdef list split_columns_dia(Dia matrix, copy=None):
    as_array = matrix.to_array()
    return [Dense(as_array[:, k], copy=False) for k in range(matrix.shape[1])]


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

reshape = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('n_rows_out', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('n_cols_out', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='reshape',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
reshape.__doc__ =\
    """
    Reshape the input matrix.  The values of `n_rows_out` and `n_cols_out` must
    match the current total number of elements of the matrix.

    Parameters
    ----------
    matrix : Data
        The input matrix to reshape.

    n_rows_out, n_cols_out : integer
        The number of rows and columns in the output matrix.
    """
reshape.add_specialisations([
    (CSR, CSR, reshape_csr),
    (Dense, Dense, reshape_dense),
    (Dia, Dia, reshape_dia),
], _defer=True)


# Similar to the `out` parameter of `matmul`, we don't include `inplace` in the
# signature of `column_stack` because the dispatcher logic currently doesn't
# support the idea of an input parameter also being the output.

column_stack = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='column_stack',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
column_stack.__doc__ =\
    """
    Stack the columns of a matrix so it becomes a ket-like vector.  For
    example, the matrix
        [[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]]
    would be transformed to
        [[0],
         [3],
         [6],
         [1],
         ...
         [5],
         [8]]
    This is used for transforming between operator and operator-ket
    representations in the super-operator formalism.

    The inverse of this operation is `column_unstack`.

    Parameters
    ----------
    matrix : Data
        The matrix to stack the columns of.
    """
column_stack.add_specialisations([
    (CSR, CSR, column_stack_csr),
    (Dense, Dense, column_stack_dense),
    (Dia, Dia, column_stack_dia),
], _defer=True)

column_unstack = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('rows', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='column_unstack',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
column_unstack.__doc__ =\
    """
    Unstack the columns of a ket-like vector so it becomes a matrix with `rows`
    number of rows.  For example, the matrix
        [[0],
         [1],
         [2],
         [3]]
    would be unstacked with `rows = 2` to
        [[0, 2],
         [1, 3]]
    This is used for transforming between the operator-ket and operator
    representations in the super-operator formalism.

    The inverse of this operation is `column_stack`.

    Parameters
    ----------
    matrix : Data
        The matrix to unstack the columns of.

    rows : integer
        The number of rows there should be in the output matrix.  This must
        divide into the total number of elements in the input.
    """
column_unstack.add_specialisations([
    (CSR, CSR, column_unstack_csr),
    (Dense, Dense, column_unstack_dense),
    (Dia, Dia, column_unstack_dia),
], _defer=True)


split_columns = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('copy', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=1),
    ]),
    name='split_columns',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
split_columns.__doc__ =\
    """
    Make a ket-shaped data out of each columns of the matrix.
    This is used for to split the eigenvectors from :obj:`eigs`.

    Parameters
    ----------
    matrix : Data
        The matrix to unstack the columns of.

    copy : bool, optional [True]
        Whether to return copy of the data or a view.
        View may not be a possible in all cases.

    Returns
    -------
    kets : list
        list of Data of each columns of the matrix.
    """
split_columns.add_specialisations([
    (CSR, split_columns_csr),
    (Dense, split_columns_dense),
    (Dia, split_columns_dia),
], _defer=True)

del _inspect, _Dispatcher
