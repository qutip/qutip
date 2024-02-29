#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memset, memcpy

from libcpp cimport bool
from libcpp.algorithm cimport sort

cimport cython

from cpython cimport mem

import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint, idxint_DTYPE
from qutip.core.data cimport csr, dense, CSR, Dense

from qutip.core.data.base import idxint_dtype

cnp.import_array()

cdef extern from *:
    void *PyMem_Calloc(size_t n, size_t elsize)

# This module is meant to be used with dot-access (e.g. `permute.dimensions`).
__all__ = []


cdef class _Indexer:
    cdef size_t ndims, size
    cdef readonly cnp.ndarray new_dimensions
    cdef idxint[:] dimensions
    cdef idxint *cumprod

    def __init__(self, idxint[:] dimensions, idxint[:] order):
        cdef size_t i
        cdef idxint dim, ord
        cdef idxint prev
        self.ndims = dimensions.shape[0]
        self.dimensions = dimensions
        self.new_dimensions = cnp.PyArray_EMPTY(1, [self.ndims], idxint_DTYPE, False)
        cdef idxint[:] new_dimensions = self.new_dimensions
        if order.shape[0] != self.ndims:
            raise ValueError("invalid order: wrong number of elements")
        cdef bint *tmp = <bint *> PyMem_Calloc(self.ndims, sizeof(bint))
        try:
            for i in range(self.ndims):
                ord = order[i]
                if ord < 0 or ord >= self.ndims:
                    raise ValueError("invalid order element: " + str(ord))
                if tmp[ord]:
                    raise ValueError("duplicate order element: " + str(ord))
                tmp[ord] = True
                dim = self.dimensions[ord]
                if dim < 0:
                    raise ValueError("found negative dimension: " + str(dim))
                elif dim == 0:
                    raise ValueError("found zero dimension")
                new_dimensions[i] = dim
        finally:
            mem.PyMem_Free(tmp)
        self.cumprod = <idxint *> mem.PyMem_Malloc(self.ndims * sizeof(idxint))
        prev = self.cumprod[order[self.ndims - 1]] = 1
        for i in range(self.ndims - 2, -1, -1):
            prev = self.cumprod[order[i]] = prev * new_dimensions[i + 1]
        self.size = self.cumprod[order[0]] * new_dimensions[0]

    @cython.cdivision(True)
    cdef cnp.ndarray all(self):
        cdef object out = cnp.PyArray_EMPTY(1, [self.size], idxint_DTYPE, False)
        cdef idxint *_out = <idxint *> cnp.PyArray_GETPTR1(out, 0)
        cdef idxint i
        for i in range(self.size):
            _out[i] = self.single(i)
        return out

    @cython.cdivision(True)
    cdef idxint single(self, idxint idx) nogil:
        cdef size_t i
        cdef idxint out=0, dim
        for i in range(self.ndims - 1, -1, -1):
            dim = self.dimensions[i]
            # Dimensions cannot be zero due to the check in __init__.
            out += self.cumprod[i] * (idx % dim)
            idx //= dim
            if idx == 0:
                break
        return out

    def __dealloc__(self):
        if self.cumprod != NULL:
            mem.PyMem_Free(self.cumprod)

cdef bint _check_indices(size_t size, idxint[:] order) except True:
    """
    Test whether the permutation `order` is a valid permutation of `size`
    number of elements.  This is functionally equivalent to

    ```
    if np.sort(order) != np.arange(size):
        raise ValueError
    return False
    ```

    In other words, we test that each integer on [0, size) is present exactly
    once in `order`, and raise ValueError if that is not the case (returns True
    if an error is detected, and False if not to help pure Cython avoiding
    exceptions).
    """
    if order.shape[0] != size:
        raise ValueError("invalid permutation: wrong number of elements")
    cdef size_t ptr
    cdef idxint value
    cdef bint *test = <bint *> PyMem_Calloc(size, sizeof(bint))
    if test == NULL:
        raise MemoryError
    try:
        for ptr in range(size):
            value = order[ptr]
            if not 0 <= value < size:
                raise ValueError("invalid entry in permutation: " + str(value))
            if test[value]:
                raise ValueError("duplicate entry in permutation: " + str(value))
            test[value] = True
        return False
    finally:
        mem.PyMem_Free(test)

cdef CSR _indices_csr_rowonly(CSR matrix, idxint[:] rows):
    cdef size_t n_rows=matrix.shape[0]
    _check_indices(n_rows, rows)
    cdef CSR out = csr.empty_like(matrix)
    cdef size_t row, ptr_in, ptr_out, len
    out.row_index[0] = 0
    for row in range(matrix.shape[0]):
        out.row_index[rows[row] + 1] =\
            matrix.row_index[row + 1] - matrix.row_index[row]
    for row in range(matrix.shape[0]):
        out.row_index[row + 1] += out.row_index[row]
    for row in range(matrix.shape[0]):
        ptr_in = matrix.row_index[row]
        ptr_out = out.row_index[rows[row]]
        len = matrix.row_index[row + 1] - ptr_in
        memcpy(&out.col_index[ptr_out], &matrix.col_index[ptr_in], len * sizeof(idxint))
        memcpy(&out.data[ptr_out], &matrix.data[ptr_in], len * sizeof(double complex))
    return out

cdef CSR _indices_csr_full(CSR matrix, idxint[:] rows, idxint[:] cols):
    _check_indices(matrix.shape[0], rows)
    _check_indices(matrix.shape[1], cols)
    cdef CSR out = csr.empty_like(matrix)
    cdef size_t row, ptr_in, ptr_out, len, n
    # First build up the row index structure by cumulative sum, so we know
    # where to place the data and column indices.  We also use this opportunity
    # to find the maximum number of non-zero elements in a row.
    with nogil:
        len = 0
        out.row_index[0] = 0
        for row in range(matrix.shape[0]):
            n = matrix.row_index[row + 1] - matrix.row_index[row]
            out.row_index[rows[row] + 1] = n
            len = n if n > len else len
        for row in range(matrix.shape[0]):
            out.row_index[row + 1] += out.row_index[row]
    # Now we know that `len` is the most number of non-zero elements in a row,
    # so we can allocate space to sort only once.
    cdef idxint *new_cols = <idxint *> mem.PyMem_Malloc(len * sizeof(idxint))
    cdef csr.Sorter sort = csr.Sorter(len)
    for row in range(matrix.shape[0]):
        ptr_in = matrix.row_index[row]
        ptr_out = out.row_index[rows[row]]
        len = matrix.row_index[row + 1] - ptr_in
        for n in range(len):
            new_cols[n] = cols[matrix.col_index[ptr_in + n]]
        sort.copy(&out.data[ptr_out], &out.col_index[ptr_out],
                  &matrix.data[ptr_in], new_cols,
                  len)
    mem.PyMem_Free(new_cols)
    return out

cpdef CSR indices_csr(CSR matrix, object row_perm=None, object col_perm=None):
    if row_perm is None and col_perm is None:
        return matrix.copy()
    if col_perm is None:
        return _indices_csr_rowonly(matrix, np.asarray(row_perm, dtype=idxint_dtype))
    cdef idxint *rows = NULL
    cdef idxint n
    if row_perm is None:
        rows = <idxint *> mem.PyMem_Malloc(matrix.shape[0] * sizeof(idxint))
        for n in range(matrix.shape[0]):
            rows[n] = n
        try:
            return _indices_csr_full(matrix,
                                     <idxint [:matrix.shape[0]]>rows,
                                     np.asarray(col_perm, dtype=idxint_dtype))
        finally:
            mem.PyMem_Free(rows)
    return _indices_csr_full(matrix,
                             np.asarray(row_perm, dtype=idxint_dtype),
                             np.asarray(col_perm, dtype=idxint_dtype))

cpdef Dense indices_dense(Dense matrix, object row_perm=None, object col_perm=None):
    if row_perm is None and col_perm is None:
        return matrix.copy()
    array = matrix.as_ndarray()
    if row_perm is not None:
        array = array[np.argsort(row_perm), :]
    if col_perm is not None:
        array = array[:, np.argsort(col_perm)]
    return Dense(array)


cdef CSR _dimensions_csr_columns(CSR matrix, _Indexer index):
    if matrix.shape[0] != 1:
        raise ValueError("expected bra-like matrix")
    cdef size_t nnz = csr.nnz(matrix)
    cdef CSR out = csr.empty_like(matrix)
    out.row_index[0] = 0
    out.row_index[1] = nnz
    cdef size_t n
    cdef csr.Sorter sort = csr.Sorter(nnz)
    cdef idxint *new_cols = <idxint *> mem.PyMem_Malloc(nnz * sizeof(idxint))
    try:
        for n in range(nnz):
            new_cols[n] = index.single(matrix.col_index[n])
        sort.copy(&out.data[0], &out.col_index[0], &matrix.data[0], new_cols, nnz)
        return out
    finally:
        mem.PyMem_Free(new_cols)

cdef CSR _dimensions_csr_sparse(CSR matrix, _Indexer index):
    cdef CSR out = csr.empty_like(matrix)
    cdef csr.Sorter sort
    cdef size_t row, n, len=0
    cdef idxint ptr_in, ptr_out, col
    cdef idxint *idx_lookup = <idxint *> mem.PyMem_Malloc(matrix.shape[0] * sizeof(idxint))
    try:
        memset(&out.row_index[0], 0, (matrix.shape[0] + 1) * sizeof(idxint))
        with nogil:
            for row in range(matrix.shape[0]):
                n = matrix.row_index[row + 1] - matrix.row_index[row]
                if n:
                    idx_lookup[row] = index.single(row)
                    out.row_index[idx_lookup[row] + 1] = n
                    len = n if n > len else len
                else:
                    # Use a sentinel value so we can avoid looking up columns
                    # that we already know about later.  Not all values in
                    # idx_lookup will even be filled---this is the speed up
                    # this function achieves over `_indices_csr_all`.
                    idx_lookup[row] = -1
            for row in range(matrix.shape[0]):
                out.row_index[row + 1] += out.row_index[row]
        # Since this is very sparse, we expect almost all rows to have at most two
        # elements in them.  It will be faster to copy them across, and perform the
        # sort in place rather than allocating temporary space and making an
        # additional copy.  This will still work even if there are more in a row,
        # it just won't be quite as efficient in that case (which should be rare).
        sort = csr.Sorter(len)
        for row in range(matrix.shape[0]):
            ptr_in = matrix.row_index[row]
            len = matrix.row_index[row + 1] - ptr_in
            if len == 0:
                continue
            ptr_out = out.row_index[idx_lookup[row]]
            for n in range(len):
                col = matrix.col_index[ptr_in + n]
                if idx_lookup[col] == -1:
                    idx_lookup[col] = index.single(col)
                out.col_index[ptr_out + n] = idx_lookup[col]
            memcpy(&out.data[ptr_out], &matrix.data[ptr_in], len*sizeof(double complex))
            sort.inplace(out, ptr_out, len)
        return out
    finally:
        mem.PyMem_Free(idx_lookup)

@cython.cdivision(True)
cpdef CSR dimensions_csr(CSR matrix, object dimensions, object order):
    cdef _Indexer index = _Indexer(np.asarray(dimensions, dtype=idxint_dtype),
                                   np.asarray(order, dtype=idxint_dtype))
    cdef idxint[:] permutation
    if matrix.shape[0] == 1 and matrix.shape[1] == 1 or csr.nnz(matrix) == 0:
        return matrix.copy()
    if matrix.shape[0] == 1:
        return _dimensions_csr_columns(matrix, index)
    if matrix.shape[1] == 1:
        return _indices_csr_rowonly(matrix, index.all())
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("dimensional permute requires square operators")
    cdef double row_density = (<double> csr.nnz(matrix)) / (<double> matrix.shape[0])
    # The speed-up for _dimensions_csr_sparse is only achieved by having fewer
    # calls to `index.single()` than the matrix dimension.  To be sure of this,
    # we actually require the density per row to be less than 1/2, because we
    # have to look up both the row _and_ column on output.  This only
    # corresponds to exceptionally sparse matrices.  We try to avoid these
    # calls because index.single has ~logarithmic complexity in the dimension
    # (so for qubit systems it's linear in the number of qubits), and
    # consequently `index.all()` is a hidden quadratic complexity.
    if row_density >= 0.5:
        permutation = index.all()
        return _indices_csr_full(matrix, permutation, permutation)
    return _dimensions_csr_sparse(matrix, index)

@cython.cdivision(True)
cpdef Dense dimensions_dense(Dense matrix, object dimensions, object order):
    cdef _Indexer index = _Indexer(np.asarray(dimensions, dtype=idxint_dtype),
                                   np.asarray(order, dtype=idxint_dtype))
    cdef idxint[:] permutation = index.all()
    row_perm, col_perm = None, None
    if matrix.shape[0] != 1:
        row_perm = permutation
    if matrix.shape[1] != 1:
        col_perm = permutation
    return indices_dense(matrix, row_perm, col_perm)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

dimensions = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('dimensions', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('order', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='dimensions',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
dimensions.__doc__ =\
    """
    Reorder the tensor-product structure of a matrix, assuming that the
    underlying structure is defined by `dimensions`.  For a separable system,
    this function produces a matrix which is equivalent to having performed
    `kron` in a different order on the separable parts.

    For example if `a`, `b` and `c` are matrices with sizes 2, 3 and 4
    respectively, then
        kron(kron(c, a), b) == permute.dimensions(kron(kron(a, b), c),
                                                  [2, 3, 4],
                                                  [1, 2, 0])
    In other words, the inputs to `kron` are reordered so that input `n` moves
    to position `order[n]`.

    Parameters
    ----------
    matrix : Data
        Input matrix to reorder.  This can either be a square matrix
        representing an operator, or a bra- or ket-like vector.

    dimensions : 1D array_like of integers
        The tensor-product structure of the space the matrix lives on.  This
        will typically be one of the two elements of `Qobj.dims` (e.g. for a
        ket, it will be `Qobj.dims[0]`).

    order : 1D array_like of integers
        The new order of the tensor-product elements.  This should be a 1D list
        with the integers from `0` to `N - 1` inclusive, if there are `N`
        elements in the tensor product.
    """
dimensions.add_specialisations([
    (CSR, CSR, dimensions_csr),
    (Dense, Dense, dimensions_dense),
], _defer=True)

indices = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('row_perm', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=None),
        _inspect.Parameter('col_perm', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=None),
    ]),
    name='indices',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
indices.__doc__ =\
    """
    Permute the rows and columns of a matrix according to a row and column
    permutation.  This is a "dumb" operation with regards to the representation
    of quantum states; if you want to "reorder" the tensor-product structure of
    a system, you want `permute.dimensions` instead.

    Parameters
    ----------
    matrix : Data
        The input matrix.

    row_perm, col_perm : 1D array_like of integer, optional
        The new order that the rows or columns should be shuffled into.  If the
        input matrix is `N x M`, then `row_perm` would be an array containing
        all the integers from `0` to `N - 1` inclusive in some new order.  Row
        `n` in the input will be at row `row_perm[n]` in the output, and
        similar for the column permutation.
    """
indices.add_specialisations([
    (CSR, CSR, indices_csr),
    (Dense, Dense, indices_dense),
], _defer=True)

del _inspect, _Dispatcher
