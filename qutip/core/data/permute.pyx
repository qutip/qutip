#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdlib cimport malloc, calloc, realloc, free
from libc.string cimport memset, memcpy

from libcpp cimport bool
from libcpp.algorithm cimport sort

cimport cython

import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint, idxint_dtype, idxint_DTYPE
from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR

cnp.import_array()

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
        cdef bint *tmp = <bint *> calloc(self.ndims, sizeof(bint))
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
                new_dimensions[i] = dim
        finally:
            free(tmp)
        self.cumprod = <idxint *> malloc(self.ndims * sizeof(idxint))
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
            out += self.cumprod[i] * (idx % dim)
            idx //= dim
            if idx == 0:
                break
        return out

    def __dealloc__(self):
        if self.cumprod != NULL:
            free(self.cumprod)

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
    cdef bint *test = <bint *>calloc(size, sizeof(bint))
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
        free(test)

cdef void _permute_csr_sparse(CSR out, CSR matrix, _Indexer index) nogil:
    cdef size_t row, i, ptr_in, ptr_out, diff, max_cols=0
    memset(&out.row_index[0], 0, out.shape[0] * sizeof(idxint))
    for row in range(matrix.shape[0]):
        diff = matrix.row_index[row + 1] - matrix.row_index[row]
        if diff:
            out.row_index[index.single(row) + 1] = diff
            max_cols = diff if diff > max_cols else max_cols
    for row in range(out.shape[0]):
        out.row_index[row + 1] += out.row_index[row]
    for row in range(matrix.shape[0]):
        ptr_in = matrix.row_index[row]
        diff = matrix.row_index[row + 1] - ptr_in
        if diff == 0:
            continue
        ptr_out = out.row_index[index.single(row)]

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
    cdef idxint *new_cols = <idxint *> malloc(len * sizeof(idxint))
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
    free(new_cols)
    return out

cpdef CSR indices_csr(CSR matrix, object row_perm, object col_perm):
    if row_perm is None and col_perm is None:
        return matrix.copy()
    if col_perm is None:
        return _indices_csr_rowonly(matrix, np.asarray(row_perm, dtype=idxint_dtype))
    cdef idxint *rows = NULL
    cdef idxint n
    if row_perm is None:
        rows = <idxint *> malloc(matrix.shape[0] * sizeof(idxint))
        for n in range(matrix.shape[0]):
            rows[n] = n
        try:
            return _indices_csr_full(matrix,
                                     <idxint [:matrix.shape[0]]>rows,
                                     np.asarray(col_perm, dtype=idxint_dtype))
        finally:
            free(rows)
    return _indices_csr_full(matrix,
                             np.asarray(row_perm, dtype=idxint_dtype),
                             np.asarray(col_perm, dtype=idxint_dtype))


cdef CSR _dimensions_csr_columns(CSR matrix, _Indexer index):
    if matrix.shape[0] != 1:
        raise ValueError("expected bra-like matrix")
    cdef size_t nnz = csr.nnz(matrix)
    cdef CSR out = csr.empty_like(matrix)
    out.row_index[0] = 0
    out.row_index[1] = nnz
    cdef size_t n
    cdef csr.Sorter sort = csr.Sorter(nnz)
    cdef idxint *new_cols = <idxint *> malloc(nnz * sizeof(idxint))
    try:
        for n in range(nnz):
            new_cols[n] = index.single(matrix.col_index[n])
        sort.copy(&out.data[0], &out.col_index[0], &matrix.data[0], new_cols, nnz)
        return out
    finally:
        free(new_cols)

cdef CSR _dimensions_csr_sparse(CSR matrix, _Indexer index):
    cdef CSR out = csr.empty_like(matrix)
    cdef size_t row, n, len=0
    cdef idxint ptr_in, ptr_out
    memset(&out.row_index[0], 0, (matrix.shape[0] + 1) * sizeof(idxint))
    for row in range(matrix.shape[0]):
        n = matrix.row_index[row + 1] - matrix.row_index[row]
        if n:
            out.row_index[index.single(row) + 1] = len
        len = n if n > len else len
    for row in range(matrix.shape[0]):
        out.row_index[row + 1] += out.row_index[row]
    # Since this is very sparse, we expect almost all rows to have at most two
    # elements in them.  It will be faster to copy them across, and perform the
    # sort in place rather than allocating temporary space and making an
    # additional copy.  This will still work even if there are more in a row,
    # it just won't be quite as efficient in that case (which should be rare).
    cdef csr.Sorter sort = csr.Sorter(len)
    for row in range(matrix.shape[0]):
        ptr_in = matrix.row_index[row]
        len = matrix.row_index[row + 1] - ptr_in
        if len == 0:
            continue
        ptr_out = out.row_index[index.single(row)]
        for n in range(len):
            out.col_index[ptr_out + n] = index.single(matrix.col_index[ptr_in + n])
        memcpy(&out.data[ptr_out], &matrix.data[ptr_in], len*sizeof(double complex))
        sort.inplace(out, ptr_out, len)
    return out

@cython.cdivision(True)
cpdef CSR dimensions_csr(CSR matrix, object dimensions, object order):
    cdef _Indexer index = _Indexer(np.asarray(dimensions, dtype=idxint_dtype),
                                   np.asarray(order, dtype=idxint_dtype))
    cdef idxint[:] permutation
    if matrix.shape[0] == 1 and matrix.shape[1] == 1:
        return matrix.copy()
    if matrix.shape[0] == 1:
        return _dimensions_csr_columns(matrix, index)
    if matrix.shape[1] == 1:
        return _indices_csr_rowonly(matrix, index.all())
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("dimensional permute requires square operators")
    if (matrix.shape[0] * matrix.shape[1]) // csr.nnz(matrix) > 0:
        permutation = index.all()
        return _indices_csr_full(matrix, permutation, permutation)
    return _dimensions_csr_sparse(matrix, index)
