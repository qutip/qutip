#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

from libcpp cimport bool
from libcpp.algorithm cimport sort

import numpy as np

from qutip.core.data.base cimport idxint, idxint_dtype, Data
from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR

cdef bint _check_permutation(Data matrix, idxint[:] rows, idxint[:] cols) except True:
    cdef size_t ptr, n_rows=matrix.shape[0], n_cols=matrix.shape[1]
    if rows.shape[0] != n_rows or cols.shape[0] != n_cols:
        raise ValueError("invalid permutation: wrong number of elements")
    cdef size_t size = n_rows if n_rows > n_cols else n_cols
    cdef idxint value
    cdef bint *test = <bint *>calloc(size, sizeof(bint))
    if test == NULL:
        raise MemoryError
    try:
        for ptr in range(n_rows):
            value = rows[ptr]
            if not 0 <= value < n_rows:
                raise ValueError("invalid entry in permutation: " + str(value))
            if test[value]:
                raise ValueError("duplicate entry in permutation: " + str(value))
            test[value] = True
        memset(test, 0, n_rows * sizeof(bint))
        for ptr in range(n_cols):
            value = cols[ptr]
            if not 0 <= value < n_rows:
                raise ValueError("invalid entry in permutation: " + str(value))
            if test[value]:
                raise ValueError("duplicate entry in permutation: " + str(value))
            test[value] = True
        return False
    finally:
        free(test)

cdef bool _permute_ptr_cmp(idxint *i, idxint *j):
    return i[0] < j[0]

cpdef CSR permute_csr(CSR matrix, object row_perm, object col_perm):
    cdef idxint [:] rows = np.asarray(row_perm, dtype=idxint_dtype)
    cdef idxint [:] cols = np.asarray(col_perm, dtype=idxint_dtype)
    _check_permutation(matrix, rows, cols)
    cdef size_t n_rows=matrix.shape[0], n_cols=matrix.shape[1]
    cdef size_t nnz=csr.nnz(matrix)
    cdef CSR out = csr.empty(n_rows, n_cols, nnz)
    cdef size_t row, ptr, ptr_in, ptr_out, len, n
    # First build up the row index structure by cumulative sum, so we know
    # where to place the data and column indices.  We also use this opporunity
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
    cdef idxint **argsort = <idxint **> malloc(len * sizeof(idxint *))
    if argsort == NULL or new_cols == NULL:
        raise MemoryError
    for row in range(matrix.shape[0]):
        ptr_in = matrix.row_index[row]
        ptr_out = out.row_index[rows[row]]
        len = matrix.row_index[row + 1] - ptr_in
        # We do the argsort with two levels of indirection to minimise memory
        # allocation and copying requirements.  First, for each non-zero
        # element, we put its _new_ column into an array, in the same order
        # they appear in `matrix.col_index`.  Then we put pointers to each of
        # those columns in the array which actually gets sorted, and use a
        # comparison function which dereferences the pointers and compares the
        # result.  After the sort, `argsort` will be the pointers sorted
        # according to the new column, and we know that the "lowest" pointer in
        # there has the value `new_cols`, so we can do pointer arithmetic to
        # know which element we should take.
        #
        # This is about 30-40% faster than allocating space for structs of
        # (double complex, idxint), copying in the data and column, sorting and
        # copying into the new arrays.  Allocating the structs actually
        # allocates more space than the pointer method (double complex is
        # very likely to be 2x the size of a pointer, _and_ the struct may need
        # extra padding to be aligned), so it's probably actually worse for
        # cache locality.  Despite the sort relying on pointer dereference in
        # this case, it's actually got very good cache locality.
        for n in range(len):
            new_cols[n] = cols[matrix.col_index[ptr_in + n]]
            argsort[n] = new_cols + n
        sort(argsort, argsort + len, _permute_ptr_cmp)
        for n in range(len):
            # ptr is not strictly a pointer itself---it's an offset, but that's
            # consistent with how the name is used in other CSR functions.
            ptr = argsort[n] - new_cols
            out.data[ptr_out + n] = matrix.data[ptr_in + ptr]
            out.col_index[ptr_out + n] = new_cols[ptr]
    free(argsort)
    free(new_cols)
    return out
