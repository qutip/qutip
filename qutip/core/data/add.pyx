#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint, Data
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data cimport csr

cnp.import_array()

cdef extern from *:
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_FREE(void *ptr)


cdef void _check_shape(Data left, Data right) nogil except *:
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )


cdef idxint _add_csr(CSR a, CSR b, CSR c) nogil:
    """
    Perform the operation
        c := a + b
    for CSR matrices, where `a` and `b` are guaranteed to be the correct shape,
    and `c` already has enough space allocated (at least nnz(a)+nnz(b)).

    Return the true value of nnz(c).
    """
    cdef:
        idxint nrows=c.shape[0], ncols=c.shape[1]
        idxint row, col_a, col_b
        double complex tmp
        # These all refer to "pointers" into the col_index or data arrays.
        idxint ptr_a, ptr_b, ptr_c=0, max_ptr_a, max_ptr_b
    c.row_index[0] = 0
    for row in range(nrows):
        ptr_a = a.row_index[row]
        ptr_b = b.row_index[row]
        max_ptr_a = a.row_index[row+1] - 1
        max_ptr_b = b.row_index[row+1] - 1
        while (ptr_a <= max_ptr_a) or (ptr_b <= max_ptr_b):
            col_a = a.col_index[ptr_a] if ptr_a <= max_ptr_a else ncols + 1
            col_b = b.col_index[ptr_b] if ptr_b <= max_ptr_b else ncols + 1
            if col_a < col_b:
                c.data[ptr_c] = a.data[ptr_a]
                c.col_index[ptr_c] = col_a
                ptr_a += 1
                ptr_c += 1
            elif col_b < col_a:
                c.data[ptr_c] = b.data[ptr_b]
                c.col_index[ptr_c] = col_b
                ptr_b += 1
                ptr_c += 1
            else:  # equal
                tmp = a.data[ptr_a] + b.data[ptr_b]
                if tmp != 0:
                    c.data[ptr_c] = tmp
                    c.col_index[ptr_c] = col_a
                    ptr_c += 1
                ptr_a += 1
                ptr_b += 1
        c.row_index[row+1] = ptr_c
    return ptr_c


cdef idxint _add_csr_scale(CSR a, CSR b, CSR c, double complex scale) nogil:
    """
    Perform the operation
        c := a + scale*b
    for CSR matrices, where `a` and `b` are guaranteed to be the correct shape,
    and `c` already has enough space allocated (at least nnz(a)+nnz(b)).

    Return the true value of nnz(c).
    """
    cdef:
        idxint nrows=c.shape[0], ncols=c.shape[1]
        idxint row, col_a, col_b
        double complex tmp
        # These all refer to "pointers" into the col_index or data arrays.
        idxint ptr_a, ptr_b, ptr_c=0, max_ptr_a, max_ptr_b
    c.row_index[0] = 0
    for row in range(nrows):
        ptr_a = a.row_index[row]
        ptr_b = b.row_index[row]
        max_ptr_a = a.row_index[row+1] - 1
        max_ptr_b = b.row_index[row+1] - 1
        while (ptr_a <= max_ptr_a) or (ptr_b <= max_ptr_b):
            col_a = a.col_index[ptr_a] if ptr_a <= max_ptr_a else ncols + 1
            col_b = b.col_index[ptr_b] if ptr_b <= max_ptr_b else ncols + 1
            if col_a < col_b:
                c.data[ptr_c] = a.data[ptr_a]
                c.col_index[ptr_c] = col_a
                ptr_a += 1
                ptr_c += 1
            elif col_b < col_a:
                c.data[ptr_c] = scale * b.data[ptr_b]
                c.col_index[ptr_c] = col_b
                ptr_b += 1
                ptr_c += 1
            else:  # equal
                tmp = a.data[ptr_a] + scale*b.data[ptr_b]
                if tmp != 0:
                    c.data[ptr_c] = tmp
                    c.col_index[ptr_c] = col_a
                    ptr_c += 1
                ptr_a += 1
                ptr_b += 1
        c.row_index[row+1] = ptr_c
    return ptr_c


cpdef CSR add_csr(CSR left, CSR right, double complex scale=1):
    """
    Matrix addition of `left` and `right` for CSR inputs and output.  If given,
    `right` is multiplied by `scale`, so the full operation is
        ``out := left + scale*right``
    The two matrices must be of exactly the same shape.

    Parameters
    ----------
    left : CSR
        Matrix to be added.
    right : CSR
        Matrix to be added.  If `scale` is given, this matrix will be
        multiplied by `scale` before addition.
    scale : optional double complex (1)
        The scalar value to multiply `right` by before addition.

    Returns
    -------
    out : CSR
        The result `left + scale*right`.
    """
    _check_shape(left, right)
    cdef idxint left_nnz = csr.nnz(left)
    cdef idxint right_nnz = csr.nnz(right)
    cdef idxint worst_nnz = left_nnz + right_nnz
    cdef idxint i
    cdef CSR out
    # Fast paths for zero matrices.
    if right_nnz == 0:
        return left.copy()
    if left_nnz == 0:
        out = right.copy()
        # Fast path if the multiplication is a no-op.
        if scale != 1:
            for i in range(right_nnz):
                out.data[i] *= scale
        return out
    # Main path.
    out = csr.empty(left.shape[0], left.shape[1], worst_nnz)
    if scale == 1:
        _add_csr(left, right, out)
    else:
        _add_csr_scale(left, right, out, scale)
    return out
