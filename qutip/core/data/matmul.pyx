#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset, memcpy

cimport cython
import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint, Data
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data cimport csr

cnp.import_array()

cdef extern from "../cy/src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                 double complex a, double complex *out, int nrows)


cdef void _check_shape(Data left, Data right, Data out=None) nogil except *:
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    if (
        out is not None
        and out.shape[0] != left.shape[0]
        and out.shape[1] != right.shape[1]
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[1]))
        )


cdef idxint _matmul_csr_estimate_nnz(CSR left, CSR right) nogil:
    """
    Produce a sensible upper-bound for the number of non-zero elements that
    will be present in a matrix multiplication between the two matrices.
    """
    cdef idxint j, k, nnz=0
    cdef idxint ii, jj, kk
    cdef idxint nrows=left.shape[0], ncols=right.shape[1]
    # Setup mask array
    cdef idxint *mask = <idxint *> malloc(ncols * sizeof(idxint))
    for ii in range(ncols):
        mask[ii] = -1
    for ii in range(nrows):
        for jj in range(left.row_index[ii], left.row_index[ii+1]):
            j = left.col_index[jj]
            for kk in range(right.row_index[j], right.row_index[j+1]):
                k = right.col_index[kk]
                if mask[k] != ii:
                    mask[k] = ii
                    nnz += 1
    free(mask)
    return nnz


cpdef CSR matmul_csr(CSR left, CSR right, CSR out=None, double complex scale=1.0):
    """
    Multiply two CSR matrices together to produce another CSR.  If `out` is
    specified, it must be pre-allocated with enough space to hold the output
    result.

    This is the operation
        ``out := left @ right``
    where `out` will be allocated if not supplied.

    Parameters
    ----------
    left : CSR
        CSR matrix on the left of the multiplication.
    right : CSR
        CSR matrix on the right of the multiplication.
    out : optional CSR
        Allocated space to store the result.  This must have enough space in
        the `data`, `col_index` and `row_index` pointers already allocated.

    Returns
    -------
    out : CSR
        The result of the matrix multiplication.  This will be the same object
        as the input parameter `out` if that was supplied.
    """
    _check_shape(left, right)
    cdef idxint nnz = _matmul_csr_estimate_nnz(left, right)
    if out is not None:
        raise TypeError("passing an `out` entry for CSR operations makes no sense")
    out = csr.empty(left.shape[0], right.shape[1], nnz if nnz != 0 else 1)
    if nnz == 0 or csr.nnz(left) == 0 or csr.nnz(right) == 0:
        # Ensure the out array row_index is zeroed.  The others need not be,
        # because they don't represent valid entries since row_index is zeroed.
        with nogil:
            memset(&out.row_index[0], 0, (out.shape[0] + 1) * sizeof(idxint))
        return out

    # Initialise actual matrix multiplication.
    nnz = 0
    cdef idxint head, length, row_l, ptr_l, row_r, ptr_r, col_r, tmp
    cdef idxint nrows=left.shape[0], ncols=right.shape[1]
    cdef double complex val
    cdef double complex *sums
    cdef idxint *nxt
    with nogil:
        sums = <double complex *> calloc(ncols, sizeof(double complex))
        nxt = <idxint *> malloc(ncols * sizeof(idxint))
        for col_r in range(ncols):
            nxt[col_r] = -1

        # Perform operation.
        out.row_index[0] = 0
        for row_l in range(nrows):
            head = -2
            length = 0
            for ptr_l in range(left.row_index[row_l], left.row_index[row_l+1]):
                row_r = left.col_index[ptr_l]
                val = left.data[ptr_l]
                for ptr_r in range(right.row_index[row_r], right.row_index[row_r+1]):
                    col_r = right.col_index[ptr_r]
                    sums[col_r] += val * right.data[ptr_r]
                    if nxt[col_r] == -1:
                        nxt[col_r] = head
                        head = col_r
                        length += 1
            for col_r in range(length):
                if sums[head] != 0:
                    out.col_index[nnz] = head
                    out.data[nnz] = scale * sums[head]
                    nnz += 1
                tmp = head
                head = nxt[head]
                nxt[tmp] = -1
                sums[tmp] = 0
            out.row_index[row_l + 1] = nnz
        free(sums)
        free(nxt)
    return out


