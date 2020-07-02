#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

cimport cython
import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data cimport csr

cnp.import_array()

cdef extern from "../cy/src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                 double complex a, double complex *out, int nrows)


cdef void mv_csr(CSR matrix, double complex *vector, double complex *out) nogil:
    """
    Perform the operation
        ``out := (matrix @ vector) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    """
    # TODO: the cast <int *> is NOT SAFE as base.idxint is _not_ guaranteed to
    # match the size of `int`.  Must be changed.
    zspmvpy(&matrix.data[0],
            <int *> &matrix.col_index[0],
            <int *> &matrix.row_index[0],
            vector,
            1.0,
            out,
            matrix.shape[0])
    return


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


cpdef CSR matmul_csr(CSR left, CSR right):
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
    # We assume the shapes are compatible, since this is a C routine.
    cdef idxint nnz = _matmul_csr_estimate_nnz(left, right)
    cdef CSR out = csr.empty(left.shape[0], right.shape[1], nnz if nnz != 0 else 1)
    if nnz == 0 or csr.nnz(left) == 0 or csr.nnz(right) == 0:
        # Ensure the out array row_index is zeroed.  The others need not be,
        # because they don't represent valid entries since row_index is zeroed.
        with nogil:
            memset(&out.row_index[0], 0, (out.shape[0] + 1) * sizeof(idxint))
        return out

    # Initialise actual matrix multiplication.
    nnz = 0
    cdef idxint head, length, temp, j, k, ii, jj, kk
    cdef idxint nrows=left.shape[0], ncols=right.shape[1]
    cdef double complex val
    cdef double complex *sums
    cdef idxint *nxt
    with nogil:
        sums = <double complex *> calloc(ncols, sizeof(double complex))
        nxt = <idxint *> malloc(ncols * sizeof(idxint))
        for ii in range(ncols):
            nxt[ii] = -1

        # Perform operation.
        out.row_index[0] = 0
        for ii in range(nrows):
            head = -2
            length = 0
            for jj in range(left.row_index[ii], left.row_index[ii+1]):
                j = left.col_index[jj]
                val = left.data[jj]
                for kk in range(right.row_index[j], right.row_index[j+1]):
                    k = right.col_index[kk]
                    sums[k] += val * right.data[kk]
                    if nxt[k] == -1:
                        nxt[k] = head
                        head = k
                        length += 1
            for jj in range(length):
                if sums[head] != 0:
                    out.col_index[nnz] = head
                    out.data[nnz] = sums[head]
                    nnz += 1
                temp = head
                head = nxt[head]
                nxt[temp] = -1
                sums[temp] = 0
            out.row_index[ii+1] = nnz
        # Free temp arrays
        free(sums)
        free(nxt)
    return out


def matmul_vector_csr(CSR matrix, object vector, object out=None):
    """
    Perform the matrix-vector operation
        y := A.x + y
    where `A` is the CSR matrix `matrix`, x is a `numpy.ndarray` `vector`, and
    `y` is optionally the output location `out`, to which the answer is added.
    If `y` is not supplied, a suitable zero vector will be initialised.

    Parameters
    ----------
    matrix : `CSR`
        The matrix used.

    vector : `numpy.ndarray[complex]`
        The vector to be multiplied.  Will be made contiguous in memory if it
        is not already so.

    out : optional `numpy.ndarray[complex]`
        The array the answer should be output into.  The result will be added
        on to the contents of this array, so this array should be zeroed if
        required.  If not supplied, a suitable zero array will be allocated.
        Must be contiguous in memory.

    Returns
    -------
    `numpy.ndarray[complex]`
        The object `out` if it was supplied, otherwise the allocated
        `numpy.ndarray` containing the result.  This is guaranteed to be
        contiguous.
    """
    cdef object vec = np.ascontiguousarray(vector, np.complex128)
    if cnp.PyArray_SIZE(vec) != matrix.shape[1]:
        raise ValueError(
            "incompatible shapes ({}, {}) and ({},)."\
                .format(matrix.shape[0], matrix.shape[1], cnp.PyArray_SIZE(vec))
        )
    if out is None:
        out = cnp.PyArray_ZEROS(1, [matrix.shape[0]], cnp.NPY_COMPLEX128, False)
    elif not (isinstance(out, np.ndarray)
              and cnp.PyArray_TYPE(out) == cnp.NPY_COMPLEX128
              and cnp.PyArray_DIMS(out)[0] >= matrix.shape[0]
              and cnp.PyArray_IS_C_CONTIGUOUS(out)):
        raise ValueError(
            "output array must be a large enough contiguous ndarray of complex."
        )
    mv_csr(matrix,
           <double complex *> cnp.PyArray_GETPTR1(vec, 0),
           <double complex *> cnp.PyArray_GETPTR1(out, 0))
    return out
