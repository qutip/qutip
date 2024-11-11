#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memset, memcpy
from libc.math cimport fabs
from libc.stdlib cimport abs
from libcpp.algorithm cimport lower_bound

import warnings
from qutip.settings import settings

cimport cython

from cpython cimport mem

import numpy as np
cimport numpy as cnp
from scipy.linalg cimport cython_blas as blas

from qutip.core.data.base import idxint_dtype
from qutip.core.data.base cimport idxint, Data
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data.dia cimport Dia
from qutip.core.data.tidyup cimport tidyup_dia
from qutip.core.data cimport csr, dense, dia
from qutip.core.data.add cimport iadd_dense, add_csr
from qutip.core.data.mul cimport imul_dense
from qutip.core.data.dense import OrderEfficiencyWarning

cnp.import_array()

cdef extern from *:
    void *PyMem_Calloc(size_t n, size_t elsize)

# This function is templated over integral types on import to allow `idxint` to
# be any signed integer (though likely things will only work for >=32-bit).  To
# change integral types, you only need to change the `idxint` definitions in
# `core.data.base` at compile-time.
cdef extern from "src/matmul_csr_vector.hpp" nogil:
    void _matmul_csr_vector[T](
        double complex *data, T *col_index, T *row_index,
        double complex *vec, double complex scale, double complex *out,
        T nrows)

cdef extern from "src/matmul_diag_vector.hpp" nogil:
    void _matmul_diag_vector[T](
        double complex *data, double complex *vec, double complex *out,
        T length, double complex scale)
    void _matmul_diag_block[T](
        double complex *data, double complex *vec, double complex *out,
        T length, T width)


__all__ = [
    'matmul', 'matmul_csr', 'matmul_dense', 'matmul_dia',
    'matmul_csr_dense_dense', 'matmul_dia_dense_dense', 'matmul_dense_dia_dense',
    'multiply', 'multiply_csr', 'multiply_dense', 'multiply_dia',
]


cdef int _check_shape(Data left, Data right, Data out=None) except -1 nogil:
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    if (
        out is not None
        and (
            out.shape[0] != left.shape[0]
            or out.shape[1] != right.shape[1]
        )
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[1]))
        )
    return 0

cdef idxint _matmul_csr_estimate_nnz(CSR left, CSR right):
    """
    Produce a sensible upper-bound for the number of non-zero elements that
    will be present in a matrix multiplication between the two matrices.
    """
    cdef idxint j, k, nnz=0
    cdef idxint ii, jj, kk
    cdef idxint nrows=left.shape[0], ncols=right.shape[1]
    # Setup mask array
    cdef idxint *mask = <idxint *> mem.PyMem_Malloc(ncols * sizeof(idxint))
    with nogil:
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
    mem.PyMem_Free(mask)
    return nnz


cpdef CSR matmul_csr(CSR left, CSR right, double complex scale=1, CSR out=None):
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
    cdef double tol = 0
    if settings.core['auto_tidyup']:
        tol = settings.core['auto_tidyup_atol']
    sums = <double complex *> PyMem_Calloc(ncols, sizeof(double complex))
    nxt = <idxint *> mem.PyMem_Malloc(ncols * sizeof(idxint))
    with nogil:
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
                if fabs(sums[head].real) < tol:
                    sums[head].real = 0
                if fabs(sums[head].imag) < tol:
                    sums[head].imag = 0
                if sums[head] != 0:
                    out.col_index[nnz] = head
                    out.data[nnz] = scale * sums[head]
                    nnz += 1
                tmp = head
                head = nxt[head]
                nxt[tmp] = -1
                sums[tmp] = 0
            out.row_index[row_l + 1] = nnz
    mem.PyMem_Free(sums)
    mem.PyMem_Free(nxt)
    return out


cpdef Dense matmul_csr_dense_dense(CSR left, Dense right,
                                   double complex scale=1, Dense out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.

    If `out` is not given, it will be allocated as if it were a zero matrix.
    """
    _check_shape(left, right, out)
    cdef Dense tmp = None
    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)
    if bool(right.fortran) != bool(out.fortran):
        msg = (
            "out matrix is {}-ordered".format('Fortran' if out.fortran else 'C')
            + " but input is {}-ordered".format('Fortran' if right.fortran else 'C')
        )
        warnings.warn(msg, OrderEfficiencyWarning)
        # Rather than making loads of copies of the same code, we just moan at
        # the user and then transpose one of the arrays.  We prefer to have
        # `right` in Fortran-order for cache efficiency.
        if right.fortran:
            tmp = out
            out = out.reorder()
        else:
            right = right.reorder()
    cdef idxint row, ptr, idx_r, idx_out, nrows=left.shape[0], ncols=right.shape[1]
    cdef double complex val
    if right.fortran:
        idx_r = idx_out = 0
        for _ in range(ncols):
            _matmul_csr_vector(left.data, left.col_index, left.row_index,
                               right.data + idx_r,
                               scale,
                               out.data + idx_out,
                               nrows)
            idx_out += nrows
            idx_r += right.shape[0]
    else:
        for row in range(nrows):
            for ptr in range(left.row_index[row], left.row_index[row + 1]):
                val = scale * left.data[ptr]
                idx_out = row * ncols
                idx_r = left.col_index[ptr] * ncols
                for _ in range(ncols):
                    out.data[idx_out] += val * right.data[idx_r]
                    idx_out += 1
                    idx_r += 1
    if tmp is None:
        return out
    memcpy(tmp.data, out.data, ncols * nrows * sizeof(double complex))
    return tmp


cpdef Dense matmul_dense(Dense left, Dense right, double complex scale=1, Dense out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.

    If `out` is not given, it will be allocated as if it were a zero matrix.
    """
    _check_shape(left, right, out)
    cdef double complex out_scale
    # If not supplied, it's more efficient from a memory allocation perspective
    # to do the calculation as `a*A.B + 0*C` with arbitrary C.
    if out is None:
        out = dense.empty(left.shape[0], right.shape[1], right.fortran)
        out_scale = 0
    else:
        out_scale = 1
    cdef double complex *a
    cdef double complex *b
    cdef char transa, transb
    cdef int m, n, k=left.shape[1], lda, ldb
    if right.shape[1] == 1:
        # Matrix Vector product
        a, b = left.data, right.data
        if left.fortran:
            lda = left.shape[0]
            transa = b'n'
            m = left.shape[0]
            n = left.shape[1]
        else:
            lda = left.shape[1]
            transa = b't'
            m = left.shape[1]
            n = left.shape[0]
        ldb = 1
        blas.zgemv(&transa, &m , &n, &scale, a, &lda, b, &ldb,
                   &out_scale, out.data, &ldb)
        return out
    # We use the BLAS routine zgemm for every single call and pretend that
    # we're always supplying it with Fortran-ordered matrices, but to achieve
    # what we want, we use the property of matrix multiplication that
    #   A.B = (B'.A')'
    # where ' is the matrix transpose, and that interpreting a Fortran-ordered
    # matrix as a C-ordered one is equivalent to taking the transpose.  If
    # `right` is supplied in C-order, then from Fortran's perspective we
    # actually have `B'`, so to retrieve `B` should we want to use it, we set
    # `transb = b't'`.  What we set `transa` and `transb` to depends on if we
    # need to switch the input order, _not_ whether we actually need B'.
    #
    # In order to make the output correct, we ensure that we put A.B in if the
    # output is Fortran ordered, or B'.A' (note no final transpose) if not.
    # This is actually more flexible than `np.dot` which requires that the
    # output is C-ordered.
    if out.fortran:
        # Need to make A.B
        a, b = left.data, right.data
        m, n = left.shape[0], right.shape[1]
        lda = left.shape[0] if left.fortran else left.shape[1]
        transa = b'n' if left.fortran else b't'
        ldb = right.shape[0] if right.fortran else right.shape[1]
        transb = b'n' if right.fortran else b't'
    else:
        # Need to make B'.A'
        a, b = right.data, left.data
        m, n = right.shape[1], left.shape[0]
        lda = right.shape[0] if right.fortran else right.shape[1]
        transa = b't' if right.fortran else b'n'
        ldb = left.shape[0] if left.fortran else left.shape[1]
        transb = b't' if left.fortran else b'n'
    blas.zgemm(&transa, &transb, &m, &n, &k, &scale, a, &lda, b, &ldb,
               &out_scale, out.data, &m)
    return out


cpdef Dia matmul_dia(Dia left, Dia right, double complex scale=1):
    _check_shape(left, right, None)
    # We could probably do faster than this...
    npoffsets = np.unique(np.add.outer(left.as_scipy().offsets, right.as_scipy().offsets))
    npoffsets = npoffsets[np.logical_and(npoffsets > -left.shape[0], npoffsets < right.shape[1])]
    cdef idxint[:] offsets = npoffsets
    if len(npoffsets) == 0:
        return dia.zeros(left.shape[0], right.shape[1])
    cdef idxint *ptr = &offsets[0]
    cdef size_t num_diag = offsets.shape[0], diag_out, diag_left, diag_right
    cdef idxint start_left, end_left, start_out, end_out, start, end, col, off_out
    npdata = np.zeros((num_diag, right.shape[1]), dtype=complex)
    cdef double complex[:, ::1] data = npdata

    with nogil:
      for diag_left in range(left.num_diag):
        for diag_right in range(right.num_diag):
          off_out = left.offsets[diag_left] + right.offsets[diag_right]
          if off_out <= -left.shape[0] or off_out >= right.shape[1]:
            continue
          diag_out = <idxint> (lower_bound(ptr, ptr + num_diag, off_out) - ptr)

          start_left = max(0, left.offsets[diag_left]) + right.offsets[diag_right]
          start_right = max(0, right.offsets[diag_right])
          start_out = max(0, off_out)
          end_left = min(left.shape[1], left.shape[0] + left.offsets[diag_left]) + right.offsets[diag_right]
          end_right = min(right.shape[1], right.shape[0] + right.offsets[diag_right])
          end_out = min(right.shape[1], left.shape[0] + off_out)
          start = max(start_left, start_right, start_out)
          end = min(end_left, end_right, end_out)

          for col in range(start, end):
              data[diag_out, col] += (
                scale
                * left.data[diag_left * left.shape[1] + col - right.offsets[diag_right]]
                * right.data[diag_right * right.shape[1] + col]
              )
    return Dia((npdata, npoffsets), shape=(left.shape[0], right.shape[1]), copy=False)


cpdef Dense matmul_dia_dense_dense(Dia left, Dense right, double complex scale=1, Dense out=None):
    _check_shape(left, right, out)
    cdef Dense tmp
    if out is not None and scale == 1.:
        tmp = out
        out = None
    else:
        tmp = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    cdef idxint start_left, end_left, start_out, end_out, length, i, start_right
    cdef idxint col, strideR_in, strideC_in, strideR_out, strideC_out
    cdef size_t diag

    with nogil:
      strideR_in = right.shape[1] if not right.fortran else 1
      strideC_in = right.shape[0] if right.fortran else 1
      strideR_out = tmp.shape[1] if not tmp.fortran else 1
      strideC_out = tmp.shape[0] if tmp.fortran else 1

      if (
        (left.shape[0] == left.shape[1])
        and (strideC_in == 1)
        and (strideC_out == 1)
      ):
          #Fast track for easy case
          for diag in range(left.num_diag):
              _matmul_diag_block(
                  right.data + max(0, left.offsets[diag]) * strideR_in,
                  left.data + diag * left.shape[1] + max(0, left.offsets[diag]),
                  tmp.data + max(0, -left.offsets[diag]) * strideR_out,
                  left.shape[1] - abs(left.offsets[diag]),
                  right.shape[1]
              )

      elif (strideR_in == 1) and (strideR_out == 1):
        for col in range(right.shape[1]):
          for diag in range(left.num_diag):
            start_left = max(0, left.offsets[diag])
            end_left = min(left.shape[1], left.shape[0] + left.offsets[diag])
            start_out = max(0, -left.offsets[diag])
            end_out = min(left.shape[0], left.shape[1] - left.offsets[diag])
            length = min(end_left - start_left, end_out - start_out)
            start_right = start_left + col * strideC_in
            start_left += diag * left.shape[1]
            start_out += col * strideC_out
            _matmul_diag_vector(
                left.data + start_left,
                right.data + start_right,
                tmp.data + start_out,
                length, 1.
            )

      else:
        for col in range(right.shape[1]):
          for diag in range(left.num_diag):
            start_left = max(0, left.offsets[diag])
            end_left = min(left.shape[1], left.shape[0] + left.offsets[diag])
            start_out = max(0, -left.offsets[diag])
            end_out = min(left.shape[0], left.shape[1] - left.offsets[diag])
            length = min(end_left - start_left, end_out - start_out)
            for i in range(length):
              tmp.data[(start_out + i) * strideR_out + col * strideC_out] += (
                left.data[diag * left.shape[1] + i + start_left]
                * right.data[(start_left + i) * strideR_in + col * strideC_in]
              )

    if out is None and scale == 1.:
        out = tmp
    elif out is None:
        imul_dense(tmp, scale)
        out = tmp
    else:
        iadd_dense(out, tmp, scale)

    return out


cpdef Dense matmul_dense_dia_dense(Dense left, Dia right, double complex scale=1, Dense out=None):
    _check_shape(left, right, out)
    cdef Dense tmp
    if out is not None and scale == 1.:
        tmp = out
        out = None
    else:
        tmp = dense.zeros(left.shape[0], right.shape[1], left.fortran)

    cdef idxint start_left, end_right, start_out, end_out, length, i, start_right
    cdef idxint row, strideR_in, strideC_in, strideR_out, strideC_out
    cdef size_t diag

    with nogil:
      strideR_in = left.shape[1] if not left.fortran else 1
      strideC_in = left.shape[0] if left.fortran else 1
      strideR_out = tmp.shape[1] if not tmp.fortran else 1
      strideC_out = tmp.shape[0] if tmp.fortran else 1

      if (
        (right.shape[0] == right.shape[1])
        and (strideR_in == 1)
        and (strideR_out == 1)
      ):
          #Fast track for easy case
          for diag in range(right.num_diag):
              _matmul_diag_block(
                  left.data + max(0, -right.offsets[diag]) * strideC_in,
                  right.data + diag * right.shape[1] + max(0, right.offsets[diag]),
                  tmp.data + max(0, right.offsets[diag]) * strideC_out,
                  right.shape[1] - abs(right.offsets[diag]),
                  left.shape[0]
              )

      elif (strideC_in == 1) and (strideC_out == 1):
        for row in range(left.shape[0]):
          for diag in range(right.num_diag):
            start_right = max(0, right.offsets[diag])
            end_right = min(right.shape[1], right.shape[0] + right.offsets[diag])
            start_out = max(0, right.offsets[diag])
            length = end_right - start_right
            start_left = max(0, -right.offsets[diag]) + row * strideR_in
            start_right += diag * right.shape[1]
            start_out = max(0, right.offsets[diag]) + row * strideR_out
            _matmul_diag_vector(
                right.data + start_right,
                left.data + start_left,
                tmp.data + start_out,
                length, 1.
            )

      else:
        for row in range(left.shape[0]):
          for diag in range(right.num_diag):
            start_right = max(0, right.offsets[diag])
            end_right = min(right.shape[1], right.shape[0] + right.offsets[diag])
            start_left = max(0, -right.offsets[diag])
            length = end_right - start_right
            for i in range(length):
              tmp.data[(start_right + i) * strideC_out + row * strideR_out] += (
                right.data[diag * right.shape[1] + i + start_right]
                * left.data[(start_left + i) * strideC_in + row * strideR_in]
              )

    if out is None and scale == 1.:
        out = tmp
    elif out is None:
        imul_dense(tmp, scale)
        out = tmp
    else:
        iadd_dense(out, tmp, scale)

    return out


cpdef CSR multiply_csr(CSR left, CSR right):
    """Element-wise multiplication of CSR matrices."""
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )

    left = left.sort_indices()
    right = right.sort_indices()

    cdef idxint col_left, left_nnz = csr.nnz(left)
    cdef idxint col_right, right_nnz = csr.nnz(right)
    cdef idxint ptr_left, ptr_right, ptr_left_max, ptr_right_max
    cdef idxint row, nnz=0, ncols=left.shape[1]
    cdef CSR out
    cdef list nans=[]
    # Fast paths for zero matrices.
    if right_nnz == 0 or left_nnz == 0:
        return csr.zeros(left.shape[0], left.shape[1])
    # Main path.
    out = csr.empty(left.shape[0], left.shape[1], max(left_nnz, right_nnz))
    out.row_index[0] = nnz
    ptr_left_max = ptr_right_max = 0

    for row in range(left.shape[0]):
        ptr_left = ptr_left_max
        ptr_left_max = left.row_index[row + 1]
        ptr_right = ptr_right_max
        ptr_right_max = right.row_index[row + 1]
        while ptr_left < ptr_left_max or ptr_right < ptr_right_max:
            col_left = left.col_index[ptr_left] if ptr_left < ptr_left_max else ncols + 1
            col_right = right.col_index[ptr_right] if ptr_right < ptr_right_max else ncols + 1
            if col_left == col_right:
                out.col_index[nnz] = col_left
                out.data[nnz] = left.data[ptr_left] * right.data[ptr_right]
                ptr_left += 1
                ptr_right += 1
                nnz += 1
            elif col_left <= col_right:
                if left.data[ptr_left] is np.nan:
                    # Test for NaN since `NaN * 0 = NaN`
                    nans.append((row, col_left))
                ptr_left += 1
            else:
                if right.data[ptr_right] != right.data[ptr_right]:
                    nans.append((row, col_right))
                ptr_right += 1
        out.row_index[row + 1] = nnz
    if nans:
        # We expect Nan to be rare enough that we don't allocate memory for
        # them, but add them here after the loop.
        nans_pos = np.array(nans, order='F', dtype=idxint_dtype)
        nnz = nans_pos.shape[0]
        nans_csr = csr.from_coo_pointers(
            <idxint *> cnp.PyArray_GETPTR2(nans_pos, 0, 0),
            <idxint *> cnp.PyArray_GETPTR2(nans_pos, 0, 1),
            <double complex *> cnp.PyArray_GETPTR1(
                np.array([np.nan]*nnz, dtype=np.complex128), 0),
            left.shape[0], left.shape[1], nnz
        )
        out = add_csr(out, nans_csr)
    return out


cpdef Dia multiply_dia(Dia left, Dia right):
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    cdef idxint diag_left=0, diag_right=0, out_diag=0, col
    cdef bint sorted=True
    cdef Dia out = dia.empty(left.shape[0], left.shape[1], min(left.num_diag, right.num_diag))

    with nogil:
      for diag_left in range(1, left.num_diag):
          if left.offsets[diag_left-1] > left.offsets[diag_left]:
              sorted = False
              continue
      if sorted:
          for diag_right in range(1, right.num_diag):
              if right.offsets[diag_right-1] > right.offsets[diag_right]:
                  sorted = False
                  continue

      if sorted:
        diag_left = 0
        diag_right = 0
        while diag_left < left.num_diag and diag_right < right.num_diag:
            if left.offsets[diag_left] == right.offsets[diag_right]:
                out.offsets[out_diag] = left.offsets[diag_left]
                for col in range(out.shape[1]):
                    if col >= left.shape[1] or col >= right.shape[1]:
                        out.data[out_diag * out.shape[1] + col] = 0
                    else:
                        out.data[out_diag * out.shape[1] + col] = (
                            left.data[diag_left * left.shape[1] + col] *
                            right.data[diag_right * right.shape[1] + col]
                        )
                out_diag += 1
                diag_left += 1
                diag_right += 1
            elif left.offsets[diag_left] < right.offsets[diag_right]:
                diag_left += 1
            else:
                diag_right += 1

      else:
        for diag_left in range(left.num_diag):
          for diag_right in range(right.num_diag):
            if left.offsets[diag_left] == right.offsets[diag_right]:
                out.offsets[out_diag] = left.offsets[diag_left]
                for col in range(right.shape[1]):
                    out.data[out_diag * out.shape[1] + col] = (
                        left.data[diag_left * left.shape[1] + col] *
                        right.data[diag_right * right.shape[1] + col]
                    )
                out_diag += 1
                break
      out.num_diag = out_diag

    if settings.core['auto_tidyup']:
        tidyup_dia(out, settings.core['auto_tidyup_atol'], True)
    return out


cpdef Dense multiply_dense(Dense left, Dense right):
    """Element-wise multiplication of Dense matrices."""
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    return Dense(left.as_ndarray() * right.as_ndarray(), copy=False)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

# At the time of putting in the dispatchers, the idea of an "out" parameter
# isn't really supported in the model; since `out` would be potentially
# modified in-place, it couldn't safely go through a conversion process.  For
# the dispatched operation, then, we omit the `scale` and `out` parameters, and
# only dispatch on the operation `a @ b`.  If the `out` and `scale` parameters
# are needed, the library will have to manually do any relevant conversions,
# and then call a direct specialisation (which are exported to the `data`
# namespace).

matmul = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('scale', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=1),
    ]),
    name='matmul',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
matmul.__doc__ =\
    """
    Compute the matrix multiplication of two matrices, with the operation
        scale * (left @ right)
    where `scale` is (optionally) a scalar, and `left` and `right` are
    matrices.

    Parameters
    ----------
    left : Data
        The left operand as either a bra or a ket matrix.

    right : Data
        The right operand as a ket matrix.

    scale : complex, optional
        The scalar to multiply the output by.
    """
matmul.add_specialisations([
    (CSR, CSR, CSR, matmul_csr),
    (CSR, Dense, Dense, matmul_csr_dense_dense),
    (Dense, Dense, Dense, matmul_dense),
    (Dia, Dia, Dia, matmul_dia),
    (Dia, Dense, Dense, matmul_dia_dense_dense),
    (Dense, Dia, Dense, matmul_dense_dia_dense),
], _defer=True)


multiply = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='multiply',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
multiply.__doc__ =\
    """Element-wise multiplication of matrices."""
multiply.add_specialisations([
    (CSR, CSR, CSR, multiply_csr),
    (Dense, Dense, Dense, multiply_dense),
    (Dia, Dia, Dia, multiply_dia),
], _defer=True)


del _inspect, _Dispatcher


cdef Dense matmul_data_dense(Data left, Dense right):
    cdef Dense out
    if type(left) is CSR:
        out = matmul_csr_dense_dense(left, right)
    elif type(left) is Dense:
        out = matmul_dense(left, right)
    elif type(left) is Dia:
        out = matmul_dia_dense_dense(left, right)
    else:
        out = matmul(left, right)
    return out


cdef void imatmul_data_dense(Data left, Dense right, double complex scale, Dense out):
    if type(left) is CSR:
        matmul_csr_dense_dense(left, right, scale, out)
    elif type(left) is Dia:
        matmul_dia_dense_dense(left, right, scale, out)
    elif type(left) is Dense:
        matmul_dense(left, right, scale, out)
    else:
        iadd_dense(out, matmul(left, right, dtype=Dense), scale)
