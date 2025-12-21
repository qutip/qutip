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


cdef extern from "<complex>" namespace "std" nogil:
    double complex conj "conj"(double complex x)

# This function is templated over integral types on import to allow `idxint` to
# be any signed integer (though likely things will only work for >=32-bit).  To
# change integral types, you only need to change the `idxint` definitions in
# `core.data.base` at compile-time.
cdef extern from "src/matmul_csr_vector.hpp" nogil:
    void _matmul_csr_vector[T](
        double complex *data, T *col_index, T *row_index,
        double complex *vec, double complex scale, double complex *out,
        T nrows)
    void _matmul_dag_csr_vector[T](
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
    'matmul_dag', 'matmul_dag_data', 'matmul_dag_dense', 'matmul_dag_dense_csr_dense',
    'matmul_outer',
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

    if left.num_diag == 0 or right.num_diag == 0:
        return dia.zeros(left.shape[0], right.shape[1])

    cdef size_t diag_out, diag_left, diag_right
    cdef idxint *offsets
    cdef idxint *ptr
    cdef double complex *data
    cdef double complex ZZERO=0.
    cdef idxint col, row, ii, num_diag_out, offset
    cdef idxint start_left, end_left, start_out, end_out, start, end, off_out
    cdef idxint start_right, end_right
    cdef int ONE=1, ZERO=0, num_elem

    out = dia.empty(
        left.shape[0], right.shape[1],
        min(left.shape[0] + right.shape[1] - 1, left.num_diag * right.num_diag)
    )

    # Fill the output matrix's offsets, sorted and without redundancies.
    offsets = out.offsets
    num_diag_out = 0
    for col in range(left.num_diag):
      for row in range(right.num_diag):
        offset = left.offsets[col] + right.offsets[row]
        if not (offset > -left.shape[0] and offset < right.shape[1]):
          continue
        loc = num_diag_out
        for ii in range(num_diag_out):
            if offsets[ii] == offset:
                loc = -1
                break
            if offsets[ii] > offset:
                loc = ii
                break
        if loc >= 0:
            for ii in range(num_diag_out, loc, -1):
                offsets[ii] = offsets[ii - 1]
            offsets[loc] = offset
            num_diag_out += 1

    out.num_diag = num_diag_out
    data = out.data
    ptr = &offsets[0]

    with nogil:
      num_elem = num_diag_out * out.shape[1]
      blas.zcopy(&num_elem, &ZZERO, &ZERO, data, &ONE)

      for diag_left in range(left.num_diag):
        for diag_right in range(right.num_diag):
          off_out = left.offsets[diag_left] + right.offsets[diag_right]
          if off_out <= -left.shape[0] or off_out >= right.shape[1]:
            continue

          diag_out = <idxint> (lower_bound(ptr, ptr + num_diag_out, off_out) - ptr)

          start_left = max(0, left.offsets[diag_left]) + right.offsets[diag_right]
          start_right = max(0, right.offsets[diag_right])
          start_out = max(0, off_out)
          end_left = min(left.shape[1], left.shape[0] + left.offsets[diag_left]) + right.offsets[diag_right]
          end_right = min(right.shape[1], right.shape[0] + right.offsets[diag_right])
          end_out = min(right.shape[1], left.shape[0] + off_out)

          start = max(start_left, start_right, start_out)
          end = min(end_left, end_right, end_out)

          for col in range(start, end):
              data[diag_out * out.shape[1] + col] += (
                scale
                * left.data[diag_left * left.shape[1] + col - right.offsets[diag_right]]
                * right.data[diag_right * right.shape[1] + col]
              )

    return out


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


cpdef Dense matmul_dag_dense_csr_dense(
    Dense left, CSR right,
    double complex scale=1, Dense out=None
):
    """
    Perform the operation
        ``out = scale * (left @ right.dag()) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.

    If `out` is not given, it will be allocated as if it were a zero matrix.
    """
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    if (
        out is not None and (
            out.shape[0] != left.shape[0]
            or out.shape[1] != right.shape[0]
        )
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[0]))
        )
    cdef Dense tmp = None
    if out is None:
        out = dense.zeros(left.shape[0], right.shape[0], left.fortran)
    if bool(left.fortran) != bool(out.fortran):
        msg = (
            "out matrix is {}-ordered".format('Fortran' if out.fortran else 'C')
            + " but input is {}-ordered".format('Fortran' if left.fortran else 'C')
        )
        warnings.warn(msg, OrderEfficiencyWarning)
        # Rather than making loads of copies of the same code, we just moan at
        # the user and then transpose one of the arrays.  We prefer to have
        # `right` in Fortran-order for cache efficiency.
        if left.fortran:
            tmp = out
            out = out.reorder()
        else:
            left = left.reorder()
    cdef idxint row, col, ptr, idx_l, idx_out, out_row, idx_c
    cdef idxint stride_in_col, stride_in_row, stride_out_row, stride_out_col
    cdef idxint nrows=left.shape[0], ncols=right.shape[1]
    cdef double complex val
    stride_in_col = left.shape[0] if left.fortran else 1
    stride_in_row = 1 if left.fortran else left.shape[1]
    stride_out_col = out.shape[0] if out.fortran else 1
    stride_out_row = 1 if out.fortran else out.shape[1]

    # A @ B.dag = (B* @ A.T).T
    if left.fortran:
        for row in range(right.shape[0]):
            for ptr in range(right.row_index[row], right.row_index[row + 1]):
                val = scale * conj(right.data[ptr])
                col = right.col_index[ptr]
                for out_row in range(out.shape[0]):
                    idx_out = row * stride_out_col + out_row * stride_out_row
                    idx_l = col * stride_in_col + out_row * stride_in_row
                    out.data[idx_out] += val * left.data[idx_l]

    else:
      idx_c = idx_out = 0
      for _ in range(nrows):
          _matmul_dag_csr_vector(
              right.data, right.col_index, right.row_index,
              left.data + idx_c,
              scale, out.data + idx_out,
              right.shape[0]
          )
          idx_out += right.shape[0]
          idx_c += left.shape[1]

    if tmp is None:
        return out
    memcpy(tmp.data, out.data, ncols * nrows * sizeof(double complex))
    return tmp


cpdef Dense matmul_dag_dense(
    Dense left, Dense right,
    double complex scale=1., Dense out=None
):
    # blas support matmul for normal, transpose, adjoint for fortran ordered
    # matrices.
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    if (
        out is not None and (
            out.shape[0] != left.shape[0]
            or out.shape[1] != right.shape[0]
        )
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[0]))
        )
    cdef Dense a, b, out_add=None
    cdef double complex alpha = 1., out_scale = 0.
    cdef int m, n, k = left.shape[1], lda, ldb, ldc
    cdef char left_code, right_code

    if not right.fortran:
        # Need a conjugate, we compute the transpose of the desired results.
        # A.conj @ B^op -> (B^T^op @ A.dag)^T
        if out is not None and out.fortran:
          # out is not the right order, create an empty out and add it back.
            out_add = out
            out = dense.empty(left.shape[0], right.shape[0], False)
        elif out is None:
            out = dense.empty(left.shape[0], right.shape[0], False)
        else:
            out_scale = 1.
        m = right.shape[0]
        n = left.shape[0]
        a, b = right, left
        lda = right.shape[1]
        ldb = left.shape[0] if left.fortran else left.shape[1]
        ldc = right.shape[0]

        left_code = b'C'
        right_code = b'T' if left.fortran else b'N'
    else:
        if out is not None and not out.fortran:
            out_add = out
            out = dense.empty(left.shape[0], right.shape[0], True)
        elif out is None:
            out = dense.empty(left.shape[0], right.shape[0], True)
        else:
            out_scale = 1.

        m = left.shape[0]
        n = right.shape[0]
        a, b = left, right
        lda = left.shape[0] if left.fortran else left.shape[1]
        ldb = right.shape[0]
        ldc = left.shape[0]

        left_code = b'N' if left.fortran else b'T'
        right_code = b'C'

    blas.zgemm(
        &left_code, &right_code, &m, &n, &k,
        &scale, a.data, &lda, b.data, &ldb,
        &out_scale, out.data, &ldc
    )

    if out_add is not None:
        out = iadd_dense(out, out_add)

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


cpdef Data matmul_dag_data(
    Data left, Data right,
    double complex scale=1, Dense out=None
):
    return matmul(left, right.adjoint(), scale, out)


matmul_dag = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('scale', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=1),
    ]),
    name='matmul_dag',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)

matmul_dag.__doc__ =\
    """
    Compute the matrix multiplication of two matrices, with the operation
        scale * (left @ right.dag)
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

matmul_dag.add_specialisations([
    (Dense, CSR, Dense, matmul_dag_dense_csr_dense),
    (Dense, Dense, Dense, matmul_dag_dense),
    (Data, Data, Data, matmul_dag_data),
], _defer=True)


cpdef CSR matmul_outer_csr_dense_sparse(Data left, Data right, double complex scale=1):
    return matmul(left, right, dtype=CSR)


cpdef Dia matmul_outer_dia_dense_sparse(Data left, Data right, double complex scale=1):
    return matmul(left, right, dtype=Dia)


cpdef Data matmul_outer_dense_Data(Dense left, Dense right, double complex scale=1):
    out_density = (
        dense.nnz(left) * 1.0 / left.shape[0]
        * dense.nnz(right) * 1.0 / right.shape[1]
    )
    if out_density < 0.3:
        return matmul(left, right, dtype=CSR)
    else:
        return matmul(left, right, dtype=Dense)



cpdef Data matmul_outer_Data(Data left, Data right, double complex scale=1):
    return matmul(left, right)


matmul_outer = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('scale', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=1),
    ]),
    name='matmul_outer',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)

matmul_outer.__doc__ =\
    """
    Alternative to matmul. Does the same operation, but with different output
    type specializations.

    It is expected to be used when `left` is a column matrix and `right` a
    row matrix, creating an output larger than the inputs.

    Parameters
    ----------
    left : Data
        The left operand as either a bra or a ket matrix.

    right : Data
        The right operand as a ket matrix.

    scale : complex, optional
        The scalar to multiply the output by.
    """

matmul_outer.add_specialisations([
    (CSR, Dense, CSR, matmul_outer_csr_dense_sparse),
    (Dense, CSR, CSR, matmul_outer_csr_dense_sparse),
    (Dia, Dense, Dia, matmul_outer_dia_dense_sparse),
    (Dense, Dia, Dia, matmul_outer_dia_dense_sparse),
    (Dense, Dense, Data, matmul_outer_dense_Data),
    (Data, Data, Data, matmul_outer_Data),
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
