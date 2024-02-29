#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg cimport cython_blas as blas
from qutip.settings import settings

from qutip.core.data.base cimport idxint, Data
from qutip.core.data.dense cimport Dense
from qutip.core.data.dia cimport Dia
from qutip.core.data.tidyup cimport tidyup_dia
from qutip.core.data.csr cimport (
    CSR, Accumulator, acc_alloc, acc_free, acc_scatter, acc_gather, acc_reset,
)
from qutip.core.data cimport csr, dense, dia

cnp.import_array()

__all__ = [
    'add', 'add_csr', 'add_dense', 'iadd_dense', 'add_dia',
    'sub', 'sub_csr', 'sub_dense', 'sub_dia',
]


cdef int _ONE=1


cdef int _check_shape(Data left, Data right) except -1 nogil:
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    return 0


cdef idxint _add_csr(Accumulator *acc, CSR a, CSR b, CSR c, double tol) nogil:
    """
    Perform the operation
        c := a + b
    for CSR matrices, where `a` and `b` are guaranteed to be the correct shape,
    and `c` already has enough space allocated (at least nnz(a)+nnz(b)).

    Return the true value of nnz(c).
    """
    cdef idxint row, ptr_a, ptr_b, ptr_a_max, ptr_b_max, nnz=0, col_a, col_b
    cdef idxint ncols = a.shape[1]
    c.row_index[0] = nnz
    ptr_a_max = ptr_b_max = 0
    for row in range(a.shape[0]):
        ptr_a = ptr_a_max
        ptr_a_max = a.row_index[row + 1]
        ptr_b = ptr_b_max
        ptr_b_max = b.row_index[row + 1]
        col_a = a.col_index[ptr_a] if ptr_a < ptr_a_max else ncols + 1
        col_b = b.col_index[ptr_b] if ptr_b < ptr_b_max else ncols + 1
        # We use this method of going through the row to give the Accumulator
        # the best chance of receiving the scatters in a sorted order.  We
        # could also safely iterate through a completely then b, which would be
        # more cache efficient, but would quite often require a sort within the
        # gather, making the algorithimic complexity worse.
        while ptr_a < ptr_a_max or ptr_b < ptr_b_max:
            if col_a < col_b:
                acc_scatter(acc, a.data[ptr_a], col_a)
                ptr_a += 1
                col_a = a.col_index[ptr_a] if ptr_a < ptr_a_max else ncols + 1
            else:
                acc_scatter(acc, b.data[ptr_b], col_b)
                ptr_b += 1
                col_b = b.col_index[ptr_b] if ptr_b < ptr_b_max else ncols + 1
            # There's no need to test col_a == col_b because the Accumulator
            # already tests that in all scatters anyway.
        nnz += acc_gather(acc, c.data + nnz, c.col_index + nnz, tol)
        acc_reset(acc)
        c.row_index[row + 1] = nnz
    return nnz


cdef idxint _add_csr_scale(Accumulator *acc, CSR a, CSR b, CSR c,
                           double complex scale, double tol) nogil:
    """
    Perform the operation
        c := a + scale*b
    for CSR matrices, where `a` and `b` are guaranteed to be the correct shape,
    and `c` already has enough space allocated (at least nnz(a)+nnz(b)).

    Return the true value of nnz(c).
    """
    cdef idxint row, ptr_a, ptr_b, ptr_a_max, ptr_b_max, nnz=0, col_a, col_b
    cdef idxint ncols = a.shape[1]
    c.row_index[0] = nnz
    ptr_a_max = ptr_b_max = 0
    for row in range(a.shape[0]):
        ptr_a = ptr_a_max
        ptr_a_max = a.row_index[row + 1]
        ptr_b = ptr_b_max
        ptr_b_max = b.row_index[row + 1]
        col_a = a.col_index[ptr_a] if ptr_a < ptr_a_max else ncols + 1
        col_b = b.col_index[ptr_b] if ptr_b < ptr_b_max else ncols + 1
        while ptr_a < ptr_a_max or ptr_b < ptr_b_max:
            if col_a < col_b:
                acc_scatter(acc, a.data[ptr_a], col_a)
                ptr_a += 1
                col_a = a.col_index[ptr_a] if ptr_a < ptr_a_max else ncols + 1
            else:
                acc_scatter(acc, scale * b.data[ptr_b], col_b)
                ptr_b += 1
                col_b = b.col_index[ptr_b] if ptr_b < ptr_b_max else ncols + 1
        nnz += acc_gather(acc, c.data + nnz, c.col_index + nnz, tol)
        acc_reset(acc)
        c.row_index[row + 1] = nnz
    return nnz


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
    cdef Accumulator acc
    cdef double tol = 0
    if settings.core['auto_tidyup']:
        tol = settings.core['auto_tidyup_atol']
    # Fast paths for zero matrices.
    if right_nnz == 0 or scale == 0:
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
    acc = acc_alloc(left.shape[1])
    if scale == 1:
        _add_csr(&acc, left, right, out, tol)
    else:
        _add_csr_scale(&acc, left, right, out, scale, tol)
    acc_free(&acc)
    return out


cdef void add_dense_eq_order_inplace(Dense left, Dense right, double complex scale):
    cdef int size = left.shape[0] * left.shape[1]
    with nogil:
        blas.zaxpy(&size, &scale, right.data, &_ONE, left.data, &_ONE)


cdef Dense _add_dense_eq_order(Dense left, Dense right, double complex scale):
    cdef Dense out = left.copy()
    cdef int size = left.shape[0] * left.shape[1]
    with nogil:
        blas.zaxpy(&size, &scale, right.data, &_ONE, out.data, &_ONE)
    return out


cpdef Dense add_dense(Dense left, Dense right, double complex scale=1):
    _check_shape(left, right)
    if not (left.fortran ^ right.fortran):
        return _add_dense_eq_order(left, right, scale)
    cdef Dense out = left.copy()
    cdef size_t nrows=left.shape[0], ncols=left.shape[1], idx
    # We always iterate through `left` and `out` in memory-layout order.
    cdef int dim1, dim2
    dim1, dim2 = (nrows, ncols) if left.fortran else (ncols, nrows)
    with nogil:
        for idx in range(dim2):
            blas.zaxpy(&dim1, &scale, right.data + idx, &dim2, out.data + idx*dim1, &_ONE)
    return out


cpdef Dense iadd_dense(Dense left, Dense right, double complex scale=1):
    _check_shape(left, right)
    cdef int size = left.shape[0] * left.shape[1]
    cdef int dim1, dim2
    cdef size_t nrows=left.shape[0], ncols=left.shape[1], idx
    dim1, dim2 = (nrows, ncols) if left.fortran else (ncols, nrows)
    with nogil:
        if not (left.fortran ^ right.fortran):
            blas.zaxpy(&size, &scale, right.data, &_ONE, left.data, &_ONE)
        else:
            for idx in range(dim2):
                blas.zaxpy(&dim1, &scale, right.data + idx, &dim2,
                           left.data + idx*dim1, &_ONE)
    return left


cpdef Dia add_dia(Dia left, Dia right, double complex scale=1):
    _check_shape(left, right)
    cdef idxint diag_left=0, diag_right=0, out_diag=0, i
    cdef double complex *ptr_out,
    cdef double complex *ptr_left
    cdef double complex *ptr_right
    cdef bint sorted=True
    cdef Dia out = dia.empty(left.shape[0], left.shape[1], left.num_diag + right.num_diag)
    cdef int length, size=left.shape[1]

    ptr_out = out.data
    ptr_left = left.data
    ptr_right = right.data

    with nogil:
        while diag_left < left.num_diag and diag_right < right.num_diag:
            if left.offsets[diag_left] == right.offsets[diag_right]:
                out.offsets[out_diag] = left.offsets[diag_left]
                blas.zcopy(&size, ptr_left, &_ONE, ptr_out, &_ONE)
                blas.zaxpy(&size, &scale, ptr_right, &_ONE, ptr_out, &_ONE)
                ptr_left += size
                diag_left += 1
                ptr_right += size
                diag_right += 1
            elif left.offsets[diag_left] <= right.offsets[diag_right]:
                out.offsets[out_diag] = left.offsets[diag_left]
                blas.zcopy(&size, ptr_left, &_ONE, ptr_out, &_ONE)
                ptr_left += size
                diag_left += 1
            else:
                out.offsets[out_diag] = right.offsets[diag_right]
                blas.zcopy(&size, ptr_right, &_ONE, ptr_out, &_ONE)
                if scale != 1:
                    blas.zscal(&size, &scale, ptr_out, &_ONE)
                ptr_right += size
                diag_right += 1
            if out_diag != 0 and out.offsets[out_diag-1] >= out.offsets[out_diag]:
                sorted=False
            ptr_out += size
            out_diag += 1

        if diag_left < left.num_diag:
            for i in range(left.num_diag - diag_left):
                out.offsets[out_diag] = left.offsets[diag_left + i]
                if out_diag != 0 and out.offsets[out_diag-1] >= out.offsets[out_diag]:
                    sorted=False
                out_diag += 1

            length = size * (left.num_diag - diag_left)
            blas.zcopy(&length, ptr_left, &_ONE, ptr_out, &_ONE)


        if diag_right < right.num_diag:
            for i in range(right.num_diag - diag_right):
                out.offsets[out_diag] = right.offsets[diag_right + i]
                if out_diag != 0 and out.offsets[out_diag-1] >= out.offsets[out_diag]:
                    sorted=False
                out_diag += 1

            length = size * (right.num_diag - diag_right)
            blas.zcopy(&length, ptr_right, &_ONE, ptr_out, &_ONE)
            if scale != 1:
                blas.zscal(&length, &scale, ptr_out, &_ONE)

        out.num_diag = out_diag

    if not sorted:
        dia.clean_dia(out, True)
    if settings.core['auto_tidyup']:
        tidyup_dia(out, settings.core['auto_tidyup_atol'], True)
    return out


cpdef CSR sub_csr(CSR left, CSR right):
    return add_csr(left, right, -1)


cpdef Dense sub_dense(Dense left, Dense right):
    return add_dense(left, right, -1)


cpdef Dia sub_dia(Dia left, Dia right):
    return add_dia(left, right, -1)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

add = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('scale', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=1),
    ]),
    name='add',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
add.__doc__ =\
    """
    Perform the operation
        left + scale*right
    where `left` and `right` are matrices, and `scale` is an optional complex
    scalar.
    """
add.add_specialisations([
    (Dense, Dense, Dense, add_dense),
    (CSR, CSR, CSR, add_csr),
    (Dia, Dia, Dia, add_dia),
], _defer=True)

sub = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='sub',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
sub.__doc__ =\
    """
    Perform the operation
        left - right
    where `left` and `right` are matrices.
    """
sub.add_specialisations([
    (Dense, Dense, Dense, sub_dense),
    (CSR, CSR, CSR, sub_csr),
    (Dia, Dia, Dia, sub_dia),
], _defer=True)

del _inspect, _Dispatcher
