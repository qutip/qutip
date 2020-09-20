#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp

from scipy.linalg cimport cython_blas as blas

from qutip.core.data.base cimport idxint, Data
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport (
    CSR, Accumulator, acc_alloc, acc_free, acc_scatter, acc_gather, acc_reset,
)
from qutip.core.data cimport csr, dense

cnp.import_array()

cdef extern from *:
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_FREE(void *ptr)

__all__ = [
    'add', 'add_csr', 'add_dense',
    'sub', 'sub_csr', 'sub_dense',
]

cdef int _ONE=1

cdef void _check_shape(Data left, Data right) nogil except *:
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )

cdef idxint _add_csr(Accumulator *acc, CSR a, CSR b, CSR c) nogil:
    """
    Perform the operation
        c := a + b
    for CSR matrices, where `a` and `b` are guaranteed to be the correct shape,
    and `c` already has enough space allocated (at least nnz(a)+nnz(b)).

    Return the true value of nnz(c).
    """
    cdef idxint row, ptr, nnz=0
    c.row_index[0] = nnz
    for row in range(a.shape[0]):
        for ptr in range(a.row_index[row], a.row_index[row + 1]):
            acc_scatter(acc, a.data[ptr], a.col_index[ptr])
        for ptr in range(b.row_index[row], b.row_index[row + 1]):
            acc_scatter(acc, b.data[ptr], b.col_index[ptr])
        nnz += acc_gather(acc, c.data + nnz, c.col_index + nnz)
        acc_reset(acc)
        c.row_index[row + 1] = nnz
    return nnz

cdef idxint _add_csr_scale(Accumulator *acc, CSR a, CSR b, CSR c, double complex scale) nogil:
    """
    Perform the operation
        c := a + scale*b
    for CSR matrices, where `a` and `b` are guaranteed to be the correct shape,
    and `c` already has enough space allocated (at least nnz(a)+nnz(b)).

    Return the true value of nnz(c).
    """
    cdef idxint row, ptr, nnz=0
    c.row_index[0] = nnz
    for row in range(a.shape[0]):
        for ptr in range(a.row_index[row], a.row_index[row + 1]):
            acc_scatter(acc, a.data[ptr], a.col_index[ptr])
        for ptr in range(b.row_index[row], b.row_index[row + 1]):
            acc_scatter(acc, scale * b.data[ptr], b.col_index[ptr])
        nnz += acc_gather(acc, c.data + nnz, c.col_index + nnz)
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
        _add_csr(&acc, left, right, out)
    else:
        _add_csr_scale(&acc, left, right, out, scale)
    acc_free(&acc)
    return out

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

cpdef CSR sub_csr(CSR left, CSR right):
    return add_csr(left, right, -1)

cpdef Dense sub_dense(Dense left, Dense right):
    return add_dense(left, right, -1)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

add = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
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
], _defer=True)

sub = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
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
], _defer=True)

del _inspect, _Dispatcher
