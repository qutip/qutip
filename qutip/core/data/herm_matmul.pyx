#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memset, memcpy
from libc.math cimport fabs, sqrt
from scipy.linalg cimport cython_blas as blas

cdef extern from "<complex>" namespace "std" nogil:
    double complex _conj "conj"(double complex x)

cimport cython

# cimport numpy as cnp

from qutip.core.data.base cimport idxint, Data
from qutip.core.data cimport csr, dense, CSR, Dense
from qutip.core.data.matmul cimport imatmul_data_dense
from qutip.core.data.add import add
from qutip.core.data.matmul import matmul

# cnp.import_array()


__all__ = [
    "herm_matmul", "herm_matmul_csr_dense_dense",
    "herm_matmul_dense", "herm_matmul_data"
]


# This function is templated over integral types on import to allow `idxint` to
# be any signed integer (though likely things will only work for >=32-bit).  To
# change integral types, you only need to change the `idxint` definitions in
# `core.data.base` at compile-time.
cdef extern from "src/matmul_csr_vector.hpp" nogil:
    void _matmul_csr_vector_herm[T](
        double complex *data, T *col_index, T *row_index,
        double complex *vec, double complex scale, double complex *out,
        T nrows, T sub_size)


cdef void _check_shape(Data left, Data right, idxint subsize, Data out=None) except * nogil:
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    if right.shape[1] != 1:
        raise ValueError(
            "invalid right matrix shape, must be a operator-ket"
        )
    if right.shape[0] != subsize * subsize:
        raise ValueError(
            "Wrong shape the density matrix form of the operator-ket"
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


cpdef Dense herm_matmul_csr_dense_dense(CSR left, Dense right,
                                        idxint subsystem_size=0,
                                        double complex scale=1,
                                        Data out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.
    `left` and `right` must be chossen so `out` is hemitian.
    `left` and 'out' must be vectorized operator: `shape = (N**2, 1)`

    Made to be used in :func:`mesolve`.

    """
    cdef Dense inplace_out=None
    if subsystem_size == 0:
        subsystem_size = <size_t> sqrt(right.shape[0])
    _check_shape(left, right, subsystem_size, out)

    if type(out) is Dense:
        inplace_out = out
        out = None
    else:
        inplace_out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    # cdef idxint row, ptr, idx_r, idx_out, dm_row, dm_col
    # cdef idxint nrows=left.shape[0], ncols=right.shape[1]
    # cdef double complex val
    # right shape (N*N, 1) is interpreted as (N, N) and we loop only on the
    # upper triangular part.
    _matmul_csr_vector_herm(
        left.data, left.col_index, left.row_index,
        right.data, scale, inplace_out.data,
        right.shape[0], subsystem_size
    )
    if out is not None:
        inplace_out = add(out, inplace_out)

    return inplace_out


cpdef Dense herm_matmul_dense(Dense left, Dense right, idxint subsystem_size=0,
                              double complex scale=1, Data out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.
    `left` and `right` must be chossen so `out` is hemitian.
    `left` and 'out' must be vectorized operator: `shape = (N**2, 1)`
    Made to be used in :func:`mesolve`.
    """
    if subsystem_size == 0:
        subsystem_size = <size_t> sqrt(right.shape[0])
    _check_shape(left, right, subsystem_size, out)
    cdef Dense inplace_out=None
    if type(out) is Dense:
        inplace_out = out
        out = None
    else:
        inplace_out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    cdef double complex val
    cdef int dm_row, dm_col, row_stride, col_stride, one=1
    row_stride = 1 if left.fortran else left.shape[1]
    col_stride = left.shape[0] if left.fortran else 1
    # right shape (N*N, 1) is interpreted as (N, N) and we loop only on the
    # upper triangular part.
    for dm_row in range(subsystem_size):
        inplace_out.data[dm_row * (subsystem_size+1)] += scale * blas.zdotu(
            &right.shape[0],
            &left.data[dm_row * (subsystem_size+1) * row_stride], &col_stride,
            right.data, &one)
        for dm_col in range(dm_row+1, subsystem_size):
            val = blas.zdotu(
                &right.shape[0],
                &left.data[(dm_row * subsystem_size + dm_col) * row_stride], &col_stride,
                right.data, &one)
            inplace_out.data[dm_row * subsystem_size + dm_col] += scale * val
            inplace_out.data[dm_col * subsystem_size + dm_row] += _conj(scale * val)
    if out is not None:
        inplace_out = add(out, inplace_out)
    return inplace_out


cpdef Data herm_matmul_data(Data left, Data right, size_t subsystem_size=0,
                             double complex scale=1, Data out=None):
    if out is None:
        return matmul(left, right, scale)
    elif type(right) is Dense and type(out) is Dense:
        imatmul_data_dense(left, right, scale, out)
        return out
    else:
        return add(out, matmul(left, right, scale))


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

herm_matmul = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter(
            'subsystem_size',
            _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0
        ),
        _inspect.Parameter(
            'scale', _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1
        ),
        _inspect.Parameter('out', _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    ]),
    name='herm_matmul',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
herm_matmul.__doc__ =\
    """
    Compute the matrix multiplication of two matrices, with the operation
        scale * (left @ right)
    where `scale` is (optionally) a scalar, and `left` and `right` are
    matrices. The output matrix should be a column stacked Hermitian matrix.
    This is not tested but only half of the matrix product is computed. The
    other half being filled with the conj of the first.

    Intented to be used in ``mesolve``.

    Parameters
    ----------
    left : Data
        The left operand as either a bra or a ket matrix.

    right : Data
        The right operand as a ket matrix.

    subsystem_size : int
        Size of the

    scale : complex, optional
        The scalar to multiply the output by.

    out : Data, optional
        If provided, ``out += scale * (left @ right)`` is computed.
        If ``type(out)`` is ``Dense``, will be done inplace.
    """
herm_matmul.add_specialisations([
    (Data, Data, Data, herm_matmul_data),
    (CSR, Dense, Dense, herm_matmul_csr_dense_dense),
    (Dense, Dense, Dense, herm_matmul_dense),
], _defer=True)


del _inspect, _Dispatcher
