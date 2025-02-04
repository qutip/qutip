#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
from libc.string cimport memset

from qutip.core.data.base cimport idxint, Data
from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from .adjoint import transpose
from qutip.core.data.dia cimport Dia
from qutip.core.data cimport csr, dia
from qutip.core.data.convert import to as _to
import numpy

__all__ = [
    'kron', 'kron_csr', 'kron_dense', 'kron_dia',
    'kron_csr_dense_csr', 'kron_dense_csr_csr',
    'kron_dia_dense_dia', 'kron_dense_dia_dia',
    'kron_transpose', 'kron_transpose_dense', 'kron_transpose_data',
]


@cython.overflowcheck(True)
cdef idxint _mul_checked(idxint a, idxint b):
    return a * b


cpdef Dense kron_dense(Dense left, Dense right):
    return Dense(numpy.kron(left.as_ndarray(), right.as_ndarray()), copy=False)


cpdef CSR kron_csr(CSR left, CSR right):
    """Kronecker product of two CSR matrices."""
    cdef idxint nrows_l=left.shape[0], nrows_r=right.shape[0]
    cdef idxint ncols_l=left.shape[1], ncols_r=right.shape[1]
    cdef idxint row_l, row_r, row_out
    cdef idxint ptr_start_l, ptr_end_l, ptr_start_r, ptr_end_r, dist_l, dist_r
    cdef idxint ptr_l, ptr_r, ptr_out, ptr_start_out, ptr_end_out
    cdef CSR out = csr.empty(_mul_checked(nrows_l, nrows_r),
                             _mul_checked(ncols_l, ncols_r),
                             _mul_checked(csr.nnz(left), csr.nnz(right)))
    with nogil:
        row_out = 0
        out.row_index[row_out] = 0
        for row_l in range(nrows_l):
            ptr_start_l = left.row_index[row_l]
            ptr_end_l = left.row_index[row_l + 1]
            dist_l = ptr_end_l - ptr_start_l

            for row_r in range(nrows_r):
                ptr_start_r = right.row_index[row_r]
                ptr_end_r = right.row_index[row_r + 1]
                dist_r = ptr_end_r - ptr_start_r

                ptr_start_out = out.row_index[row_out]
                ptr_end_out = ptr_start_out + dist_r

                out.row_index[row_out+1] = out.row_index[row_out] + dist_l*dist_r
                row_out += 1

                for ptr_l in range(ptr_start_l, ptr_end_l):
                    ptr_r = ptr_start_r
                    for ptr_out in range(ptr_start_out, ptr_end_out):
                        out.col_index[ptr_out] =\
                            left.col_index[ptr_l]*ncols_r + right.col_index[ptr_r]
                        out.data[ptr_out] = left.data[ptr_l] * right.data[ptr_r]
                        ptr_r += 1

                    ptr_start_out += dist_r
                    ptr_end_out += dist_r
    return out


cpdef CSR kron_csr_dense_csr(CSR left, Dense right):
    # Since the dispatcher does not have precise control on which function to
    # use when the signature is missing, we specify the output to be the sparse
    # in this case.
    return kron_csr(left, _to(CSR, right))


cpdef CSR kron_dense_csr_csr(Dense left, CSR right):
    # Since the dispatcher does not have precise control on which function to
    # use when the signature is missing, we specify the output to be the sparse
    # in this case.
    return kron_csr(_to(CSR, left), right)


cdef inline void _vec_kron(
    double complex * ptr_l, double complex * ptr_r, double complex * ptr_out,
    idxint size_l, idxint size_r, idxint step
):
    cdef idxint i, j
    for i in range(size_l):
        for j in range(size_r):
            ptr_out[i*step+j] = ptr_l[i] * ptr_r[j]


cpdef Dia kron_dia(Dia left, Dia right):
    cdef idxint nrows_l=left.shape[0], nrows_r=right.shape[0]
    cdef idxint ncols_l=left.shape[1], ncols_r=right.shape[1]
    cdef idxint nrows=_mul_checked(nrows_l, nrows_r)
    cdef idxint ncols=_mul_checked(ncols_l, ncols_r)
    cdef idxint max_diag=_mul_checked(right.num_diag, left.num_diag)
    cdef idxint num_diag=0, diag_left, diag_right, delta, col_left, col_right
    cdef idxint start_left, end_left, start_right, end_right
    cdef Dia out

    if right.shape[0] == right.shape[1]:
        out = dia.empty(nrows, ncols, max_diag)
        memset(
            out.data, 0,
            max_diag * out.shape[1] * sizeof(double complex)
        )
        for diag_left in range(left.num_diag):
            for diag_right in range(right.num_diag):
                out.offsets[num_diag] = (
                    left.offsets[diag_left] * right.shape[0]
                    + right.offsets[diag_right]
                )
                start_left = max(0, left.offsets[diag_left])
                end_left = min(left.shape[1], left.offsets[diag_left] + left.shape[0])
                _vec_kron(
                    left.data + (diag_left * ncols_l) + max(0, left.offsets[diag_left]),
                    right.data + (diag_right * ncols_r) + max(0, right.offsets[diag_right]),
                    out.data + (num_diag * ncols) + max(0, left.offsets[diag_left]) * right.shape[0] + max(0, right.offsets[diag_right]),
                    end_left - start_left,
                    right.shape[1] - abs(right.offsets[diag_right]),
                    right.shape[1]
                )
                num_diag += 1
        out.num_diag = num_diag

    else:
        max_diag = _mul_checked(max_diag, ncols_l)
        if max_diag < nrows:
            out = dia.empty(nrows, ncols, max_diag)
            delta = right.shape[0] - right.shape[1]
            for diag_left in range(left.num_diag):
                for diag_right in range(right.num_diag):
                    start_left = max(0, left.offsets[diag_left])
                    end_left = min(left.shape[1], left.shape[0] + left.offsets[diag_left])
                    for col_left in range(start_left, end_left):
                        memset(
                            out.data + (num_diag * out.shape[1]), 0,
                            out.shape[1] * sizeof(double complex)
                        )

                        out.offsets[num_diag] = (
                            left.offsets[diag_left] * right.shape[0]
                            + right.offsets[diag_right]
                            - col_left * delta
                        )

                        start_right = max(0, right.offsets[diag_right])
                        end_right = min(right.shape[1], right.shape[0] + right.offsets[diag_right])
                        for col_right in range(start_right, end_right):
                            out.data[num_diag * out.shape[1] + col_left * right.shape[1] + col_right] = (
                                right.data[diag_right * right.shape[1] + col_right]
                                * left.data[diag_left * left.shape[1] + col_left]
                            )
                        num_diag += 1
            out.num_diag = num_diag

        else:
            # The output is not sparse enough ant the empty data array would be
            # larger than the dense array.
            # Fallback on dense operation
            left_dense = _to(Dense, left)
            right_dense = _to(Dense, right)
            out_dense = kron_dense(left_dense, right_dense)
            out = _to(Dia, out_dense)

    out = dia.clean_dia(out, True)
    return out


cpdef Dia kron_dia_dense_dia(Dia left, Dense right):
    # Since the dispatcher does not have precise control on which function to
    # use when the signature is missing, we specify the output to be the sparse
    # in this case.
    return kron_dia(left, _to(Dia, right))


cpdef Dia kron_dense_dia_dia(Dense left, Dia right):
    # Since the dispatcher does not have precise control on which function to
    # use when the signature is missing, we specify the output to be the sparse
    # in this case.
    return kron_dia(_to(Dia, left), right)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

kron = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='kron',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
kron.__doc__ =\
    """
    Compute the Kronecker product of two matrices.  This is used to represent
    quantum tensor products of vector spaces.
    """
kron.add_specialisations([
    (CSR, CSR, CSR, kron_csr),
    (Dense, Dense, Dense, kron_dense),
    (Dia, Dia, Dia, kron_dia),
    (CSR, Dense, CSR, kron_csr_dense_csr),
    (Dense, CSR, CSR, kron_dense_csr_csr),
    (Dia, Dense, Dia, kron_dia_dense_dia),
    (Dense, Dia, Dia, kron_dense_dia_dia),
], _defer=True)


cpdef Data kron_transpose_data(Data left, Data right):
    return kron(transpose(left), right)


cpdef Dense kron_transpose_dense(Dense left, Dense right):
    return Dense(numpy.kron(left.as_ndarray().T, right.as_ndarray()), copy=False)


kron_transpose = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='kron_transpose',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
kron_transpose.__doc__ =\
    """
    Compute the Kronecker product of two matrices with transposing the first
    one.  This is used to represent superoperator.
    """
kron_transpose.add_specialisations([
    (Data, Data, Data, kron_transpose_data),
    (Dense, Dense, Dense, kron_transpose_dense),
], _defer=True)


del _inspect, _Dispatcher
