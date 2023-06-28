#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython

from qutip.core.data.base cimport idxint
from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data cimport csr
import numpy

__all__ = [
    'kron', 'kron_csr', 'kron_dense'
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
], _defer=True)

del _inspect, _Dispatcher
