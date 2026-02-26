# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False
from qutip import settings
from qutip.core.data cimport base, CSR, Dia, Dense

cdef extern from "<complex>" namespace "std":
    double abs(double complex z)

cdef inline bint is_small(double complex z, double atol):
    return abs(z.real) <= atol and abs(z.imag) <= atol

cdef double complex _mean_generic(
    double complex* data,
    size_t start,
    size_t end,
    double atol
) noexcept:
    cdef base.idxint i, count = 0
    cdef double complex total = 0

    for i in range(start, end):
        if not is_small(data[i], atol):
            total += data[i]
            count += 1
    return total / <double complex>count if count > 0 else 0.0

cdef double _mean_abs_generic(
    double complex* data,
    size_t start,
    size_t end,
    double atol
) noexcept:
    cdef size_t i, count = 0
    cdef double total = 0

    for i in range(start, end):
        if not is_small(data[i], atol):
            total += abs(data[i])
            count += 1
    return total / <double>count if count > 0 else 0.0

# This module is meant to be accessed by dot-access (e.g. mean.mean_csr).
__all__ = []

cpdef double complex mean_csr(CSR matrix, double atol=-1) noexcept:
    # Take the global absolute tolerance in case not provided by user
    if atol < 0:
        atol = settings.core['atol']

    cdef base.idxint nnz = 0


    nnz = matrix.row_index[matrix.shape[0]]

    if nnz == 0:
        return 0.0

    return _mean_generic(matrix.data, 0, nnz, atol)

cpdef double complex mean_dia(Dia matrix, double atol=-1) noexcept:
    # Take the global absolute tolerance in case not provided by user
    if atol < 0:
        atol = settings.core['atol']
    cdef int offset, diag, start, end, col = 1
    cdef double complex cur_el
    cdef base.idxint nnz = 0
    cdef double complex mean = 0

    for diag in range(matrix.num_diag):
        offset = matrix.offsets[diag]
        start = max(0, offset)
        end = min(matrix.shape[1], matrix.shape[0] + offset)
        if end < start:
            continue

        for col in range(start, end):
            cur_el = matrix.data[diag * matrix.shape[1] + col]
            if is_small(cur_el, atol):
                continue
            mean += cur_el
            nnz += 1
    
    if nnz == 0:
        return 0.0
    return mean / <double complex>nnz

cpdef double complex mean_dense(Dense matrix, double atol=-1) noexcept:
    # Take the global absolute tolerance in case not provided by user
    if atol < 0:
        atol = settings.core['atol']

    return _mean_generic(
        matrix.data,
        0,
        matrix.shape[0] * matrix.shape[1],
        atol
    )

cpdef double mean_abs_csr(CSR matrix, double atol=-1) noexcept:
    # Take the global absolute tolerance in case not provided by user
    if atol < 0:
        atol = settings.core['atol']

    cdef base.idxint nnz = 0

    nnz = matrix.row_index[matrix.shape[0]]

    if nnz == 0:
        return 0.0

    return _mean_abs_generic(matrix.data, 0, nnz, atol)

cpdef double mean_abs_dia(Dia matrix, double atol=-1) noexcept:
    # Take the global absolute tolerance in case not provided by user
    if atol < 0:
        atol = settings.core['atol']
    cdef int offset, diag, start, end, col = 1
    cdef double complex cur_el
    cdef double mean_abs = 0
    cdef base.idxint nnz = 0

    for diag in range(matrix.num_diag):
        offset = matrix.offsets[diag]
        start = max(0, offset)
        end = min(matrix.shape[1], matrix.shape[0] + offset)

        if end < start:
            continue

        for col in range(start, end):
            cur_el = matrix.data[diag * matrix.shape[1] + col]
            if is_small(cur_el, atol):
                continue
            mean_abs += abs(cur_el)
            nnz += 1
    if nnz == 0:
        return 0.0
    return mean_abs / <double>nnz

cpdef double mean_abs_dense(Dense matrix, double atol=-1) noexcept:
    # Take the global absolute tolerance in case not provided by user
    if atol < 0:
        atol = settings.core['atol']

    return _mean_abs_generic(
        matrix.data,
        0,
        matrix.shape[0] * matrix.shape[1],
        atol
    )

from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

mean_nonzero = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='mean_nonzero',
    module=__name__,
    inputs=('matrix',),
)
mean_nonzero.__doc__ = """
    Adapted mean value: compute the mean value of non-zero entries of a matrix.
"""
mean_nonzero.add_specialisations([
    (Dense, mean_dense),
    (Dia, mean_dia),
    (CSR, mean_csr),
], _defer=True)

mean_abs_nonzero = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='mean_abs_nonzero',
    module=__name__,
    inputs=('matrix',),
)
mean_abs_nonzero.__doc__ = """
    Adapted mean value: \
    compute the mean value of absolute values of non-zero entries of a matrix.
"""
mean_abs_nonzero.add_specialisations([
    (Dense, mean_abs_dense),
    (Dia, mean_abs_dia),
    (CSR, mean_abs_csr),
], _defer=True)

del _inspect, _Dispatcher
