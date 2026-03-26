# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False
from cython cimport cdivision
from qutip import settings
from qutip.core.data cimport base, CSR, Dia, Dense

# This module is meant to be accessed by dot-access (e.g. mean.mean_csr).
__all__ = []

cdef extern from "<complex>" namespace "std":
    double abs(double complex z)

cdef inline bint is_small(double complex z, double atol):
    """
    Check if a complex number is small within a given absolute tolerance.

    Parameters
    ----------
    z : double complex
        The complex number to check.
    atol : double
        Absolute tolerance.

    Returns
    -------
    bint
        True if both real and imaginary parts are within ±atol of zero.
    """
    return abs(z.real) <= atol and abs(z.imag) <= atol

@cdivision(True)
cdef double complex _mean_generic(
    double complex* data,
    size_t start,
    size_t end,
    double atol
) noexcept:
    """
    Compute the mean of non-small elements in a contiguous array segment.

    Parameters
    ----------
    data : double complex*
        Pointer to the array data.
    start : size_t
        Starting index (inclusive).
    end : size_t
        Ending index (exclusive).
    atol : double
        Absolute tolerance for considering elements as zero.

    Returns
    -------
    double complex
        Mean of elements not considered small (within atol).
        Returns 0.0 if all elements are small.
    """
    cdef size_t i, count = 0
    cdef double complex total = 0

    for i in range(start, end):
        if not is_small(data[i], atol):
            total += data[i]
            count += 1
    return total / count if count > 0 else 0.0

@cdivision(True)
cdef double _mean_abs_generic(
    double complex* data,
    size_t start,
    size_t end,
    double atol
) noexcept:
    """
    Compute the mean of absolute values of non-small elements in a contiguous array segment.

    Parameters
    ----------
    data : double complex*
        Pointer to the array data.
    start : size_t
        Starting index (inclusive).
    end : size_t
        Ending index (exclusive).
    atol : double
        Absolute tolerance for considering elements as zero.

    Returns
    -------
    double
        Mean of absolute values of elements not considered small (within atol).
        Returns 0.0 if all elements are small.
    """
    cdef size_t i, count = 0
    cdef double total = 0

    for i in range(start, end):
        if not is_small(data[i], atol):
            total += abs(data[i])
            count += 1
    return total / count if count > 0 else 0.0


cpdef double complex mean_csr(CSR matrix, double atol=-1) noexcept:
    """
    Compute the mean of non-small elements in a CSR matrix.

    Parameters
    ----------
    matrix : CSR
        Compressed Sparse Row matrix.
    atol : double, optional
        Absolute tolerance for considering elements as zero.
        If negative, uses the global tolerance from `settings.core['atol']`.

    Returns
    -------
    double complex
        Mean of non-small elements. Returns 0.0 if all elements are small.
    """
    if atol < 0:
        atol = settings.core['atol']

    cdef size_t nnz = 0

    nnz = matrix.row_index[matrix.shape[0]]

    if nnz == 0:
        return 0.0

    return _mean_generic(matrix.data, 0, nnz, atol)

@cdivision(True)
cpdef double complex mean_dia(Dia matrix, double atol=-1) noexcept:
    """
    Compute the mean of non-small elements in a DIA matrix.

    Parameters
    ----------
    matrix : Dia
        Diagonal sparse matrix.
    atol : double, optional
        Absolute tolerance for considering elements as zero.
        If negative, uses the global tolerance from `settings.core['atol']`.

    Returns
    -------
    double complex
        Mean of non-small elements. Returns 0.0 if all elements are small.
    """
    if atol < 0:
        atol = settings.core['atol']
    cdef int offset, diag, start, end, col = 1
    cdef double complex cur_el
    cdef size_t nnz = 0
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
    return mean / nnz

cpdef double complex mean_dense(Dense matrix, double atol=-1) noexcept:
    """
    Compute the mean of non-small elements in a dense matrix.

    Parameters
    ----------
    matrix : Dense
        Dense matrix.
    atol : double, optional
        Absolute tolerance for considering elements as zero.
        If negative, uses the global tolerance from `settings.core['atol']`.

    Returns
    -------
    double complex
        Mean of non-small elements. Returns 0.0 if all elements are small.
    """
    if atol < 0:
        atol = settings.core['atol']

    return _mean_generic(
        matrix.data,
        0,
        <size_t> matrix.shape[0] * <size_t> matrix.shape[1],
        atol
    )

cpdef double mean_abs_csr(CSR matrix, double atol=-1) noexcept:
    """
    Compute the mean of absolute values of non-small elements in a CSR matrix.

    Parameters
    ----------
    matrix : CSR
        Compressed Sparse Row matrix.
    atol : double, optional
        Absolute tolerance for considering elements as zero.
        If negative, uses the global tolerance from `settings.core['atol']`.

    Returns
    -------
    double
        Mean of absolute values of non-small elements.
        Returns 0.0 if all elements are small.
    """
    if atol < 0:
        atol = settings.core['atol']

    cdef size_t nnz = 0

    nnz = matrix.row_index[matrix.shape[0]]

    if nnz == 0:
        return 0.0

    return _mean_abs_generic(matrix.data, 0, nnz, atol)

@cdivision(True)
cpdef double mean_abs_dia(Dia matrix, double atol=-1) noexcept:
    """
    Compute the mean of absolute values of non-small elements in a DIA matrix.

    Parameters
    ----------
    matrix : Dia
        Diagonal sparse matrix.
    atol : double, optional
        Absolute tolerance for considering elements as zero.
        If negative, uses the global tolerance from `settings.core['atol']`.

    Returns
    -------
    double
        Mean of absolute values of non-small elements.
        Returns 0.0 if all elements are small.
    """
    if atol < 0:
        atol = settings.core['atol']
    cdef int offset, diag, start, end, col = 1
    cdef double complex cur_el
    cdef double mean_abs = 0
    cdef size_t nnz = 0

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
    return mean_abs / nnz

cpdef double mean_abs_dense(Dense matrix, double atol=-1) noexcept:
    """
    Compute the mean of absolute values of non-small elements in a dense matrix.

    Parameters
    ----------
    matrix : Dense
        Dense matrix.
    atol : double, optional
        Absolute tolerance for considering elements as zero.
        If negative, uses the global tolerance from `settings.core['atol']`.

    Returns
    -------
    double
        Mean of absolute values of non-small elements.
        Returns 0.0 if all elements are small.
    """
    if atol < 0:
        atol = settings.core['atol']

    return _mean_abs_generic(
        matrix.data,
        0,
        <size_t> matrix.shape[0] * <size_t> matrix.shape[1],
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
    Compute the mean value of non-zero entries of a matrix.

    This is a dispatcher that calls the appropriate implementation based on
    the matrix type (Dense, Dia, or CSR).

    Parameters
    ----------
    matrix : Dense, Dia, or CSR
        The input matrix.

    Returns
    -------
    double complex
        Mean of non-small elements (using global tolerance).
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
    Compute the mean value of absolute values of non-zero entries of a matrix.

    This is a dispatcher that calls the appropriate implementation based on
    the matrix type (Dense, Dia, or CSR).

    Parameters
    ----------
    matrix : Dense, Dia, or CSR
        The input matrix.

    Returns
    -------
    double
        Mean of absolute values of non-small elements (using global tolerance).
"""
mean_abs_nonzero.add_specialisations([
    (Dense, mean_abs_dense),
    (Dia, mean_abs_dia),
    (CSR, mean_abs_csr),
], _defer=True)

del _inspect, _Dispatcher
