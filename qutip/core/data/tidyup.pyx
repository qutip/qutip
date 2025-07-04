#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.math cimport fabs

cimport numpy as cnp
from scipy.linalg cimport cython_blas as blas

from qutip.core.data cimport csr, dense, CSR, Dense, dia, Dia, base

cdef extern from "<complex>" namespace "std" nogil:
    # abs is templated such that Cython treats std::abs as complex->complex
    double abs(double complex x)

__all__ = [
    'tidyup', 'tidyup_csr', 'tidyup_dense', 'tidyup_dia',
]


cpdef CSR tidyup_csr(CSR matrix, double tol, bint inplace=True):
    cdef bint re, im
    cdef size_t row, ptr, ptr_start, ptr_end=0, nnz
    cdef double complex value
    cdef CSR out = matrix if inplace else matrix.copy()
    nnz = 0
    out.row_index[0] = 0
    for row in range(matrix.shape[0]):
        ptr_start, ptr_end = ptr_end, matrix.row_index[row + 1]
        for ptr in range(ptr_start, ptr_end):
            re = im = False
            value = matrix.data[ptr]
            if fabs(value.real) < tol:
                re = True
                value.real = 0
            if fabs(value.imag) < tol:
                im = True
                value.imag = 0
            if not (re & im):
                out.data[nnz] = value
                out.col_index[nnz] = matrix.col_index[ptr]
                nnz += 1
        out.row_index[row + 1] = nnz
    return out


cpdef Dense tidyup_dense(Dense matrix, double tol, bint inplace=True):
    cdef Dense out = matrix if inplace else matrix.copy()
    cdef double complex value
    cdef size_t ptr
    for ptr in range(matrix.shape[0] * matrix.shape[1]):
        value = matrix.data[ptr]
        if fabs(value.real) < tol:
            matrix.data[ptr].real = 0
        if fabs(value.imag) < tol:
            matrix.data[ptr].imag = 0
    return out


cpdef Dia tidyup_dia(Dia matrix, double tol, bint inplace=True):
    cdef Dia out = matrix if inplace else matrix.copy()
    cdef base.idxint diag=0, new_diag=0, ONE=1, start, end, col
    cdef bint re, im, has_data
    cdef double complex value
    cdef int length

    while diag < out.num_diag:
        start = max(0, out.offsets[diag])
        end = min(out.shape[1], out.shape[0] + out.offsets[diag])
        has_data = False
        for col in range(start, end):
            re = False
            im = False
            if fabs(out.data[diag * out.shape[1] + col].real) < tol:
                re = True
                out.data[diag * out.shape[1] + col].real = 0
            if fabs(out.data[diag * out.shape[1] + col].imag) < tol:
                im = True
                out.data[diag * out.shape[1] + col].imag = 0
            has_data |= not (re & im)

        if has_data and new_diag < diag:
            length = out.shape[1]
            blas.zcopy(
                &length,
                &out.data[diag * out.shape[1]], &ONE,
                &out.data[new_diag * out.shape[1]], &ONE
            )
            out.offsets[new_diag] = out.offsets[diag]

        if has_data:
            new_diag += 1
        diag += 1
    out.num_diag = new_diag
    if out._scipy is not None:
        out._scipy.data = out._scipy.data[:new_diag]
        out._scipy.offsets = out._scipy.offsets[:new_diag]
    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

# In this case, to support the `inplace` argument, we do not support
# dispatching on the output.

tidyup = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('tol', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('inplace', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=True),
    ]),
    name='tidyup',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
tidyup.__doc__ =\
    """
    Tidy up the input matrix by truncating small values to zero.  The real and
    imaginary parts are treated individually, so (for example) the number
        1e-18 + 2j
    would be truncated with a tolerance of `1e-15` to just
        2j

    By default, this operation is in-place.  The output type will always match
    the input type; no dispatching takes place on the output.

    Parameters
    ----------
    matrix : Data
        The matrix to tidy up.

    tol : real
        The absolute tolerance to use to determine whether a real or imaginary
        part should be truncated to zero.

    inplace : bool, optional (True)
        Whether to do the operation in-place.  The output matrix will always be
        returned, even if this argument is `True`; it will just be the same
        Python object as was input.
    """
tidyup.add_specialisations([
    (CSR, tidyup_csr),
    (Dense, tidyup_dense),
    (Dia, tidyup_dia),
], _defer=True)

del _inspect, _Dispatcher
