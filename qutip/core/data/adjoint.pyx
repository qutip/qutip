#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memset

cimport cython

from qutip.core.data.base cimport idxint
from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data cimport csr, dense

# Import std::conj as `_conj` to avoid clashing with our 'conj' dispatcher.
cdef extern from "<complex>" namespace "std" nogil:
    double complex _conj "conj"(double complex x)

__all__ = [
    'adjoint', 'adjoint_csr', 'adjoint_dense',
    'conj', 'conj_csr', 'conj_dense',
    'transpose', 'transpose_csr', 'transpose_dense',
]


cpdef CSR transpose_csr(CSR matrix):
    """Transpose the CSR matrix, and return a new object."""
    cdef CSR out = csr.empty(matrix.shape[1], matrix.shape[0], csr.nnz(matrix))
    cdef idxint row, col, ptr, ptr_out
    cdef idxint rows_in=matrix.shape[0], rows_out=matrix.shape[1]
    with nogil:
        memset(&out.row_index[0], 0, (rows_out + 1) * sizeof(idxint))
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr] + 1
                out.row_index[col] += 1
        for row in range(rows_out):
            out.row_index[row + 1] += out.row_index[row]
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr]
                ptr_out = out.row_index[col]
                out.data[ptr_out] = matrix.data[ptr]
                out.col_index[ptr_out] = row
                out.row_index[col] = ptr_out + 1
        for row in range(rows_out, 0, -1):
            out.row_index[row] = out.row_index[row - 1]
        out.row_index[0] = 0
    return out


cpdef CSR adjoint_csr(CSR matrix):
    """Conjugate-transpose the CSR matrix, and return a new object."""
    cdef idxint nnz_ = csr.nnz(matrix)
    cdef CSR out = csr.empty(matrix.shape[1], matrix.shape[0], nnz_)
    cdef idxint row, col, ptr, ptr_out
    cdef idxint rows_in=matrix.shape[0], rows_out=matrix.shape[1]
    with nogil:
        memset(&out.row_index[0], 0, (rows_out + 1) * sizeof(idxint))
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr] + 1
                out.row_index[col] += 1
        for row in range(rows_out):
            out.row_index[row + 1] += out.row_index[row]
        for row in range(rows_in):
            for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
                col = matrix.col_index[ptr]
                ptr_out = out.row_index[col]
                out.data[ptr_out] = _conj(matrix.data[ptr])
                out.col_index[ptr_out] = row
                out.row_index[col] = ptr_out + 1
        for row in range(rows_out, 0, -1):
            out.row_index[row] = out.row_index[row - 1]
        out.row_index[0] = 0
    return out


cpdef CSR conj_csr(CSR matrix):
    """Conjugate the CSR matrix, and return a new object."""
    cdef CSR out = csr.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            out.data[ptr] = _conj(matrix.data[ptr])
    return out


cpdef Dense adjoint_dense(Dense matrix):
    cdef Dense out = dense.empty_like(matrix, fortran=not matrix.fortran)
    out.shape = (out.shape[1], out.shape[0])
    with nogil:
        for ptr in range(matrix.shape[0] * matrix.shape[1]):
            out.data[ptr] = _conj(matrix.data[ptr])
    return out


cpdef Dense transpose_dense(Dense matrix):
    cdef Dense out = matrix.copy()
    out.shape = (out.shape[1], out.shape[0])
    out.fortran = not out.fortran
    return out


cpdef Dense conj_dense(Dense matrix):
    cdef Dense out = dense.empty_like(matrix)
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0] * matrix.shape[1]):
            out.data[ptr] = _conj(matrix.data[ptr])
    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

adjoint = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='adjoint',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
adjoint.__doc__ = """Hermitian adjoint (matrix conjugate transpose)."""
adjoint.add_specialisations([
    (Dense, Dense, adjoint_dense),
    (CSR, CSR, adjoint_csr),
], _defer=True)

transpose = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='transpose',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
transpose.__doc__ = """Transpose of a matrix."""
transpose.add_specialisations([
    (Dense, Dense, transpose_dense),
    (CSR, CSR, transpose_csr),
], _defer=True)

conj = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='conj',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
conj.__doc__ = """Element-wise conjugation of a matrix."""
conj.add_specialisations([
    (Dense, Dense, conj_dense),
    (CSR, CSR, conj_csr),
], _defer=True)

del _inspect, _Dispatcher
