#cython: language_level=3
#cython: boundscheck=False, wrapround=False, initializedcheck=False

from qutip.core.data cimport idxint, csr, CSR, dense, Dense, Data, Dia, dia
from scipy.linalg.cython_blas cimport zscal

__all__ = [
    'mul', 'mul_csr', 'mul_dense', 'mul_dia',
    'imul', 'imul_csr', 'imul_dense', 'imul_dia', 'imul_data',
    'neg', 'neg_csr', 'neg_dense', 'neg_dia',
]


cpdef CSR imul_csr(CSR matrix, double complex value):
    """Multiply this CSR `matrix` by a complex scalar `value`."""
    cdef idxint l = csr.nnz(matrix)
    cdef int ONE=1
    zscal(&l, &value, matrix.data, &ONE)
    return matrix

cpdef CSR mul_csr(CSR matrix, double complex value):
    """Multiply this CSR `matrix` by a complex scalar `value`."""
    if value == 0:
        return csr.zeros(matrix.shape[0], matrix.shape[1])
    cdef CSR out = csr.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            out.data[ptr] = value * matrix.data[ptr]
    return out

cpdef CSR neg_csr(CSR matrix):
    """Unary negation of this CSR `matrix`.  Return a new object."""
    cdef CSR out = csr.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            out.data[ptr] = -matrix.data[ptr]
    return out


cpdef Dia imul_dia(Dia matrix, double complex value):
    """Multiply this Dia `matrix` by a complex scalar `value`."""
    cdef idxint l = matrix.num_diag * matrix.shape[1]
    cdef int ONE=1
    zscal(&l, &value, matrix.data, &ONE)
    return matrix

cpdef Dia mul_dia(Dia matrix, double complex value):
    """Multiply this Dia `matrix` by a complex scalar `value`."""
    if value == 0:
        return dia.zeros(matrix.shape[0], matrix.shape[1])
    cdef Dia out = dia.empty_like(matrix)
    cdef idxint ptr, diag, l = matrix.num_diag * matrix.shape[1]
    with nogil:
        for ptr in range(l):
            out.data[ptr] = value * matrix.data[ptr]
        for ptr in range(matrix.num_diag):
            out.offsets[ptr] = matrix.offsets[ptr]
        out.num_diag = matrix.num_diag
    return out

cpdef Dia neg_dia(Dia matrix):
    """Unary negation of this Dia `matrix`.  Return a new object."""
    cdef Dia out = matrix.copy()
    cdef idxint ptr, l = matrix.num_diag * matrix.shape[1]
    with nogil:
        for ptr in range(l):
            out.data[ptr] = -matrix.data[ptr]
    return out


cpdef Dense imul_dense(Dense matrix, double complex value):
    """Multiply this Dense `matrix` by a complex scalar `value`."""
    cdef size_t ptr
    cdef int ONE=1
    cdef idxint l = matrix.shape[0]*matrix.shape[1]
    zscal(&l, &value, matrix.data, &ONE)
    return matrix

cpdef Dense mul_dense(Dense matrix, double complex value):
    """Multiply this Dense `matrix` by a complex scalar `value`."""
    cdef Dense out = dense.empty_like(matrix)
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0]*matrix.shape[1]):
            out.data[ptr] = value * matrix.data[ptr]
    return out

cpdef Dense neg_dense(Dense matrix):
    """Unary negation of this Dense `matrix`.  Return a new object."""
    cdef Dense out = dense.empty_like(matrix)
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0]*matrix.shape[1]):
            out.data[ptr] = -matrix.data[ptr]
    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

mul = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('value', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='mul',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
mul.__doc__ =\
    """Multiply a matrix element-wise by a scalar."""
mul.add_specialisations([
    (CSR, CSR, mul_csr),
    (Dia, Dia, mul_dia),
    (Dense, Dense, mul_dense),
], _defer=True)

imul = _Dispatcher(
    # Will not be inplce if specialisation does not exist but should still
    # give expected results if used as:
    # mat = imul(mat, x)
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('value', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='imul',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
imul.__doc__ =\
    """Multiply inplace a matrix element-wise by a scalar."""
imul.add_specialisations([
    (CSR, CSR, imul_csr),
    (Dia, Dia, imul_dia),
    (Dense, Dense, imul_dense),
], _defer=True)

neg = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='neg',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
neg.__doc__ =\
    """Unary element-wise negation of a matrix."""
neg.add_specialisations([
    (CSR, CSR, neg_csr),
    (Dia, Dia, neg_dia),
    (Dense, Dense, neg_dense),
], _defer=True)

del _inspect, _Dispatcher


cpdef Data imul_data(Data matrix, double complex value):
    if type(matrix) is CSR:
        return imul_csr(matrix, value)
    elif type(matrix) is Dense:
        return imul_dense(matrix, value)
    elif type(matrix) is Dia:
        return imul_dia(matrix, value)
    else:
        return imul(matrix, value)
