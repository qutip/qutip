#cython: language_level=3
#cython: boundscheck=False, wrapround=False, initializedcheck=False

from qutip.core.data cimport idxint, csr, CSR, dense, Dense, csc, CSC

__all__ = [
    'mul', 'mul_csr', 'mul_dense', 'mul_csc',
    'neg', 'neg_csr', 'neg_dense', 'neg_csc',
]


cpdef void mul_csr_inplace(CSR matrix, double complex value):
    """Multiply this CSR `matrix` by a complex scalar `value`."""
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            matrix.data[ptr] *= value

cpdef void mul_csc_inplace(CSC matrix, double complex value):
    """Multiply this CSR `matrix` by a complex scalar `value`."""
    cdef idxint ptr
    with nogil:
        for ptr in range(csc.nnz(matrix)):
            matrix.data[ptr] *= value

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

cpdef CSC mul_csc(CSC matrix, double complex value):
    """Multiply this CSR `matrix` by a complex scalar `value`."""
    if value == 0:
        return csc.zeros(matrix.shape[0], matrix.shape[1])
    cdef CSC out = csc.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csc.nnz(matrix)):
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

cpdef CSC neg_csc(CSC matrix):
    """Unary negation of this CSR `matrix`.  Return a new object."""
    cdef CSC out = csc.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csc.nnz(matrix)):
            out.data[ptr] = -matrix.data[ptr]
    return out


cpdef void mul_dense_inplace(Dense matrix, double complex value):
    """Multiply this Dense `matrix` by a complex scalar `value`."""
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0]*matrix.shape[1]):
            matrix.data[ptr] *= value

cpdef Dense mul_dense(Dense matrix, double complex value):
    """Multiply this Dense `matrix` by a complex scalar `value`."""
    cdef Dense out = dense.empty_like(matrix)
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0]*matrix.shape[1]):
            out.data[ptr] = value * matrix.data[ptr]
    return out

cpdef Dense neg_dense(Dense matrix):
    """Unary negation of this CSR `matrix`.  Return a new object."""
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
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
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
    (CSC, CSC, mul_csc),
    (Dense, Dense, mul_dense),
], _defer=True)

mul_inplace = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('value', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='mul_inplace',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
mul_inplace.__doc__ =\
    """Multiply inplace a matrix element-wise by a scalar."""
mul_inplace.add_specialisations([
    (CSR, mul_csr_inplace),
    (CSC, mul_csc_inplace),
    (Dense, mul_dense_inplace),
], _defer=True)

neg = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
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
    (CSC, CSC, neg_csc),
    (Dense, Dense, neg_dense),
], _defer=True)

del _inspect, _Dispatcher
