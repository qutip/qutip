#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
from libc.math cimport sqrt

from qutip.core.data cimport Data, CSR, Dense
from qutip.core.data cimport base

__all__ = [
    'trace', 'trace_csr', 'trace_dense',
    'trace_oper_ket', 'trace_oper_ket_csr', 'trace_oper_ket_dense',
]


cdef void _check_shape(Data matrix) nogil except *:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("".join([
            "matrix shape ", str(matrix.shape), " is not square.",
        ]))


cdef void _check_shape_oper_ket(int N, Data matrix) nogil except *:
    if matrix.shape[0] != N * N or matrix.shape[1] != 1:
        raise ValueError("".join([
            "matrix ", str(matrix.shape), " is not a stacked square matrix."
        ]))


cpdef double complex trace_csr(CSR matrix) nogil except *:
    _check_shape(matrix)
    cdef size_t row, ptr
    cdef double complex trace = 0
    for row in range(matrix.shape[0]):
        for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
            if matrix.col_index[ptr] == row:
                trace += matrix.data[ptr]
                break
    return trace

cpdef double complex trace_dense(Dense matrix) nogil except *:
    _check_shape(matrix)
    cdef double complex trace = 0
    cdef size_t ptr = 0
    cdef size_t stride = matrix.shape[0] + 1
    for _ in range(matrix.shape[0]):
        trace += matrix.data[ptr]
        ptr += stride
    return trace


cpdef double complex trace_oper_ket_csr(CSR matrix) nogil except *:
    cdef size_t N = <size_t>sqrt(matrix.shape[0])
    _check_shape_oper_ket(N, matrix)
    cdef size_t row
    cdef double complex trace = 0
    cdef size_t stride = N + 1
    for row in range(N):
        if matrix.row_index[row * stride] != matrix.row_index[row * stride + 1]:
            trace += matrix.data[matrix.row_index[row * stride]]
    return trace

cpdef double complex trace_oper_ket_dense(Dense matrix) nogil except *:
    cdef size_t N = <size_t>sqrt(matrix.shape[0])
    _check_shape_oper_ket(N, matrix)
    cdef double complex trace = 0
    cdef size_t ptr = 0
    cdef size_t stride = N + 1
    for ptr in range(N):
        trace += matrix.data[ptr * stride]
    return trace


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

trace = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='trace',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
trace.__doc__ =\
    """Compute the trace (sum of digaonal elements) of a square matrix."""
trace.add_specialisations([
    (CSR, trace_csr),
    (Dense, trace_dense),
], _defer=True)

trace_oper_ket = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='trace_oper_ket',
    module=__name__,
    inputs=('matrix',),
    out=False,
)
trace_oper_ket.__doc__ =\
    """Compute the trace (sum of digaonal elements) of a stacked square matrix ."""
trace_oper_ket.add_specialisations([
    (CSR, trace_oper_ket_csr),
    (Dense, trace_oper_ket_dense),
], _defer=True)

del _inspect, _Dispatcher
