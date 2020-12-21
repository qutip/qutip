#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

# The exported function `expect(op, state)` is equivalent to
# `inner_op(state.adjoint(), op, state)` if `state` is a ket, or
# `trace(op @ state)` if state is a density matrix, but it's optimised to not
# make unnecessary extra allocations, or calculate extra factors.

from libc.math cimport sqrt

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

from qutip.core.data.base cimport idxint, Data
from qutip.core.data cimport csr, csc, CSR, CSC, Dense

__all__ = [
    'expect', 'expect_super',
    'expect_csr', 'expect_dense', 'expect_super_csr', 'expect_super_dense',
    'expect_csr_dense', 'expect_csc_dense',
    'expect_super_csr_dense', 'expect_super_csc_dense',
]

cdef void _check_shape_ket(Data op, Data state) nogil except *:
    if op.shape[1] != state.shape[0] or state.shape[1] != 1:
        raise ValueError("incorrect input shapes "
                         + str(op.shape) + " and " + str(state.shape))

cdef void _check_shape_dm(Data op, Data state) nogil except *:
    if op.shape[0] != state.shape[1] or op.shape[1] != state.shape[0]:
        raise ValueError("incorrect input shapes "
                         + str(op.shape) + " and " + str(state.shape))

cdef void _check_shape_super(Data op, Data state) nogil except *:
    if state.shape[1] != 1:
        raise ValueError("expected a column-stacked matrix")
    if op.shape[1] != state.shape[0]:
        raise ValueError("incompatible shapes " + str(op.shape) + ", " + str(state.shape))


cdef double complex _expect_csr_ket(CSR op, CSR state) nogil except *:
    """
    Perform the operation
        state.adjoint() @ op @ state
    for a ket `state` and a square operator `op`.
    """
    _check_shape_ket(op, state)
    cdef double complex out=0, sum=0, mul
    cdef size_t row, col, ptr_op, ptr_ket
    for row in range(state.shape[0]):
        ptr_ket = state.row_index[row]
        if ptr_ket == state.row_index[row + 1]:
            continue
        sum = 0
        mul = conj(state.data[ptr_ket])
        for ptr_op in range(op.row_index[row], op.row_index[row + 1]):
            col = op.col_index[ptr_op]
            ptr_ket = state.row_index[col]
            if ptr_ket != state.row_index[col + 1]:
                sum += op.data[ptr_op] * state.data[ptr_ket]
        out += mul * sum
    return out

cdef double complex _expect_csr_dm(CSR op, CSR state) nogil except *:
    """
    Perform the operation
        tr(op @ state)
    for an operator `op` and a density matrix `state`.
    """
    _check_shape_dm(op, state)
    cdef double complex out=0
    cdef size_t row, col, ptr_op, ptr_state
    for row in range(op.shape[0]):
        for ptr_op in range(op.row_index[row], op.row_index[row + 1]):
            col = op.col_index[ptr_op]
            for ptr_state in range(state.row_index[col], state.row_index[col + 1]):
                if state.col_index[ptr_state] == row:
                    out += op.data[ptr_op] * state.data[ptr_state]
                    break
    return out


cpdef double complex expect_super_csr(CSR op, CSR state) nogil except *:
    """
    Perform the operation `tr(op @ state)` where `op` is supplied as a
    superoperator, and `state` is a column-stacked operator.
    """
    _check_shape_super(op, state)
    cdef double complex out = 0.0
    cdef size_t row=0, ptr, col
    cdef size_t n = <size_t> sqrt(state.shape[0])
    for _ in range(n):
        for ptr in range(op.row_index[row], op.row_index[row + 1]):
            col = op.col_index[ptr]
            if state.row_index[col] != state.row_index[col + 1]:
                out += op.data[ptr] * state.data[state.row_index[col]]
        row += n + 1
    return out


cpdef double complex expect_csr(CSR op, CSR state) nogil except *:
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        return _expect_csr_ket(op, state)
    return _expect_csr_dm(op, state)


cdef double complex _expect_csc_dense_ket(CSC op, Dense state) nogil except *:
    _check_shape_ket(op, state)
    cdef double complex out=0, sum
    cdef size_t row, ptr
    for row in range(op.shape[0]):
        if op.col_index[row] == op.col_index[row + 1]:
            continue
        sum = 0
        for ptr in range(op.col_index[row], op.col_index[row + 1]):
            sum += op.data[ptr] * conj(state.data[op.row_index[ptr]])
        out += sum * state.data[row]
    return out

cdef double complex _expect_csr_dense_ket(CSR op, Dense state) nogil except *:
    _check_shape_ket(op, state)
    cdef double complex out=0, sum
    cdef size_t row, ptr
    for row in range(op.shape[0]):
        if op.row_index[row] == op.row_index[row + 1]:
            continue
        sum = 0
        for ptr in range(op.row_index[row], op.row_index[row + 1]):
            sum += op.data[ptr] * state.data[op.col_index[ptr]]
        out += sum * conj(state.data[row])
    return out

cdef double complex _expect_csr_dense_dm(CSR op, Dense state) nogil except *:
    _check_shape_dm(op, state)
    cdef double complex out=0
    cdef size_t row, ptr_op, ptr_state=0, row_stride, col_stride
    row_stride = 1 if state.fortran else state.shape[1]
    col_stride = state.shape[0] if state.fortran else 1
    for row in range(op.shape[0]):
        if op.row_index[row] == op.row_index[row + 1]:
            continue
        ptr_state = row * col_stride
        for ptr_op in range(op.row_index[row], op.row_index[row + 1]):
            out += op.data[ptr_op] * state.data[ptr_state + row_stride*op.col_index[ptr_op]]
    return out


cdef double complex _expect_dense_ket(Dense op, Dense state) nogil except *:
    _check_shape_ket(op, state)
    cdef double complex out=0, sum
    cdef size_t row, col, op_row_stride, op_col_stride
    op_row_stride = 1 if op.fortran else op.shape[1]
    op_col_stride = op.shape[0] if op.fortran else 1

    for row in range(op.shape[0]):
        sum = 0
        for col in range(op.shape[0]):
            sum += (op.data[row * op_row_stride + col * op_col_stride] *
                    state.data[col])
        out += sum * conj(state.data[row])
    return out

cdef double complex _expect_dense_dense_dm(Dense op, Dense state) nogil except *:
    _check_shape_dm(op, state)
    cdef double complex out=0
    cdef size_t row, col, op_row_stride, op_col_stride
    cdef size_t state_row_stride, state_col_stride
    state_row_stride = 1 if state.fortran else state.shape[1]
    state_col_stride = state.shape[0] if state.fortran else 1
    op_row_stride = 1 if op.fortran else op.shape[1]
    op_col_stride = op.shape[0] if op.fortran else 1

    for row in range(op.shape[0]):
        for col in range(op.shape[1]):
            out += op.data[row * op_row_stride + col * op_col_stride] * \
                   state.data[col * state_row_stride + row * state_col_stride]
    return out


cpdef double complex expect_csr_dense(CSR op, Dense state) nogil except *:
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        return _expect_csr_dense_ket(op, state)
    return _expect_csr_dense_dm(op, state)


<<<<<<< HEAD
cpdef double complex expect_csc_dense(CSC op, Dense state) except *:
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        return _expect_csc_dense_ket(op, state)

    return _expect_csr_dense_dm(csc._as_tr_csr(op, False), state.transpose())


cpdef double complex expect_dense_dense(Dense op, Dense state) nogil except *:
=======
cpdef double complex expect_dense(Dense op, Dense state) nogil except *:
>>>>>>> solve.core
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        return _expect_dense_ket(op, state)
    return _expect_dense_dense_dm(op, state)


cpdef double complex expect_super_csr_dense(CSR op, Dense state) nogil except *:
    """
    Perform the operation `tr(op @ state)` where `op` is supplied as a
    superoperator, and `state` is a column-stacked operator.
    """
    _check_shape_super(op, state)
    cdef double complex out=0
    cdef size_t row=0, ptr
    cdef size_t n = <size_t> sqrt(state.shape[0])
    for _ in range(n):
        for ptr in range(op.row_index[row], op.row_index[row + 1]):
            out += op.data[ptr] * state.data[op.col_index[ptr]]
        row += n + 1
    return out


<<<<<<< HEAD
cpdef double complex expect_super_csc_dense(CSC op, Dense state) except *:
    """
    Perform the operation `tr(op @ state)` where `op` is supplied as a
    superoperator, and `state` is a column-stacked operator.
    """
    return expect_super_csr_dense(csc._as_tr_csr(op, False), state.transpose())


cpdef double complex expect_super_dense_dense(Dense op, Dense state) nogil except *:
=======
cpdef double complex expect_super_dense(Dense op, Dense state) nogil except *:
>>>>>>> solve.core
    """
    Perform the operation `tr(op @ state)` where `op` is supplied as a
    superoperator, and `state` is a column-stacked operator.
    """
    _check_shape_super(op, state)
    cdef double complex out=0
    cdef size_t row=0, col, N = state.shape[0]
    cdef size_t n = <size_t> sqrt(state.shape[0])
    cdef size_t op_row_stride, op_col_stride
    op_row_stride = 1 if op.fortran else op.shape[1]
    op_col_stride = op.shape[0] if op.fortran else 1

    for _ in range(n):
        for col in range(N):
            out += op.data[row * op_row_stride + col * op_col_stride] * \
                   state.data[col]
        row += n + 1
    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

expect = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('op', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('state', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='expect',
    module=__name__,
    inputs=('op', 'state'),
    out=False,
)
expect.__doc__ =\
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.  Returns a complex number.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
expect.add_specialisations([
    (CSR, CSR, expect_csr),
    (CSR, Dense, expect_csr_dense),
<<<<<<< HEAD
    (CSC, Dense, expect_csc_dense),
    (Dense, Dense, expect_dense_dense),
=======
    (Dense, Dense, expect_dense),
>>>>>>> solve.core
], _defer=True)

expect_super = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('op', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('state', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='expect_super',
    module=__name__,
    inputs=('op', 'state'),
    out=False,
)
expect_super.__doc__ =\
    """
    Perform the operation `tr(op @ state)` where `op` is supplied as a
    superoperator, and `state` is a column-stacked operator.  Returns a complex
    number.
    """
expect_super.add_specialisations([
    (CSR, CSR, expect_super_csr),
    (CSR, Dense, expect_super_csr_dense),
<<<<<<< HEAD
    (CSC, Dense, expect_super_csc_dense),
    (Dense, Dense, expect_super_dense_dense),
=======
    (Dense, Dense, expect_super_dense),
>>>>>>> solve.core
], _defer=True)

del _inspect, _Dispatcher


cdef double complex expect_data_dense(Data op, Dense state):
    cdef double complex out
    if type(op) is CSR:
        out = expect_csr_dense(op, state)
    elif type(op) is Dense:
        out = expect_dense(op, state)
    else:
        out = expect(op, state)
    return out


cdef double complex expect_super_data_dense(Data op, Dense state):
    cdef double complex out
    if type(op) is CSR:
        out = expect_super_csr_dense(op, state)
    elif type(op) is Dense:
        out = expect_super_dense(op, state)
    else:
        out = expect_super(op, state)
    return out
