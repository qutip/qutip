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
from qutip.core.data cimport csr, CSR, Dense, Dia
from .inner import inner
from .trace import trace, trace_oper_ket
from .matmul import matmul

__all__ = [
    'expect', 'expect_csr', 'expect_dense', 'expect_dia', 'expect_data',
    'expect_csr_dense', 'expect_dia_dense',
    'expect_super', 'expect_super_csr', 'expect_super_dia', 'expect_super_dense',
    'expect_super_csr_dense', 'expect_super_dia_dense', 'expect_super_data',
]

cdef int _check_shape_ket(Data op, Data state) except -1 nogil:
    if (
        op.shape[1] != state.shape[0]  # Matrix multiplication
        or state.shape[1] != 1  # State is ket
        or op.shape[0] != op.shape[1]  # op must be square matrix
    ):
        raise ValueError("incorrect input shapes "
                         + str(op.shape) + " and " + str(state.shape))
    return 0

cdef int _check_shape_dm(Data op, Data state) except -1 nogil:
    if (
        op.shape[1] != state.shape[0]  # Matrix multiplication
        or state.shape[0] != state.shape[1]  # State is square
        or op.shape[0] != op.shape[1]  # Op is square
    ):
        raise ValueError("incorrect input shapes "
                         + str(op.shape) + " and " + str(state.shape))
    return 0

cdef int _check_shape_super(Data op, Data state) except -1 nogil:
    if state.shape[1] != 1:
        raise ValueError("expected a column-stacked matrix")
    if (
        op.shape[1] != state.shape[0]  # Matrix multiplication
        or op.shape[0] != op.shape[1]  # Square matrix
    ):
        raise ValueError("incompatible shapes "
                         + str(op.shape) + ", " + str(state.shape))
    return 0


cdef double complex _expect_csr_ket(CSR op, CSR state) except * nogil:
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

cdef double complex _expect_csr_dm(CSR op, CSR state) except * nogil:
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


cpdef double complex expect_super_csr(CSR op, CSR state) except *:
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


cpdef double complex expect_csr(CSR op, CSR state) except *:
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

cdef double complex _expect_csr_dense_ket(CSR op, Dense state) except * nogil:
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

cdef double complex _expect_csr_dense_dm(CSR op, Dense state) except * nogil:
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


cdef double complex _expect_dense_ket(Dense op, Dense state) except * nogil:
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

cdef double complex _expect_dense_dense_dm(Dense op, Dense state) except * nogil:
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


cpdef double complex expect_csr_dense(CSR op, Dense state) except *:
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


cpdef double complex expect_dense(Dense op, Dense state) except *:
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


cpdef double complex expect_super_csr_dense(CSR op, Dense state) except * nogil:
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


cpdef double complex expect_super_dense(Dense op, Dense state) except * nogil:
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


cpdef double complex expect_dia(Dia op, Dia state) except *:
    cdef double complex expect = 0.
    cdef idxint diag_bra, diag_op, diag_ket, i, length
    cdef idxint start_op, start_state, end_op, end_state
    if state.shape[1] == 1:
        _check_shape_ket(op, state)
        # Since the ket is sparse and possibly unsorted. Taking the n'th
        # element of the state require a loop on the diags. Thus 3 loops are
        # needed.
        for diag_ket in range(state.num_diag):
          #if -state.offsets[diag_ket] >= op.shape[1]:
          #  continue
          for diag_bra in range(state.num_diag):
            for diag_op in range(op.num_diag):
              if state.offsets[diag_ket] - state.offsets[diag_bra] + op.offsets[diag_op] == 0:
                expect += (
                  conj(state.data[diag_bra * state.shape[1]])
                  * state.data[diag_ket * state.shape[1]]
                  * op.data[diag_op * op.shape[1] - state.offsets[diag_ket]]
                )
    else:
      _check_shape_dm(op, state)
      for diag_op in range(op.num_diag):
        for diag_state in range(state.num_diag):
          if op.offsets[diag_op] == -state.offsets[diag_state]:

            start_op = max(0, op.offsets[diag_op])
            start_state = max(0, state.offsets[diag_state])
            end_op = min(op.shape[1], op.shape[0] + op.offsets[diag_op])
            end_state = min(state.shape[1], state.shape[0] + state.offsets[diag_state])
            length = min(end_op - start_op, end_state - start_state)

            for i in range(length):
              expect += (
                op.data[diag_op * op.shape[1] + i + start_op]
                * state.data[diag_state * state.shape[1] + i + start_state]
              )
    return expect


cpdef double complex expect_dia_dense(Dia op, Dense state) except *:
    cdef double complex expect = 0.
    cdef idxint i, diag_op, start_op, end_op, strideR, stride, start_state
    if state.shape[1] == 1:
        _check_shape_ket(op, state)
        for diag_op in range(op.num_diag):
            start_op = max(0, op.offsets[diag_op])
            end_op = min(op.shape[1], op.shape[0] + op.offsets[diag_op])
            for i in range(start_op, end_op):
                expect += (
                  op.data[diag_op * op.shape[1] + i]
                  * state.data[i]
                  * conj(state.data[i - op.offsets[diag_op]])
                )
    else:
      _check_shape_dm(op, state)
      stride = state.shape[0] + 1
      strideR = state.shape[0] if state.fortran else 1
      for diag_op in range(op.num_diag):
        start_op = max(0, op.offsets[diag_op])
        end_op = min(op.shape[1], op.shape[0] + op.offsets[diag_op])
        start_state = -op.offsets[diag_op] * strideR
        for i in range(start_op, end_op):
          expect += (
            op.data[diag_op * op.shape[1] + i]
            * state.data[start_state + i * stride]
          )
    return expect


cpdef double complex expect_super_dia(Dia op, Dia state) except *:
    cdef double complex expect = 0.
    _check_shape_super(op, state)
    cdef idxint diag_op, diag_state
    cdef idxint stride = <size_t> sqrt(state.shape[0]) + 1
    for diag_op in range(op.num_diag):
      for diag_state in range(state.num_diag):
        if (
            -state.offsets[diag_state] < op.shape[1]
            and -op.offsets[diag_op] - state.offsets[diag_state] >= 0
            and (-op.offsets[diag_op] - state.offsets[diag_state]) % stride == 0
        ):
            expect += state.data[diag_state * state.shape[1]] * op.data[diag_op * op.shape[1] - state.offsets[diag_state]]

    return expect


cpdef double complex expect_super_dia_dense(Dia op, Dense state) except *:
    cdef double complex expect = 0.
    _check_shape_super(op, state)
    cdef idxint col, diag_op, start, end
    cdef idxint stride = <size_t> sqrt(state.shape[0]) + 1
    for diag_op in range(op.num_diag):
      start = max(0, op.offsets[diag_op])
      end = min(op.shape[1], op.shape[0] + op.offsets[diag_op])
      col = (((start - op.offsets[diag_op] - 1) // stride) + 1) * stride + op.offsets[diag_op]
      while col < end:
        expect += op.data[diag_op * op.shape[1] + col] * state.data[col]
        col += stride
    return expect


def expect_data(Data op, Data state):
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        _check_shape_ket(op, state)
        return inner(state, matmul(op, state))
    _check_shape_dm(op, state)
    return trace(matmul(op, state))


def expect_super_data(Data op, Data state):
    """
    Perform the operation `tr(op @ state)` where `op` is supplied as a
    superoperator, and `state` is a column-stacked operator.
    """
    _check_shape_super(op, state)
    return trace_oper_ket(matmul(op, state))


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

expect = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('op', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('state', _inspect.Parameter.POSITIONAL_ONLY),
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
    (Dense, Dense, expect_dense),
    (Dia, Dense, expect_dia_dense),
    (Dia, Dia, expect_dia),
    (Data, Data, expect_data),
], _defer=True)

expect_super = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('op', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('state', _inspect.Parameter.POSITIONAL_ONLY),
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
    (Dense, Dense, expect_super_dense),
    (Dia, Dense, expect_super_dia_dense),
    (Dia, Dia, expect_super_dia),
    (Data, Data, expect_super_data),
], _defer=True)

del _inspect, _Dispatcher


cdef double complex expect_data_dense(Data op, Dense state) except *:
    cdef double complex out
    if type(op) is CSR:
        out = expect_csr_dense(op, state)
    elif type(op) is Dense:
        out = expect_dense(op, state)
    else:
        out = expect(op, state)
    return out


cdef double complex expect_super_data_dense(Data op, Dense state) except *:
    cdef double complex out
    if type(op) is CSR:
        out = expect_super_csr_dense(op, state)
    elif type(op) is Dense:
        out = expect_super_dense(op, state)
    else:
        out = expect_super(op, state)
    return out
