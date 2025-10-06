#cython: language_level=3

from qutip.core.data cimport Data, Dense, dense, Dia
from qutip.core.data.matmul cimport imatmul_data_dense, matmul_dense
from qutip.core.data.matmul import matmul, matmul_dense_dia_dense
from qutip.core.data.add cimport iadd_dense
import numpy as np


__all__ = ["one_mode_matmul_data_dense", "one_mode_matmul_dual_dense_data"]


cdef int _mul(list numbers) except -1:
    cdef int t, out = 1, N = len(numbers)
    for i in range(N):
        out = out * numbers[i]
    return out


cdef Dense _null_dense(int rows, int cols, bint fortran=True):
    cdef Dense out = Dense.__new__(Dense)
    out.shape = (rows, cols)
    out.data = <double complex *> 0
    out._deallocate = False
    out.fortran = fortran
    return out


cdef void _cut_dense(Dense base, Dense out, int loc):
    out.data = &base.data[loc]


cdef void imatmul_dense_data(Dense left, Data right, double complex scale, Dense out):
    if type(right) is Dia:
        matmul_dense_dia_dense(left, right, scale, out)
    elif type(right) is Dense:
        matmul_dense(left, right, scale, out)
    else:
        iadd_dense(out, matmul(left, right, dtype=Dense), scale)


cdef struct Shape:
    size_t num_row
    size_t num_col


cdef void _one_mode_matmul_kernel(
    Data oper,
    Dense state,
    Dense out,
    bint is_dual,
    int loop_count,
    Shape loc_shape_state,
    Shape loc_shape_out,
):
    """
    Generic kernel to perform block-wise matrix multiplication.
    """
    cdef Dense state_loc = _null_dense(
        loc_shape_state.num_row, loc_shape_state.num_col, is_dual
    )
    cdef Dense out_loc = _null_dense(
        loc_shape_out.num_row, loc_shape_out.num_col, is_dual
    )
    cdef size_t i, step_out, step_state
    step_state = loc_shape_state.num_row * loc_shape_state.num_col
    step_out = loc_shape_out.num_row * loc_shape_out.num_col

    for i in range(loop_count):
        # Set the data pointers for the current block views
        _cut_dense(state, state_loc, i * step_state)
        _cut_dense(out, out_loc, i * step_out)

        if not is_dual:
            # oper @ state_loc -> out_loc
            imatmul_data_dense(oper, state_loc, 1., out_loc)
        else:
            # state_loc @ oper -> out_loc
            imatmul_dense_data(state_loc, oper, 1., out_loc)


cpdef Dense one_mode_matmul_data_dense(
    Data oper, Dense state, list hilbert, int mode
):
    assert oper.shape[1] == hilbert[mode]
    cdef int N = _mul(hilbert)
    assert state.shape[0] == N
    cdef int before = _mul(hilbert[:mode])
    cdef int after = _mul(hilbert[mode+1:])
    cdef int N_out
    if oper.shape[0] != oper.shape[1]:
        N_out = N // oper.shape[1] * oper.shape[0]
    else:
        N_out = N
    cdef Dense out = dense.zeros(N_out, state.shape[1], state.fortran)

    cdef int loop_count, step_state, step_out
    cdef Shape loc_shape_state, loc_shape_out
    cdef bint loc_fortran

    if state.fortran:
        # Fortran-order parameters
        # Increments look like [[4, 2, 1], [32, 16, 8]]
        # We want blocks with increment [[N], [1]]
        loop_count = before * state.shape[1]
        loc_shape_state = Shape(oper.shape[1], after)
        loc_shape_out = Shape(oper.shape[0], after)
    else:
        # C-order parameters
        # Increments look like [[32, 16, 8], [4, 2, 1]]
        # We want blocks with increment [[N], [1]]
        loop_count = before
        loc_shape_state = Shape(oper.shape[1], after * state.shape[1])
        loc_shape_out = Shape(oper.shape[0], after * state.shape[1])

    _one_mode_matmul_kernel(
        oper, state, out,
        is_dual=False, loop_count=loop_count,
        loc_shape_state=loc_shape_state,
        loc_shape_out=loc_shape_out
    )

    return out


cpdef Dense one_mode_matmul_dual_dense_data(
    Dense state, Data oper, list hilbert, int mode
):
    cdef int N = _mul(hilbert)
    cdef int before = _mul(hilbert[:mode])
    cdef int after = _mul(hilbert[mode+1:])
    cdef int loop_count, N_out = N // oper.shape[0] * oper.shape[1]
    cdef Dense out = dense.zeros(state.shape[0], N_out, state.fortran)
    cdef Shape loc_shape_state, loc_shape_out

    if oper.shape[0] != hilbert[mode]:
        raise TypeError("Operator shape does not match the hilbert mode.")

    if state.shape[1] != N:
        raise TypeError("State shape does not match the hilbert space.")

    if state.fortran:
        # Fortran-order parameters
        # Increments look like [[4, 2, 1], [32, 16, 8]]
        # We want blocks with increment [[1], [N]]
        loop_count = before
        loc_shape_state = Shape(after * state.shape[0], oper.shape[0])
        loc_shape_out = Shape(after * state.shape[0], oper.shape[1])
    else:
        # C-order parameters
        # Increments look like [[32, 16, 8], [4, 2, 1]]
        # We want blocks with increment [[1], [N]]
        loop_count = before * state.shape[0]
        loc_shape_state = Shape(after, oper.shape[0])
        loc_shape_out = Shape(after, oper.shape[1])

    _one_mode_matmul_kernel(
        oper, state, out,
        is_dual=True, loop_count=loop_count,
        loc_shape_state=loc_shape_state,
        loc_shape_out=loc_shape_out,
    )

    return out
