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


cpdef Dense one_mode_matmul_data_dense(Data oper, Dense state, list hilbert, int mode):
    cdef Dense out, out_loc, state_loc
    assert oper.shape[1] == hilbert[mode]
    cdef int step_state, step_out, i, N = _mul(hilbert)
    assert state.shape[0] == N

    cdef int before = _mul(hilbert[:mode])
    cdef int after = _mul(hilbert[mode+1:])

    if oper.shape[0] != oper.shape[1]:
        N_out = N // oper.shape[1] * oper.shape[0]
    else:
        N_out = N

    out = dense.zeros(N_out, state.shape[1], state.fortran)
    if state.fortran:
        # With fortran ordering
        # Increments look like [[4, 2, 1], [64, 32, 16]]
        # We want blocks with increment [[N], [1]]
        # the column idx move farter, as with the before sizes
        # So we need shape[1] * before loops of mode size * after block
        step_state = oper.shape[1] * after
        state_loc = _null_dense(oper.shape[1], after, False)
        step_out = oper.shape[0] * after
        out_loc = _null_dense(oper.shape[0], after, False)
        for i in range(before * state.shape[1]):
            _cut_dense(state, state_loc, i * step_state)
            _cut_dense(out, out_loc, i * step_out)
            imatmul_data_dense(oper, state_loc, 1., out_loc)
    else:
        # With C ordering, the rows idx move farter, as with the before sizes
        # So we need before loops of shape[1] * after * mode size block
        step_state = oper.shape[1] * after * state.shape[1]
        state_loc = _null_dense(oper.shape[1], after * state.shape[1], False)
        step_out = oper.shape[0] * after * state.shape[1]
        out_loc = _null_dense(oper.shape[0], after * state.shape[1], False)
        for i in range(before):
            _cut_dense(state, state_loc, i * step_state)
            _cut_dense(out, out_loc, i * step_out)
            imatmul_data_dense(oper, state_loc, 1., out_loc)  # Need real inplace

    return out


cdef void imatmul_dense_data(Dense left, Data right, double complex scale, Dense out):
    #if type(left) is CSR:
    #    matmul_csr_dense_dense(left, right, scale, out)
    if type(right) is Dia:
        matmul_dense_dia_dense(left, right, scale, out)
    elif type(right) is Dense:
        matmul_dense(left, right, scale, out)
    else:
        iadd_dense(out, matmul(left, right, dtype=Dense), scale)


cpdef Dense one_mode_matmul_dual_dense_data(Dense state, Data oper, list hilbert, int mode):
    cdef Dense out, out_loc, state_loc
    assert oper.shape[0] == hilbert[mode]
    cdef int step_state, step_out, i, N = _mul(hilbert)
    assert state.shape[1] == N

    cdef int before = _mul(hilbert[:mode])
    cdef int after = _mul(hilbert[mode+1:])

    if oper.shape[0] != oper.shape[1]:
        N_out = N // oper.shape[0] * oper.shape[1]
    else:
        N_out = N

    out = dense.zeros(state.shape[0], N_out, state.fortran)
    if state.fortran:
        # With fortran ordering
        # Increments look like [[4, 2, 1], [64, 32, 16]]
        # We want blocks with increment [[1], [N]]
        # the column idx move farter, as with the before sizes
        # So we need before loops of mode size * after * shape[0] block
        step_state = oper.shape[0] * after * state.shape[0]
        state_loc = _null_dense(after * state.shape[0], oper.shape[0], True)
        step_out = oper.shape[1] * after * state.shape[0]
        out_loc = _null_dense(after * state.shape[0], oper.shape[1], True)
        for i in range(before):
            _cut_dense(state, state_loc, i * step_state)
            _cut_dense(out, out_loc, i * step_out)
            imatmul_dense_data(state_loc, oper, 1., out_loc)
    else:
        # With C ordering, the rows idx move farter, as with the before sizes
        # Increments look like [[64, 32, 16], [4, 2, 1]]
        # We want blocks with increment [[1], [N]]
        # So we need before * shape[0] loops of  after * mode size block
        step_state = oper.shape[0] * after
        state_loc = _null_dense(after, oper.shape[1], True)
        step_out = oper.shape[1] * after
        out_loc = _null_dense(after, oper.shape[0], True)
        for i in range(before * state.shape[0]):
            _cut_dense(state, state_loc, i * step_state)
            _cut_dense(out, out_loc, i * step_out)
            imatmul_dense_data(state_loc, oper, 1., out_loc)  # Need real inplace

    return out
