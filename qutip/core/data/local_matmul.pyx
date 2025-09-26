#cython: language_level=3

from qutip.core.data cimport Data, Dense, dense, Dia
from qutip.core.data.matmul cimport imatmul_data_dense, matmul_dense
from qutip.core.data.matmul import matmul, matmul_dense_dia_dense
from qutip.core.data.add cimport iadd_dense
import numpy as np


__all__ = ["one_mode_matmul_data_dense", "one_mode_matmul_dual_dense_data"]


#TODO:
# - One head function
# - Reusable interne functions + factory
# - What about jax etc?



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


cdef class Hilbert_Iterator:
    """
    Allow to iterate over the ravelsome mode of an hilbert space.
    if:

        hilbert = [2, 3, 4]
        mode = [0, 2]

    then we want to iterate with:

        0, 1, 2, 3, 12, 13, 14, 16
    """
    cdef:
        object np_arr
        int[::1] iterator, step, sizes
        int N_mode, position

    def __init__(self, hilbert, mode):
        self.N_mode = len(mode)
        self.np_arr = np.zeros((3, self.N_mode), dtype=np.int32)
        self.position = 0
        self.iterator = self.np_arr[0, :]
        self.step = self.np_arr[1, :]
        self.sizes = self.np_arr[2, :]

        hilberts_steps = [1]
        for i in range(len(hilbert)-1):
            hilberts_steps = [
                hilberts_steps[0] * hilbert[len(hilbert) - i - 1]
            ] + hilberts_steps

        for i in range(self.N_mode):
            self.step[i] = hilberts_steps[mode[i]]
            self.sizes[i] = hilbert[mode[i]]


    cdef int start(self):
        cdef int i
        self.position = 0
        for i in range(self.N_mode):
            self.iterator[i] = 0

    cdef int next(self):
        cdef int i
        old_pos = self.position
        i = self.N_mode
        while i:
            i = i-1
            self.iterator[i] += 1
            self.position += self.step[i]
            if self.iterator[i] < self.sizes[i]:
                break
            self.iterator[i] = 0
            self.position -= self.sizes[i] * self.step[i]
        else:
            self.position = -1

        return old_pos

    cdef int at(int i):
        """return the ith position"""
        # same as self.start(); for _ in range(i): self.next(); return self.position
        # For small hilbert space, precomputing as a vector would be the fastest.
        # But


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef Dense cy_N_mode_dense(
    Dense oper, Dense state,
    list hilbert, list mode,
    list hilbert_out=None
):
    if hilbert_out is None:
        hilbert_out = hilbert

    cdef int out_size = state.shape[0] // oper.shape[1] * oper.shape[1]
    cdef Dense out = dense.zeros(out_size, state.shape[1], state.fortran)
    #TODO: oper sanity check, oper shape match hilbert and mode

    pass_through = [hilbert[i] for i in range(len(hilbert)) if i not in mode]
    not_mode = [i for i in range(len(hilbert)) if i not in mode]
    cdef int N_pass_through = int(np.prod(pass_through))

    cdef Hilbert_Iterator in_iter = Hilbert_Iterator(hilbert, mode)
    cdef Hilbert_Iterator out_iter = Hilbert_Iterator(hilbert_out, mode)
    cdef Hilbert_Iterator pass_iter = Hilbert_Iterator(hilbert, not_mode)

    cdef int row, col, row_idx, col_idx, idx_pass, i, j, row_state, row_out
    cdef list idx_state, idx_out
    cdef int oper_row_stride, oper_col_stride, state_row_stride, state_col_stride, out_row_stride, out_col_stride
    cdef double complex val

    oper_row_stride = 1 if oper.fortran else oper.shape[0]
    oper_col_stride = 1 if not oper.fortran else oper.shape[1]
    state_row_stride = 1 if state.fortran else state.shape[0]
    state_col_stride = 1 if not state.fortran else state.shape[1]
    out_row_stride = 1 if state.fortran else state.shape[0]
    out_col_stride = 1 if not state.fortran else state.shape[1]
    in_iter.start()
    out_iter.start()
    pass_iter.start()

    for row in range(oper.shape[0]):
        row_idx = in_iter.next()

        out_iter.start()
        for col in range(oper.shape[1]):
            val = oper.data[row * oper_row_stride + col * oper_col_stride]
            col_idx = out_iter.next()

            pass_iter.start()
            for i in range(N_pass_through):
                # TODO: continuous entry should use zaxpy
                idx_pass = pass_iter.next()
                row_state = idx_pass + col_idx
                row_out = idx_pass + row_idx

                for j in range(state.shape[1]):  # TODO: if `j` are continuous, use zaxpy
                    out.data[row_out * out_row_stride + j * out_col_stride] += \
                        val * state.data[row_state * state_row_stride + j * state_col_stride]

    return out


# TODO merge left and right?

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef Dense N_mode_data_dense(
    Data oper, Dense state,
    list hilbert, list mode,
    list hilbert_out=None
):
    if hilbert_out is None:
        hilbert_out = hilbert

    cdef int out_size = state.shape[0] // oper.shape[1] * oper.shape[1]
    cdef Dense out = dense.zeros(out_size, state.shape[1], state.fortran)
    #TODO: oper sanity check, oper shape match hilbert and mode

    pass_through = [hilbert[i] for i in range(len(hilbert)) if i not in mode]
    not_mode = [i for i in range(len(hilbert)) if i not in mode]
    cdef int N_pass_through = int(np.prod(pass_through))

    cdef Hilbert_Iterator in_iter = Hilbert_Iterator(hilbert, mode)
    cdef Hilbert_Iterator out_iter = Hilbert_Iterator(hilbert_out, mode)
    cdef Hilbert_Iterator pass_iter = Hilbert_Iterator(hilbert, not_mode)

    cdef int row, col, row_idx, col_idx, idx_pass, i, j, row_state, row_out
    cdef list idx_state, idx_out
    cdef int oper_row_stride, oper_col_stride, state_row_stride, state_col_stride, out_row_stride, out_col_stride
    cdef double complex val

    oper_row_stride = 1 if oper.fortran else oper.shape[0]
    oper_col_stride = 1 if not oper.fortran else oper.shape[1]
    state_row_stride = 1 if state.fortran else state.shape[0]
    state_col_stride = 1 if not state.fortran else state.shape[1]
    out_row_stride = 1 if state.fortran else state.shape[0]
    out_col_stride = 1 if not state.fortran else state.shape[1]
    in_iter.start()
    out_iter.start()
    pass_iter.start()

    for i in range(oper.nnz):
        row, col, val = oper.at(i)  # Iterator over non-zeros values returning a triplet row, col, val would make it work with all
        row_idx = in_iter.at(row)
        col_idx = out_iter.at(col)

        pass_iter.start()
        for i in range(N_pass_through):
            idx_pass = pass_iter.next()
            row_state = idx_pass + col_idx
            row_out = idx_pass + row_idx

            for j in range(state.shape[1]):
                out.data[row_out * out_row_stride + j * out_col_stride] += \
                    val * state.data[row_state * state_row_stride + j * state_col_stride]

    return out
