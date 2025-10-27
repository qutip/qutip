#cython: language_level=3

from qutip.core.data cimport Data, Dense, dense, Dia, CSR
from qutip.core.data.matmul cimport (
    imatmul_data_dense, matmul_dense, matmul_dense_dia_dense
)
from qutip.core.data.matmul import matmul
from qutip.core.data.add cimport iadd_dense
from qutip.core.data.data_iterator cimport Data_iterator, _make_iter
import numpy as np
cimport cython
from scipy.linalg.cython_blas cimport zaxpy

cdef extern from "<complex>" namespace "std" nogil:
    double complex _conj "conj"(double complex x)


__all__ = [
    "one_mode_matmul_data_dense",
    "one_mode_matmul_dual_dense_data",
    "n_mode_matmul_data_dense",
]


cdef int ONE = 1
cdef double NaN = np.nan


cdef int _mul(list numbers) except -1:
    cdef int t, out = 1, N = len(numbers)
    for i in range(N):
        out = out * numbers[i]
    return out


cpdef Dense _flatten_view_dense(Dense state):
    """ Get a view with a flat (ket) shape """
    cdef Dense out = Dense.__new__(Dense)
    state.as_ndarray()  # Ensure the data owned by numpy
    out.data = state.data
    out._np = state._np
    out.fortran = state.fortran
    out._deallocate = False
    out.shape[0] = state.shape[0]*state.shape[1]
    out.shape[1] = 1
    return out


cpdef Dense _unflatten_view_dense(Dense state, (int, int) shape):
    """ Get a reshaped view of the state """
    cdef Dense out = Dense.__new__(Dense)
    state.as_ndarray()  # Ensure the data owned by numpy
    out.data = state.data
    out._np = state._np
    out.fortran = state.fortran
    out._deallocate = False
    if shape[0] * shape[1] != state.shape[0]*state.shape[1]:
        raise ValueError("Unflattening the state cannot change total size.")
    out.shape[0] = shape[0]
    out.shape[1] = shape[1]
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


cdef class NModeMeta:
    cdef readonly tuple hilbert, mode, hilbert_out
    cdef readonly object np_maps
    cdef int[::1] out_map, in_map, pass_map_out, pass_map_in
    cdef int n_pass_through, pass_through_step
    cdef int state_row, out_row, oper_row, oper_col

    def __init__(self, hilbert, mode, hilbert_out=None):
        if hilbert_out is None:
            hilbert_out = hilbert
        square_oper = hilbert == hilbert_out

        if not square_oper:
            if len(hilbert) != len(hilbert_out):
                raise ValueError(
                    "The number of hilbert subspaces the must stay constant."
                )
            for i, (N_in, N_out) in enumerate(zip(hilbert, hilbert_out)):
                if i not in mode and N_in != N_out:
                    raise ValueError(
                        "Hilbert space dimensions for inactive modes must not change."
                    )
        for i in range(1, len(mode)):
            if not (0 <= mode[i] < len(hilbert)):
                raise ValueError(f"Mode outside the hilbert space: {mode}")
            for j in range(i):
                if mode[i] == mode[j]:
                    raise ValueError(f"Duplicated mode are not supported: {mode}")

        # Once created, these won't be used, but kept for output.
        self.mode = tuple(mode)
        self.hilbert = tuple(hilbert)
        self.hilbert_out = tuple(hilbert_out)

        # For sanity check at time of application
        self.state_row = _mul(hilbert)
        self.out_row = _mul(hilbert_out)
        self.oper_row = _mul([hilbert_out[i] for i in range(len(hilbert_out)) if i in mode])
        self.oper_col = _mul([hilbert[i] for i in range(len(hilbert)) if i in mode])

        pass_through = [hilbert[i] for i in range(len(hilbert)) if i not in mode]
        not_mode = [i for i in range(len(hilbert)) if i not in mode]
        self.n_pass_through = _mul(pass_through)
        i = len(hilbert)
        self.pass_through_step = 1
        while i:
            # Last hilbert space are continuous in memory
            # We batch them together in a single blas call
            i -= 1
            if i not in mode:
                self.pass_through_step *= hilbert[i]
                not_mode = not_mode[:-1]
            else:
                break
        self.n_pass_through //= self.pass_through_step

        self.np_maps = []
        self.in_map = self._compute_map(hilbert, mode)
        self.pass_map_in = self._compute_map(hilbert, not_mode)
        if square_oper:
            self.out_map = self.in_map
            self.pass_map_out = self.pass_map_in
        else:
            self.out_map = self._compute_map(hilbert_out, mode)
            self.pass_map_out = self._compute_map(hilbert_out, not_mode)

    cdef int[::1] _compute_map(self, list hilbert, list modes):
        hilberts_steps = [1]
        for i in range(len(hilbert)-1):
            hilberts_steps = [
                hilberts_steps[0] * hilbert[len(hilbert) - 1 - i]
            ] + hilberts_steps
        step = [0] * len(modes)
        sizes = [0] * len(modes)
        iterator = [0] * len(modes)
        for i, mode in enumerate(modes):
            step[i] = hilberts_steps[mode]
            sizes[i] = hilbert[mode]

        N = _mul(sizes)
        np_map = np.zeros(N, dtype=np.int32)
        position = 0
        self.np_maps.append(np_map)
        for i in range(N):
            np_map[i] = position
            j = len(modes)
            while j:
                j = j-1
                iterator[j] += 1
                position += step[j]
                if iterator[j] < sizes[j]:
                    break
                iterator[j] = 0
                position -= sizes[j] * step[j]
        return np_map


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef void n_mode_kernel(
    Data oper, Dense state, Dense out,
    NModeMeta meta, bint trans=False, bint conj=False
):
    # Even for 1 mode cases, there is a large speed difference according to the
    # mode. With mode = 0 the fastest.
    if (
        not trans and
        oper.shape[0] != meta.oper_row and
        oper.shape[1] != meta.oper_col
    ):
        raise TypeError(
            f"Expected an operator of shape ({meta.oper_row}, {meta.oper_col}),"
            f"but got ({oper.shape[0]}, {oper.shape[1]})"
        )

    if (
        trans and
        oper.shape[1] != meta.oper_row and
        oper.shape[0] != meta.oper_col
    ):
        raise TypeError(
            f"Expected an operator of shape ({meta.oper_row}, {meta.oper_col}),"
            f"but got ({oper.shape[1]}, {oper.shape[0]})"
        )

    if state.shape[0] != meta.state_row:
        raise TypeError(
            f"Expected a state of {meta.state_row} rows,"
            f"but got {state.shape[0]} rows."
        )

    if out.shape[0] != meta.out_row:
        raise TypeError(
            f"Expected a out of {meta.out_row} rows,"
            f"but got {out.shape[0]} rows."
        )

    if out.shape[1] != state.shape[1]:
        raise TypeError("state and out don't have the same number of columns.")

    cdef int row_idx, col_idx, idx_pass, i, j, k, row_state, row_out, idx_out, idx_state

    cdef int row, col
    cdef int state_row_stride, state_col_stride, out_row_stride, out_col_stride
    cdef double complex val

    cdef Data_iterator iter = _make_iter(oper, trans, conj)

    state_row_stride = 1 if state.fortran else state.shape[1]
    state_col_stride = 1 if not state.fortran else state.shape[0]
    out_row_stride = 1 if out.fortran else out.shape[1]
    out_col_stride = 1 if not out.fortran else out.shape[0]

    cdef int chunk_len = meta.pass_through_step
    if not state.fortran and not out.fortran:
        chunk_len *= state.shape[1]

    for _ in range(iter.nnz):
        row, col, val = iter.next()
        row_idx = meta.out_map[row]
        col_idx = meta.in_map[col]

        if val == 0+0j:
            continue
        if val == NaN:
            raise RuntimeError(f"Oper out of bound, row={row}, col={col}")

        for i in range(meta.n_pass_through):
            row_state = meta.pass_map_in[i] + col_idx
            row_out = meta.pass_map_out[i] + row_idx

            if not state.fortran and not out.fortran:
                zaxpy(
                    &chunk_len, &val,
                    &state.data[row_state * state_row_stride], &ONE,
                    &out.data[row_out * out_row_stride], &ONE
                )
            elif state.fortran and out.fortran:
                for j in range(state.shape[1]):
                    idx_out = row_out * out_row_stride + j * out_col_stride
                    idx_state = row_state * state_row_stride + j * state_col_stride

                    zaxpy(
                        &chunk_len, &val,
                        &state.data[idx_state], &ONE,
                        &out.data[idx_out], &ONE
                    )
            else:
                for k in range(chunk_len):
                  for j in range(state.shape[1]):
                    idx_out = (row_out+k) * out_row_stride + j * out_col_stride
                    idx_state = (row_state+k) * state_row_stride + j * state_col_stride
                    out.data[idx_out] += val * state.data[idx_state]


cpdef Dense n_mode_matmul_data_dense(
    Data oper, Dense state,
    list hilbert,
    list mode,
    list hilbert_out=None,
    bint transpose=False,
    bint conj=False,
):
    cdef NModeMeta meta = NModeMeta(hilbert, mode, hilbert_out)
    cdef Dense out = dense.zeros(meta.out_row, state.shape[1], state.fortran)
    n_mode_kernel(oper, state, out, meta, transpose, conj)
    return out
