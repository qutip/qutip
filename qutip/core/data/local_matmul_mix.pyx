#cython: language_level=3

from qutip.core.data cimport Data, Dense, dense, Dia, CSR
from qutip.core.data.matmul cimport imatmul_data_dense, matmul_dense
from qutip.core.data.matmul import matmul, matmul_dense_dia_dense
from qutip.core.data.add cimport iadd_dense
import numpy as np
cimport cython
from scipy.linalg.cython_blas cimport zaxpy


#TODO:
# - One head function
# - What about jax etc?
# - merge left and right?


cdef int ONE = 1


cdef int _mul(list numbers) except -1:
    cdef int t, out = 1, N = len(numbers)
    for i in range(N):
        out = out * numbers[i]
    return out


cdef class Data_iterator:
    """
    iterator over Data instance.
    Data_iterator.next() return a (row, col, value) tuple.
    Data_iterator.nnz is the number of iteration to go over all values.

    ** After nnz call of next, it will return 0, 0, inf **
    """
    cdef readonly int nnz

    def __init__(self, Data oper):
        self.nnz = 0

    cpdef (int, int, double complex) next(self):
        return 0, 0, float('inf')


cdef class Dense_iterator(Data_iterator):
    cdef Dense oper
    cdef int position

    def __init__(self, Dense oper):
        self.position = -1
        self.oper = oper
        self.nnz = oper.shape[0] * oper.shape[1]

    cpdef (int, int, double complex) next(self):
        self.position += 1

        if self.position >= self.nnz:
            return 0, 0, np.inf

        cdef double complex val = self.oper.data[self.position]
        cdef int row, col
        if self.oper.fortran:
            row = self.position % self.oper.shape[0]
            col = self.position // self.oper.shape[0]
        else:
            row = self.position // self.oper.shape[1]
            col = self.position % self.oper.shape[1]

        return row, col, val


cdef class CSR_iterator(Data_iterator):
    cdef CSR oper
    cdef int row, idx

    def __init__(self, CSR oper):
        self.row = 0
        self.idx = -1
        self.oper = oper
        self.nnz = oper.row_index[oper.shape[0]]

    cpdef (int, int, double complex) next(self):
        self.idx += 1

        if self.idx >= self.nnz:
            return 0, 0, np.inf

        cdef double complex val = self.oper.data[self.idx]
        cdef int col = self.oper.col_index[self.idx]

        while self.oper.row_index[self.row + 1] == self.idx:
            self.row += 1

        return self.row, col, val


cdef class Dia_iterator(Data_iterator):
    cdef Dia oper
    cdef int diag_N, col, offset, diag_end

    def __init__(self, Dia oper):
        self.oper = oper

        self.nnz = 0
        for i in range(oper.num_diag):
            start = max(0, oper.offsets[i])
            end = min(oper.shape[1], oper.shape[0] + oper.offsets[i])
            self.nnz += max(end - start, 0)

        self.diag_N = -1
        self.diag_end = -1
        self.col = 0
        self.offset = 0

    cpdef (int, int, double complex) next(self):
        self.col += 1

        while self.col >= self.diag_end:
            # While protect against badly formatted Dia.
            self.diag_N += 1
            if self.diag_N < self.oper.num_diag:
                self.offset = self.oper.offsets[self.diag_N]
                self.col = max(0, self.offset)
                self.diag_end = min(self.oper.shape[1], self.oper.shape[0] + self.offset)
            else:
                return 0, 0, np.inf

        return (
            self.col - self.offset,
            self.col,
            self.oper.data[self.diag_N * self.oper.shape[1] + self.col]
        )


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
                        "hilbert not affected by the operators must not change"
                    )
        for i in range(1, len(mode)):
            if not (0 <= mode[i] < len(hilbert)):
                ValueError(f"Mode outside the hilbert space: {mode}")
            for j in range(i):
                if mode[i] == mode[j]:
                    ValueError(f"Duplicated mode are not supported: {mode}")

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
        self.n_pass_through /= self.pass_through_step

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


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
cpdef void N_mode_kernel(Data oper, Dense state, Dense out, NModeMeta meta):
    if oper.shape[0] != meta.oper_row and oper.shape[1] != meta.oper_col:
        raise TypeError(
            f"Expected an operator of shape ({meta.oper_row}, {meta.oper_col}),"
            f"but got ({oper.shape[0]}, {oper.shape[1]})"
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
        raise TypeError("state and out don't have the same number of colunms.")

    cdef int row_idx, col_idx, idx_pass, i, j, row_state, row_out

    cdef int row, col
    cdef int state_row_stride, state_col_stride, out_row_stride, out_col_stride
    cdef double complex val

    cdef Data_iterator iter
    if type(oper) is Dense:
        iter = Dense_iterator(oper)
    elif type(oper) is CSR:
        iter = CSR_iterator(oper)
    elif type(oper) is Dia:
        iter = Dia_iterator(oper)
    else:
        raise RuntimeError(f"Can't make an iter out of {type(oper)}")

    state_row_stride = 1 if state.fortran else state.shape[1]
    state_col_stride = 1 if not state.fortran else state.shape[0]
    out_row_stride = 1 if out.fortran else out.shape[1]
    out_col_stride = 1 if not out.fortran else out.shape[0]

    cdef int chunk_len = meta.pass_through_step
    if not state.fortran and not out.fortran:
        chunk_len *= state.shape[1]

    debug = False

    if not state.fortran and not out.fortran and debug:
        print("long zaxpy")
        print(f"{chunk_len=}")
        print(f"{meta.np_maps=}")

    if state.fortran and out.fortran and debug:
        print("short zaxpy")
        print(f"{chunk_len=}")
        print(f"{meta.np_maps=}")

    for _ in range(iter.nnz):
        row, col, val = iter.next()
        row_idx = meta.out_map[row]
        col_idx = meta.in_map[col]

        if val == 0+0j:
            continue
        if val == np.inf:
            raise RuntimeError(f"Oper out of bound, {row=}, {col=}, {val=}")

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


cpdef Dense N_mode_data_dense(
    Data oper, Dense state,
    list hilbert, list mode,
    list hilbert_out=None
):
    cdef NModeMeta meta = NModeMeta(hilbert, mode, hilbert_out)
    cdef Dense out = dense.zeros(meta.out_row, state.shape[1], state.fortran)
    N_mode_kernel(oper, state, out, meta)
    return out
