#cython: language_level=3

from qutip.core.data cimport Data, Dense, dense, Dia, CSR
from qutip.core.data.matmul cimport imatmul_data_dense, matmul_dense
from qutip.core.data.matmul import matmul, matmul_dense_dia_dense
from qutip.core.data.add cimport iadd_dense
import numpy as np
cimport cython

#TODO:
# - One head function
# - Reusable interne functions + factory
# - What about jax etc?
# - merge left and right?


cdef class Hilbert_Iterator:
    """
    Allow to iterate over the ravelsome mode of an hilbert space.
    if:

        hilbert = [2, 3, 4]
        mode = [0, 2]

    then we want to iterate with:

        0, 1, 2, 3, 12, 13, 14, 16
    """
    # There are 2 methods implemented in parallel:
    # - Create an array of all indexes at initialization
    # - Compute the index as needed
    cdef:
        readonly object np_arr, np_index
        int[::1] iterator, step, sizes
        int[::1] indexes
        int N_mode, position
        bint pre_computed

    def __init__(self, hilbert, mode, pre_compute_size=1024):
        self.N_mode = len(mode)
        self.np_arr = np.zeros((3, self.N_mode), dtype=np.int32)
        self.position = 0
        self.iterator = self.np_arr[0, :]
        self.step = self.np_arr[1, :]
        self.sizes = self.np_arr[2, :]
        self.pre_computed = False

        hilberts_steps = [1]
        for i in range(len(hilbert)-1):
            hilberts_steps = [
                hilberts_steps[0] * hilbert[len(hilbert) - 1 - i]
            ] + hilberts_steps

        for i in range(self.N_mode):
            self.step[i] = hilberts_steps[mode[i]]
            self.sizes[i] = hilbert[mode[i]]

        if hilberts_steps[0] <= pre_compute_size:
            self.np_index = np.zeros(np.prod(self.np_arr[2, :]), dtype=np.int32)
            for i in range(np.prod(self.np_arr[2, :])):
                self.np_index[i] = self.next()
            self.indexes = self.np_index[:]
            self.pre_computed = True
            self.start()

    cdef int start(self):
        cdef int i
        self.position = 0
        for i in range(self.N_mode):
            self.iterator[i] = 0

    cdef int next(self):
        """
        Get the indexes in order
        """
        cdef int i = self.N_mode
        cdef int old_pos = self.position
        if self.pre_computed:


            old_pos = self.indexes[self.position]
            self.position += 1
            return old_pos

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

    cdef int at(self, int linear_idx):
        """
        Calculates and returns the index for the i-th element
        of the iteration sequence without changing the iterator's state.
        """
        cdef int final_pos = 0
        cdef int j, current_mode_idx

        if self.pre_computed:
            final_pos = self.indexes[linear_idx]
        else:
            for j in range(self.N_mode - 1, -1, -1):
                current_mode_idx = linear_idx % self.sizes[j]
                final_pos += current_mode_idx * self.step[j]
                linear_idx = linear_idx // self.sizes[j]

        return final_pos

    cdef (int, int) slices(self):
        """
        Return the starts and number of consecutive steps.
        """
        cdef int i = self.N_mode - 1
        cdef int old_pos = self.position

        if self.step[self.N_mode-1] != 1:
            return self.next(), 1

        if self.pre_computed:
            old_pos = self.indexes[self.position]
            self.position += self.sizes[self.N_mode-1]
            return old_pos, self.sizes[self.N_mode-1]

        while i:
            i = i - 1
            self.iterator[i] += 1
            self.position += self.step[i]
            if self.iterator[i] < self.sizes[i]:
                break
            self.iterator[i] = 0
            self.position -= self.sizes[i] * self.step[i]
        else:
            self.position = -1

        return old_pos, self.sizes[self.N_mode-1]


cdef class Data_interator:
    """
    iterator over Data instance.
    Data_interator.next() return a (row, col, value) tuple.
    Data_interator.nnz is the number of iteration to go over all values.

    ** After nnz call of next, it will return junk **
    """
    cdef int nnz

    def __init__(self, Data oper):
        self.nnz = 0

    cdef (int, int, double complex) next(self):
        return 0, 0, 0j


cdef class Dense_iterator(Data_interator):
    cdef Dense oper
    cdef int position

    def __init__(self, Dense oper):
        self.position = 0
        self.oper = oper
        self.nnz = oper.shape[0] * oper.shape[1]

    cdef (int, int, double complex) next(self):
        cdef double complex val = self.oper.data[self.position]
        cdef int row, col
        if self.oper.fortran:
            row = self.position % self.oper.shape[0]
            col = self.position // self.oper.shape[0]
        else:
            row = self.position // self.oper.shape[1]
            col = self.position % self.oper.shape[1]
        self.position += 1
        return row, col, val


cdef class CSR_iterator(Data_interator):
    cdef CSR oper
    cdef int row, idx

    def __init__(self, CSR oper):
        self.row = 0
        self.idx = 0
        while self.oper.row_index[self.row + 1] == self.idx:
            self.row += 1
        self.oper = oper
        self.nnz = oper.row_index[oper.shape[0]]

    cdef (int, int, double complex) next(self):
        cdef double complex val = self.oper.data[self.idx]
        cdef int row = self.row
        cdef int col = self.oper.col_index[self.idx]

        self.idx += 1
        while self.oper.row_index[self.row + 1] == self.idx:
            self.row += 1

        return row, col, val


cdef class Dia_iterator(Data_interator):
    cdef Dia oper
    cdef int diag_N, col, offset

    def __init__(self, Dia oper):
        self.diag_N = 0
        self.oper = oper
        self.offset = oper.offsets[0]
        self.col = max(0, self.offset)
        self.nnz = 0
        for i in range(oper.num_diag):
            start = max(0, oper.offsets[i])
            end = min(oper.shape[1], oper.shape[0] + oper.offsets[i])
            self.nnz += end - start

    cdef (int, int, double complex) next(self):
        cdef double complex val =\
            self.oper.data[self.diag_N * self.oper.shape[1] + self.col]
        cdef int row, col = self.col
        row = self.col - self.offset

        end = min(self.oper.shape[1], self.oper.shape[0] + self.offset)
        self.col += 1
        if self.col == end:
            self.diag_N += 1
            self.offset = self.oper.offsets[self.diag_N]
            self.col = max(0, self.offset)

        return row, col, val


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
    cdef int state_row_stride, state_col_stride, out_row_stride, out_col_stride
    cdef double complex val
    cdef Data_interator iter
    if type(oper) is Dense:
      iter = Dense_iterator(oper)
    if type(oper) is CSR:
      iter = CSR_iterator(oper)
    if type(oper) is Dia:
      iter = Dia_iterator(oper)

    state_row_stride = 1 if state.fortran else state.shape[0]
    state_col_stride = 1 if not state.fortran else state.shape[1]
    out_row_stride = 1 if state.fortran else state.shape[0]
    out_col_stride = 1 if not state.fortran else state.shape[1]
    pass_iter.start()

    for _ in range(iter.nnz):
        row, col, val = iter.next()
        row_idx = in_iter.at(row)
        col_idx = out_iter.at(col)
        if val == 0+0j:
            continue

        pass_iter.start()
        for i in range(N_pass_through):
            idx_pass = pass_iter.next()
            row_state = idx_pass + col_idx
            row_out = idx_pass + row_idx

            for j in range(state.shape[1]):
                out.data[row_out * out_row_stride + j * out_col_stride] += \
                    val * state.data[row_state * state_row_stride + j * state_col_stride]

    return out
