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
from qutip.core.data cimport csr, CSR, Dense

cdef double complex _expect_csr_ket(CSR op, CSR state) nogil except *:
    if op.shape[1] != state.shape[0] or state.shape[1] != 1:
        raise ValueError("incorrect input shapes "
                         + str(op.shape) + " and " + str(state.shape))
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
    if op.shape[0] != state.shape[1] or op.shape[1] != state.shape[0]:
        raise ValueError("incorrect input shapes "
                         + str(op.shape) + " and " + str(state.shape))
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
    # TODO: work out what on Earth this is.  Corresponds to cy_expect_rho_vec.
    if state.shape[1] != 1:
        raise ValueError("expected a column-stacked matrix")
    if op.shape[1] != state.shape[0]:
        raise ValueError("incompatible shapes " + str(op.shape) + ", " + str(state.shape))
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
    if state.shape[1] == 1:
        return _expect_csr_ket(op, state)
    return _expect_csr_dm(op, state)
