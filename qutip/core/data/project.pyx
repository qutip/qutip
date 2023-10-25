#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.string cimport memcpy, memset

from qutip.core.data.base cimport idxint
from qutip.core.data cimport csr, dense, Dense, dia, Dia
from qutip.core.data.csr cimport CSR

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

__all__ = [
    'project', 'project_csr', 'project_dense', 'project_dia',
]


cdef int _project_ket_csr(CSR ket, CSR out) except -1 nogil:
    """
    Calculate the projection of the given ket, and place the output in out.
    """
    cdef size_t row, ptr, offset=0
    cdef idxint nnz_in = csr.nnz(ket)
    for row in range(ket.shape[0]):
        out.row_index[row] = ket.row_index[row] * nnz_in
        if ket.row_index[row + 1] != ket.row_index[row]:
            for ptr in range(nnz_in):
                out.col_index[offset + ptr*nnz_in] = row
            offset += 1
    out.row_index[ket.shape[0]] = nnz_in * nnz_in
    offset = 0
    for row in range(nnz_in):
        for ptr in range(nnz_in):
            out.data[offset + ptr] = ket.data[row] * conj(ket.data[ptr])
        offset += nnz_in
    return 0


cdef int _project_bra_csr(CSR bra, CSR out) except -1 nogil:
    """
    Calculate the projection of the given bra, and place the output in out.
    """
    cdef size_t row_out=0, ptr_bra, ptr, ptr_out=0, cur=0
    cdef double complex mul
    cdef idxint nnz_in = csr.nnz(bra)
    # The algorithm is much more simple conceptually if the indices are sorted,
    # and doesn't affect runtime much since sorting is worst-case
    # O(nnz log(nnz)) and the projection is O(nnz^2).  Also, much better to
    # sort now than later while there are fewer entries.  No need to sort when
    # projecting a ket because all the col_index values are 0.
    with gil:
        bra.sort_indices()
    out.row_index[0] = cur
    for ptr_bra in range(nnz_in):
        # Handle all zero rows between the last non-zero entry and this one.
        for row_out in range(row_out, bra.col_index[ptr_bra]):
            out.row_index[row_out + 1] = cur
        row_out = bra.col_index[ptr_bra]
        cur += nnz_in
        out.row_index[row_out + 1] = cur
        memcpy(&out.col_index[ptr_out], &bra.col_index[0], nnz_in * sizeof(idxint))
        mul = conj(bra.data[ptr_bra])
        for ptr in range(nnz_in):
            out.data[ptr_out] = mul * bra.data[ptr]
            ptr_out += 1
        row_out += 1
    # Handle all zero rows after the last non-zero entry.
    for row_out in range(row_out, out.shape[0]):
        out.row_index[row_out + 1] = cur
    return 0


cpdef CSR project_csr(CSR state):
    """
    Calculate the projection |state><state|.  The shape of `state` will be used
    to determine if it has been supplied as a ket or a bra.  The result of this
    function will be identical is passed `state` or `adjoint(state)`.
    """
    cdef size_t nnz_in = csr.nnz(state)
    cdef CSR out
    # We don't actually need to both handling the shape=(1, 1) case specially,
    # because it doesn't matter if we interpret it as a ket or a bra---the
    # output will be a shape=(1, 1) matrix with the only data element equal to
    # abs(state.data[0])**2 either way.
    if state.shape[1] == 1:
        out = csr.empty(state.shape[0], state.shape[0], nnz_in*nnz_in)
        _project_ket_csr(state, out)
        return out
    if state.shape[0] == 1:
        out = csr.empty(state.shape[1], state.shape[1], nnz_in*nnz_in)
        _project_bra_csr(state, out)
        return out
    raise ValueError("state must be a ket or a bra.")


cpdef Dense project_dense(Dense state):
    """
    Calculate the projection |state><state|.  The shape of `state` will be used
    to determine if it has been supplied as a ket or a bra.  The result of this
    function will be identical is passed `state` or `adjoint(state)`.
    """
    cdef Dense out
    cdef size_t size, i, j
    cdef bint fortran
    if state.shape[1] == 1:
        size = state.shape[0]
        fortran = True
    elif state.shape[0] == 1:
        size = state.shape[1]
        fortran = False
    else:
        raise ValueError("state must be a ket or a bra.")
    out = dense.zeros(size, size, fortran)
    for i in range(size):
        for j in range(size):
            out.data[i*size+j] = conj(state.data[i]) * state.data[j]
    return out


cpdef Dia project_dia(Dia state):
    """
    Calculate the projection |state><state|.  The shape of `state` will be used
    to determine if it has been supplied as a ket or a bra.  The result of this
    function will be identical is passed `state` or `adjoint(state)`.
    """
    cdef Dia out
    cdef size_t size, i, j, k, num_diag=0, max_diag
    if state.shape[1] == 1:
        size = state.shape[0]
    elif state.shape[0] == 1:
        size = state.shape[1]
    else:
        raise ValueError("state must be a ket or a bra.")

    max_diag = state.num_diag * state.num_diag - state.num_diag + 1
    out = dia.empty(size, size, max_diag)
    memset(out.data, 0, max_diag * out.shape[1] * sizeof(double complex))

    if state.shape[1] == 1:
        for i in range(state.num_diag):
            for j in range(state.num_diag):
                k = 0
                out_offset = (state.offsets[i] - state.offsets[j])
                while k < num_diag:
                    if out_offset == out.offsets[k]:
                        num_diag -= 1
                        break
                    k += 1
                num_diag += 1
                out.offsets[k] = out_offset

                out.data[k * size + abs(state.offsets[j])] = (
                    state.data[i] * conj(state.data[j])
                )
    else:
        for i in range(state.num_diag):
            for j in range(state.num_diag):
                k = 0
                out_offset = (state.offsets[i] - state.offsets[j])
                while k < num_diag:
                    if out_offset == out.offsets[k]:
                        num_diag -= 1
                        break
                    k += 1
                num_diag += 1
                out.offsets[k] = out_offset

                out.data[k * size + abs(state.offsets[i])] = (
                    state.data[i * state.shape[1] + abs(state.offsets[i])]
                    * conj(state.data[j * state.shape[1] + abs(state.offsets[j])])
                )

    out.num_diag = num_diag
    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

project = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('state', _inspect.Parameter.POSITIONAL_ONLY),
    ]),
    name='project',
    module=__name__,
    inputs=('state',),
    out=True,
)
project.__doc__ =\
    """
    Get the projector of a state with itself.  Mathematically, if passed an
    object `|a>` or `<a|`, then return the matrix `|a><a|`.

    Parameters
    ----------
    state : Data
        The input state bra- or ket-like vector.
    """
project.add_specialisations([
    (CSR, CSR, project_csr),
    (Dia, Dia, project_dia),
    (Dense, Dense, project_dense),
], _defer=True)

del _inspect, _Dispatcher
