#cython: language_level=3
#cython: boundscheck=False, wraparound=False

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

from qutip.core.data.base cimport idxint
from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR

cdef double complex _inner_csr_bra_ket(CSR left, CSR right) nogil:
    cdef size_t col, ptr_bra, ptr_ket
    cdef double complex out = 0
    for ptr_bra in range(csr.nnz(left)):
        col = left.col_index[ptr_bra]
        ptr_ket = right.row_index[col]
        if right.row_index[col + 1] != ptr_ket:
            out += left.data[ptr_bra] * right.data[ptr_ket]
    return out

cdef double complex _inner_csr_ket_ket(CSR left, CSR right) nogil:
    cdef size_t row, ptr_l, ptr_r
    cdef double complex out = 0
    for row in range(left.shape[0]):
        ptr_l = left.row_index[row]
        ptr_r = right.row_index[row]
        if left.row_index[row+1] != ptr_l and right.row_index[row+1] != ptr_r:
            out += conj(left.data[ptr_l]) * right.data[ptr_r]
    return out

cpdef double complex inner_csr(CSR left, CSR right, bint scalar_is_ket=False) nogil:
    """
    Compute the complex inner product <left|right>.  The shape of `left` is
    used to determine if it has been supplied as a ket or a bra.  The result of
    this function will be identical if passed `left` or `adjoint(left)`.

    The parameter `scalar_is_ket` is only intended for the case where `left`
    and `right` are both of shape (1, 1).  In this case, `left` will be assumed
    to be a ket unless `scalar_is_ket` is False.  This parameter is ignored at
    all other times.
    """
    # At this level we simply assume that we were passed objects of the correct
    # shape, and don't check `right.shape[0]`.
    if left.shape[0] == left.shape[1] == right.shape[1] == 1:
        if csr.nnz(left) and csr.nnz(right):
            return (
                conj(left.data[0]) * right.data[0] if scalar_is_ket
                else left.data[0] * right.data[0]
            )
        return 0
    if left.shape[0] == 1:
        return _inner_csr_bra_ket(left, right)
    return _inner_csr_ket_ket(left, right)
