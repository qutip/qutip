#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

from qutip.core.data.base cimport idxint, Data
from qutip.core.data cimport csr
from qutip.core.data.csr cimport CSR

cdef void _check_shape_inner(Data left, Data right) nogil except *:
    if (
        (left.shape[0] != 1 and left.shape[1] != 1)
        or right.shape[1] != 1
    ):
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )

cdef void _check_shape_inner_op(Data left, Data op, Data right) nogil except *:
    cdef bint left_shape = left.shape[0] == 1 or left.shape[1] == 1
    cdef bint left_op = (
        (left.shape[0] == 1 and left.shape[1] == op.shape[0])
        or (left.shape[1] == 1 and left.shape[0] == op.shape[0])
    )
    cdef bint op_right = op.shape[1] == right.shape[0]
    cdef bint right_shape = right.shape[1] == 1
    if not (left_shape and left_op and op_right and right_shape):
        raise ValueError("".join([
            "incompatible matrix shapes ",
            str(left.shape),
            ", ",
            str(op.shape),
            " and ",
            str(right.shape),
        ]))

cdef double complex _inner_csr_bra_ket(CSR left, CSR right) nogil:
    cdef size_t col, ptr_bra, ptr_ket
    cdef double complex out = 0
    # We actually don't care if left is sorted or not.
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

cpdef double complex inner_csr(CSR left, CSR right, bint scalar_is_ket=False) nogil except *:
    """
    Compute the complex inner product <left|right>.  The shape of `left` is
    used to determine if it has been supplied as a ket or a bra.  The result of
    this function will be identical if passed `left` or `adjoint(left)`.

    The parameter `scalar_is_ket` is only intended for the case where `left`
    and `right` are both of shape (1, 1).  In this case, `left` will be assumed
    to be a ket unless `scalar_is_ket` is False.  This parameter is ignored at
    all other times.
    """
    _check_shape_inner(left, right)
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


cdef double complex _inner_op_csr_bra_ket(CSR left, CSR op, CSR right) nogil:
    cdef size_t ptr_l, ptr_op, ptr_r, row, col
    cdef double complex sum, out=0
    # left does not need to be sorted.
    for ptr_l in range(csr.nnz(left)):
        row = left.col_index[ptr_l]
        sum = 0
        for ptr_op in range(op.row_index[row], op.row_index[row + 1]):
            col = op.col_index[ptr_op]
            ptr_r = right.row_index[col]
            if ptr_r != right.row_index[col + 1]:
                sum += op.data[ptr_op] * right.data[ptr_r]
        out += left.data[ptr_l] * sum
    return out

cdef double complex _inner_op_csr_ket_ket(CSR left, CSR op, CSR right) nogil:
    cdef size_t ptr_l, ptr_op, ptr_r, row, col
    cdef double complex sum, out=0
    for row in range(op.shape[0]):
        ptr_l = left.row_index[row]
        if left.row_index[row + 1] == ptr_l:
            continue
        sum = 0
        for ptr_op in range(op.row_index[row], op.row_index[row + 1]):
            col = op.col_index[ptr_op]
            ptr_r = right.row_index[col]
            if ptr_r != right.row_index[col + 1]:
                sum += op.data[ptr_op] * right.data[ptr_r]
        out += conj(left.data[ptr_l]) * sum
    return out

cpdef double complex inner_op_csr(CSR left, CSR op, CSR right,
                                  bint scalar_is_ket=False) nogil except *:
    """
    Compute the complex inner product <left|op|right>.  The shape of `left` is
    used to determine if it has been supplied as a ket or a bra.  The result of
    this function will be identical if passed `left` or `adjoint(left)`.

    The parameter `scalar_is_ket` is only intended for the case where `left`
    and `right` are both of shape (1, 1).  In this case, `left` will be assumed
    to be a ket unless `scalar_is_ket` is False.  This parameter is ignored at
    all other times.
    """
    _check_shape_inner_op(left, op, right)
    cdef double complex l
    if left.shape[0] == left.shape[1] == op.shape[0] == op.shape[1] == right.shape[1] == 1:
        if not (csr.nnz(left) and csr.nnz(op) and csr.nnz(right)):
            return 0
        l = conj(left.data[0]) if scalar_is_ket else left.data[0]
        return l * op.data[0] * right.data[0]
    if left.shape[0] == 1:
        return _inner_op_csr_bra_ket(left, op, right)
    return _inner_op_csr_ket_ket(left, op, right)
