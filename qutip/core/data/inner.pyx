#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

from qutip.core.data.base cimport idxint, Data
from qutip.core.data cimport csr, dense
from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.dia cimport Dia
from qutip.core.data.matmul cimport matmul_dense
from .matmul import matmul
from .trace import trace
from .adjoint import adjoint

__all__ = [
    'inner', 'inner_csr', 'inner_dense', 'inner_dia', 'inner_data',
    'inner_op', 'inner_op_csr', 'inner_op_dense', 'inner_op_dia',
    'inner_op_data',
]


cdef int _check_shape_inner(Data left, Data right) except -1 nogil:
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
    return 0

cdef int _check_shape_inner_op(Data left, Data op, Data right) except -1 nogil:
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
    return 0

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

cpdef double complex inner_csr(CSR left, CSR right, bint scalar_is_ket=False) except *:
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

cpdef double complex inner_dia(Dia left, Dia right, bint scalar_is_ket=False) except * nogil:
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
    cdef double complex inner = 0.
    cdef idxint diag_left, diag_right
    cdef bint is_ket
    if right.shape[0] == 1:
        is_ket = scalar_is_ket
    else:
        is_ket = left.shape[0] == right.shape[0]

    if is_ket:
      for diag_right in range(right.num_diag):
        for diag_left in range(left.num_diag):
          if left.offsets[diag_left] - right.offsets[diag_right] == 0:
            inner += (
              conj(left.data[diag_left * left.shape[1]])
              * right.data[diag_right * right.shape[1]]
            )
    else:
      for diag_right in range(right.num_diag):
        for diag_left in range(left.num_diag):
          if left.offsets[diag_left] + right.offsets[diag_right] == 0:
            inner += (
              left.data[diag_left * left.shape[1] + left.offsets[diag_left]]
              * right.data[diag_right * right.shape[1]]
            )

    return inner

cpdef double complex inner_dense(Dense left, Dense right, bint scalar_is_ket=False) except * nogil:
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
        return (
                conj(left.data[0]) * right.data[0] if scalar_is_ket
                else left.data[0] * right.data[0]
        )
    cdef double complex out = 0
    cdef size_t i
    if left.shape[0] == 1:
        for i in range(right.shape[0]):
            out += left.data[i] * right.data[i]
    else:
        for i in range(right.shape[0]):
            out += conj(left.data[i]) * right.data[i]
    return out


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

cpdef double complex inner_op_dia(Dia left, Dia op, Dia right,
                                   bint scalar_is_ket=False) except * nogil:
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
    cdef double complex inner = 0., val
    cdef idxint diag_left, diag_op, diag_right
    cdef int is_ket
    if op.shape[0] == 1:
        is_ket = scalar_is_ket
    else:
        is_ket = left.shape[0] == op.shape[0]

    if is_ket:
      for diag_right in range(right.num_diag):
        for diag_left in range(left.num_diag):
          for diag_op in range(op.num_diag):
            if -left.offsets[diag_left] + right.offsets[diag_right] + op.offsets[diag_op] == 0:
              inner += (
                conj(left.data[diag_left])
                * right.data[diag_right]
                * op.data[diag_op * op.shape[1] - right.offsets[diag_right]]
              )
    else:
      for diag_right in range(right.num_diag):
        for diag_left in range(left.num_diag):
          for diag_op in range(op.num_diag):
            if left.offsets[diag_left] + right.offsets[diag_right] + op.offsets[diag_op] == 0:
              inner += (
                left.data[diag_left * left.shape[1] + left.offsets[diag_left]]
                * right.data[diag_right]
                * op.data[diag_op * op.shape[1] - right.offsets[diag_right]]
              )

    return inner

cpdef double complex inner_op_csr(CSR left, CSR op, CSR right,
                                  bint scalar_is_ket=False) except *:
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
    if 1 == left.shape[1] == left.shape[0] == op.shape[0] == op.shape[1] == right.shape[1]:
        if not (csr.nnz(left) and csr.nnz(op) and csr.nnz(right)):
            return 0
        l = conj(left.data[0]) if scalar_is_ket else left.data[0]
        return l * op.data[0] * right.data[0]
    if left.shape[0] == 1:
        return _inner_op_csr_bra_ket(left, op, right)
    return _inner_op_csr_ket_ket(left, op, right)

cpdef double complex inner_op_dense(Dense left, Dense op, Dense right,
                                  bint scalar_is_ket=False) except *:
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
    return inner_dense(left, matmul_dense(op, right), scalar_is_ket)


cpdef inner_data(Data left, Data right, bint scalar_is_ket=False):
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
        return (
            trace(left).conjugate() * trace(right) if scalar_is_ket
            else trace(left) * trace(right)
        )

    if left.shape[0] != 1:
        left = adjoint(left)
    # We use trace so we don't force convertion to complex.
    return trace(matmul(left, right))


cpdef inner_op_data(Data left, Data op, Data right, bint scalar_is_ket=False):
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
    return inner_data(left, matmul(op, right), scalar_is_ket)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

inner = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('scalar_is_ket', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=False),
    ]),
    name='inner',
    module=__name__,
    inputs=('left', 'right'),
    out=False,
)
inner.__doc__ =\
    """
    Compute the complex inner product <left|right>.  Return the complex value.

    The shape of `left` is used to determine if it has been supplied as a ket
    or a bra.  The result of this function will be identical if passed `left`
    or `adjoint(left)`.

    The parameter `scalar_is_ket` is only intended for the case where `left`
    and `right` are both of shape (1, 1).  In this case, `left` will be assumed
    to be a ket unless `scalar_is_ket` is False.  This parameter is ignored at
    all other times.

    Parameters
    ----------
    left : Data
        The left operand as either a bra or a ket matrix.

    right : Data
        The right operand as a ket matrix.

    scalar_is_ket : bool, optional (False)
        If `False`, then `left` is assumed to be a bra if it is
        one-dimensional.  If `True`, then it is assumed to be a ket.  This
        parameter is ignored if `left` and `right` are not one-dimensional.
    """
inner.add_specialisations([
    (CSR, CSR, inner_csr),
    (Dia, Dia, inner_dia),
    (Dense, Dense, inner_dense),
    (Data, Data, inner_data),
], _defer=True)

inner_op = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('op', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('scalar_is_ket', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=False),
    ]),
    name='inner_op',
    module=__name__,
    inputs=('left', 'op', 'right'),
    out=False,
)
inner_op.__doc__ =\
    """
    Compute the complex inner product <left|op|right>.  Return the complex
    value.  This operation is also known as taking a matrix element.

    The shape of `left` is used to determine if it has been supplied as a ket
    or a bra.  The result of this function will be identical if passed `left`
    or `adjoint(left)`.

    The parameter `scalar_is_ket` is only intended for the case where `left`
    and `right` are both of shape (1, 1).  In this case, `left` will be assumed
    to be a ket unless `scalar_is_ket` is False.  This parameter is ignored at
    all other times.

    Parameters
    ----------
    left : Data
        The left operand as either a bra or a ket matrix.

    op : Data
        The operator of which to take the matrix element.  Must have dimensions
        which match `left` and `right`.

    right : Data
        The right operand as a ket matrix.

    scalar_is_ket : bool, optional (False)
        If `False`, then `left` is assumed to be a bra if it is
        one-dimensional.  If `True`, then it is assumed to be a ket.  This
        parameter is ignored if `left` and `right` are not one-dimensional.
    """
inner_op.add_specialisations([
    (CSR, CSR, CSR, inner_op_csr),
    (Dia, Dia, Dia, inner_op_dia),
    (Dense, Dense, Dense, inner_op_dense),
    (Data, Data, Data, inner_op_data),
], _defer=True)

del _inspect, _Dispatcher
