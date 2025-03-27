#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numbers

import numpy as np

cimport numpy as cnp
cimport cython

from qutip.core.data cimport csr, dense, idxint, CSR, Dense, Data, Dia, dia
from qutip.core.data.base import idxint_dtype
from qutip.settings import settings

cnp.import_array()

__all__ = [
    'ptrace', 'ptrace_csr', 'ptrace_dense', 'ptrace_csr_dense', 'ptrace_dia',
]

cdef tuple _parse_inputs(object dims, object sel, tuple shape):
    cdef Py_ssize_t i

    dims = np.atleast_1d(dims).astype(idxint_dtype).ravel()
    sel = np.atleast_1d(sel).astype(idxint_dtype)
    sel.sort()

    if shape[0] != shape[1]:
        raise ValueError("ptrace is only defined for square density matrices")

    if shape[0] != np.prod(dims, dtype=int):
        raise ValueError(f"the input matrix shape, {shape} and the"
                         f" dimension argument, {dims}, are not compatible.")
    if sel.ndim != 1:
        raise ValueError("Selection must be one-dimensional")

    if any(d < 1 for d in dims):
        raise ValueError("dimensions must be greated than zero but where"
                         f" dims={dims}.")

    for i in range(sel.shape[0]):
        if sel[i] < 0 or sel[i] >= dims.size:
            raise IndexError("Invalid selection index in ptrace.")
        if i > 0 and sel[i] == sel[i - 1]:
            raise ValueError("Duplicate selection index in ptrace.")

    return dims, sel

cdef idxint _populate_tensor_table(dims, sel, idxint[:, ::1] tensor_table) except -1:
    """
    Populate the helper structure `tensor_table`.  Returns the size (number of
    rows and number of columns) of the matrix which will be output.
    """
    cdef size_t ii
    cdef idxint[::1] _dims = np.asarray(dims, dtype=idxint_dtype).ravel()
    cdef size_t num_dims = _dims.shape[0]
    cdef idxint factor_tensor=1, factor_keep=1, factor_trace=1
    cdef idxint[::1] _sel = np.asarray(sel, dtype=idxint_dtype)
    for ii in range(_sel.shape[0]):
        if _sel[ii] < 0 or _sel[ii] >= num_dims:
            raise TypeError("Invalid selection index in ptrace.")
    for ii in range(num_dims - 1,-1,-1):
        tensor_table[ii, 0] = factor_tensor
        factor_tensor *= _dims[ii]
        if _in(ii, _sel):
            tensor_table[ii, 1] = factor_keep
            factor_keep *= _dims[ii]
        else:
            tensor_table[ii, 2]  = factor_trace
            factor_trace *= _dims[ii]
    return factor_keep


cdef bint _in(idxint val, idxint[::1] vec):
    cdef int ii
    for ii in range(vec.shape[0]):
        if val == vec[ii]:
            return True
    return False


cdef inline void _i2_k_t(idxint N, idxint[:, ::1] tensor_table, idxint out[2]):
    # indices determining function for ptrace
    cdef size_t ii
    cdef idxint t1, t2
    out[0] = out[1] = 0
    for ii in range(tensor_table.shape[0]):
        t1 = tensor_table[ii, 0]
        t2 = N / t1
        N = N % t1
        out[0] += tensor_table[ii, 1] * t2
        out[1] += tensor_table[ii, 2] * t2


cpdef CSR ptrace_csr(CSR matrix, object dims, object sel):
    dims, sel = _parse_inputs(dims, sel, matrix.shape)

    if len(sel) == len(dims):
        return matrix.copy()
    cdef idxint[:, ::1] tensor_table = np.zeros((dims.shape[0], 3), dtype=idxint_dtype)
    cdef idxint size
    size = _populate_tensor_table(dims, sel, tensor_table)
    cdef size_t p=0, nnz=csr.nnz(matrix), row, ptr
    cdef idxint pos_c[2]
    cdef idxint pos_r[2]
    cdef cnp.ndarray[double complex, ndim=1, mode='c'] new_data = np.zeros(nnz, dtype=complex)
    cdef cnp.ndarray[idxint, ndim=1, mode='c'] new_col = np.zeros(nnz, dtype=idxint_dtype)
    cdef cnp.ndarray[idxint, ndim=1, mode='c'] new_row = np.zeros(nnz, dtype=idxint_dtype)
    cdef double tol = 0
    if settings.core['auto_tidyup']:
        tol = settings.core['auto_tidyup_atol']
    for row in range(matrix.shape[0]):
        for ptr in range(matrix.row_index[row], matrix.row_index[row + 1]):
            _i2_k_t(matrix.col_index[ptr], tensor_table, pos_c)
            _i2_k_t(row, tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                new_data[p] = matrix.data[ptr]
                new_row[p] = pos_r[0]
                new_col[p] = pos_c[0]
                p += 1
    return csr.from_coo_pointers(&new_row[0], &new_col[0], &new_data[0],
                                 size, size, p, tol)


def ptrace_dia(matrix, dims, sel):
    if len(sel) == len(dims):
        return matrix.copy()
    dims, sel = _parse_inputs(dims, sel, matrix.shape)
    mat = matrix.as_scipy()
    cdef idxint[:, ::1] tensor_table = np.zeros((dims.shape[0], 3), dtype=idxint_dtype)
    cdef idxint pos_row[2]
    cdef idxint pos_col[2]
    size = _populate_tensor_table(dims, sel, tensor_table)
    data = {}
    for i, offset in enumerate(mat.offsets):
        start = max(0, offset)
        end = min(matrix.shape[0] + offset, matrix.shape[1])
        for col in range(start, end):
            _i2_k_t(col - offset, tensor_table, pos_row)
            _i2_k_t(col, tensor_table, pos_col)
            if pos_row[1] == pos_col[1]:
                new_offset = pos_col[0] - pos_row[0]
                if new_offset not in data:
                    data[new_offset] = np.zeros(size, dtype=complex)
                data[new_offset][pos_col[0]] += mat.data[i, col]

    if len(data) == 0:
        return dia.zeros(size, size)
    offsets = np.array(list(data.keys()), dtype=idxint_dtype)
    data = np.array(list(data.values()), dtype=complex)
    out = Dia((data, offsets), shape=(size, size), copy=False)
    out = dia.clean_dia(out, True)
    return out


cpdef Dense ptrace_csr_dense(CSR matrix, object dims, object sel):
    dims, sel = _parse_inputs(dims, sel, matrix.shape)

    if len(sel) == len(dims):
        return dense.from_csr(matrix)
    cdef idxint[:, ::1] tensor_table = np.zeros((dims.shape[0], 3), dtype=idxint_dtype)
    cdef idxint size
    size = _populate_tensor_table(dims, sel, tensor_table)
    cdef size_t ii, jj
    cdef idxint pos_c[2]
    cdef idxint pos_r[2]
    cdef Dense out = dense.zeros(size, size, fortran=False)
    for ii in range(matrix.shape[0]):
        for jj in range(matrix.row_index[ii], matrix.row_index[ii+1]):
            _i2_k_t(matrix.col_index[jj], tensor_table, pos_c)
            _i2_k_t(ii, tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                out.data[pos_r[0]*size + pos_c[0]] += matrix.data[jj]
    return out


cpdef Dense ptrace_dense(Dense matrix, object dims, object sel):
    dims, sel = _parse_inputs(dims, sel, matrix.shape)

    if len(sel) == len(dims):
        return matrix.copy()
    nd = dims.shape[0]
    dkeep = [dims[x] for x in sel]
    qtrace = list(set(np.arange(nd)) - set(sel))
    dtrace = [dims[x] for x in qtrace]
    dims = list(dims)
    sel = list(sel)
    rhomat = np.trace(matrix.as_ndarray()
                      .reshape(dims + dims)
                      .transpose(qtrace + [nd + q for q in qtrace] +
                                 sel + [nd + q for q in sel])
                      .reshape([np.prod(dtrace, dtype=int),
                                np.prod(dtrace, dtype=int),
                                np.prod(dkeep, dtype=int),
                                np.prod(dkeep, dtype=int)]))
    return dense.fast_from_numpy(rhomat)


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

ptrace = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('dims', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('sel', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='ptrace',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
ptrace.__doc__ =\
    """
    Compute the partial trace of this matrix, leaving the subspaces whose
    indices are in `sel`.  This is only defined for square density matrices,
    and always returns a density matrix.

    The order of the indices in `sel` do not matter; the output matrix will
    always have the selected subspaces in the same order that they are in the
    input matrix.

    For example, if the input is the matrix backing a `Qobj` with
    `dims = [[2, 3, 4], [2, 3, 4]]`, then the output of
        ptrace(data, [2, 3, 4], [0, 2])
    will be a matrix with effective dimensions `[[2, 4], [2, 4]]`.

    Parameters
    ----------
    matrix : Data
        The density matrix to be partially traced.

    dims : array_like of integer
        The dimensions of the subspaces.  This is most likely a 1D list, and
        since this is only defined on square matrices, there is no need to pass
        before the left and right sides.  Typically the input matrix will be
        taken from a `Qobj`, and then this parameter will be `Qobj.dims[0]`.

    sel : integer or array_like of integer
        The indices of the subspaces which should be _kept_.
    """
ptrace.add_specialisations([
    (CSR, CSR, ptrace_csr),
    (CSR, Dense, ptrace_csr_dense),
    (Dense, Dense, ptrace_dense),
    (Dia, Dia, ptrace_dia),
], _defer=True)

del _inspect, _Dispatcher
