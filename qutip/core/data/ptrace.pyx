#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
import scipy.sparse

cimport numpy as cnp

from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data cimport csr

cnp.import_array()

__all__ = [
    'ptrace_csr', 'ptrace_dense',
]


cdef cnp.ndarray[int, ndim=1, mode='c'] _list2ind(int[:,::1] ilist, int[::1] dims):
    """!
    Private function returning indicies
    """
    cdef size_t kk, ll
    cdef int[::1] fact = np.ones(dims.shape[0],dtype=np.int32)
    for kk in range(dims.shape[0]):
        for ll in range(kk+1,dims.shape[0]):
            fact[kk] *= dims[ll]
    # If we make ilist a csr_matrix, then this is just spmv then sort
    return np.sort(np.dot(ilist, fact), 0)


cdef cnp.ndarray[int, ndim=2, mode='c'] _select(int[::1] sel, int[::1] dims, int M):
    """
    Private function finding selected components
    """
    cdef size_t ii, jj, kk
    cdef int _sel, _prd
    cdef cnp.ndarray[int, ndim=2, mode='c'] ilist = np.zeros((M, dims.shape[0]), dtype=np.int32)
    for jj in range(sel.shape[0]):
        _sel =  sel[jj]
        _prd = 1
        for kk in range(jj+1,sel.shape[0]):
            _prd *= dims[sel[kk]]
        for ii in range(M):
            ilist[ii, _sel] = (ii / _prd) % dims[_sel]
    return ilist


cdef bint _in(int val, int[::1] vec):
    cdef int ii
    for ii in range(vec.shape[0]):
        if val == vec[ii]:
            return True
    return False


cdef void _i2_k_t(int N,
                 int[:, ::1] tensor_table,
                 int[::1] out):
    # indices determining function for ptrace
    cdef int ii, t1, t2
    out[0] = 0
    out[1] = 0
    for ii in range(tensor_table.shape[1]):
        t1 = tensor_table[0, ii]
        t2 = N / t1
        N = N % t1
        out[0] += tensor_table[1, ii] * t2
        out[1] += tensor_table[2, ii] * t2


cpdef ptrace_csr(object rho, sel):
    cdef int[::1] _sel
    cdef size_t ii
    cdef size_t factor_keep = 1, factor_trace = 1, factor_tensor = 1
    cdef cnp.ndarray[int, ndim=1, mode='c'] drho = np.asarray(rho.dims[0], dtype=np.int32).ravel()
    cdef int num_dims = drho.shape[0]
    cdef int[:, ::1] tensor_table = np.zeros((3, num_dims), dtype=np.int32)

    if isinstance(sel, int):
        _sel = np.array([sel], dtype=np.int32)
    else:
        _sel = np.asarray(sel, dtype=np.int32)

    for ii in range(_sel.shape[0]):
        if _sel[ii] < 0 or _sel[ii] >= num_dims:
            raise TypeError("Invalid selection index in ptrace.")

    if np.prod(rho.shape[1]) == 1:
        rho = rho.proj()

    for ii in range(num_dims-1,-1,-1):
        tensor_table[0, ii] = factor_tensor
        factor_tensor *= drho[ii]
        if _in(ii, _sel):
            tensor_table[1, ii] = factor_keep
            factor_keep *= drho[ii]
        else:
            tensor_table[2, ii] = factor_trace
            factor_trace *= drho[ii]

    dims_kept0 = drho.take(_sel).tolist()
    rho1_dims = [dims_kept0, dims_kept0]

    # TODO: actually dispatch properly
    return _ptrace_csr_sparse(rho.data, tensor_table, factor_keep), rho1_dims
    # Try to evaluate how sparse the result will be.
    #if factor_keep*factor_keep > csr.nnz(rho.data):
    #    return _ptrace_csr_sparse(rho.data, tensor_table, factor_keep), rho1_dims
    #return _ptrace_csr_dense(rho.data.to_array(), tensor_table, factor_keep), rho1_dims


cdef CSR _ptrace_csr_sparse(CSR rho, int[:, ::1] tensor_table, int num_sel_dims):
    cdef size_t p=0, nnz=csr.nnz(rho), row, ptr
    cdef int[::1] pos_c = np.empty(2, dtype=np.int32)
    cdef int[::1] pos_r = np.empty(2, dtype=np.int32)
    cdef cnp.ndarray[double complex, ndim=1, mode='c'] new_data = np.zeros(nnz, dtype=complex)
    cdef cnp.ndarray[int, ndim=1, mode='c'] new_col = np.zeros(nnz, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1, mode='c'] new_row = np.zeros(nnz, dtype=np.int32)
    for row in range(rho.shape[0]):
        for ptr in range(rho.row_index[row], rho.row_index[row + 1]):
            _i2_k_t(rho.col_index[ptr], tensor_table, pos_c)
            _i2_k_t(row, tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                new_data[p] = rho.data[ptr]
                new_row[p] = pos_r[0]
                new_col[p] = pos_c[0]
                p += 1
    return CSR(scipy.sparse.coo_matrix((new_data, [new_row, new_col]),
                                       shape=(num_sel_dims,num_sel_dims)).tocsr())


cdef CSR _ptrace_csr_dense(rho, int[:, ::1] tensor_table, int num_sel_dims):
    cdef int ii, jj, nrow=rho.shape[0]
    cdef int[::1] pos_c = np.empty(2, dtype=np.int32)
    cdef int[::1] pos_r = np.empty(2, dtype=np.int32)
    cdef cnp.ndarray[complex, ndim=1, mode='c'] data = rho.data
    cdef cnp.ndarray[int, ndim=1, mode='c'] ptr = rho.indptr
    cdef cnp.ndarray[int, ndim=1, mode='c'] ind = rho.indices
    cdef complex[:, ::1] data_mat = np.zeros((num_sel_dims, num_sel_dims),
                                          dtype=complex)
    for ii in range(nrow):
        for jj in range(ptr[ii], ptr[ii+1]):
            _i2_k_t(ind[jj], tensor_table, pos_c)
            _i2_k_t(ii, tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                data_mat[pos_r[0], pos_c[0]] += data[jj]
    return CSR(scipy.sparse.csr_matrix(data_mat))


cpdef ptrace_dense(Q, sel):
    rd = np.asarray(Q.dims[0], dtype=np.int32).ravel()
    nd = rd.shape[0]
    sel = [sel] if isinstance(sel, int) else list(np.sort(sel))
    dkeep = rd[sel].tolist()
    qtrace = list(set(np.arange(nd)) - set(sel))
    dtrace = rd[qtrace].tolist()
    rd = list(rd)
    if Q.type == 'ket':
        vmat = (Q.full()
                .reshape(rd)
                .transpose(sel + qtrace)
                .reshape([np.prod(dkeep), np.prod(dtrace)]))
        rhomat = vmat.dot(vmat.conj().T)
    else:
        rhomat = np.trace(Q.full()
                          .reshape(rd + rd)
                          .transpose(qtrace + [nd + q for q in qtrace] +
                                     sel + [nd + q for q in sel])
                          .reshape([np.prod(dtrace),
                                    np.prod(dtrace),
                                    np.prod(dkeep),
                                    np.prod(dkeep)]))
    return Dense(rhomat), [dkeep, dkeep]
