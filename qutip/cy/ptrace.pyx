#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
import numpy as np
from qutip.cy.spconvert import zcsr_reshape
from qutip.cy.spmath import zcsr_mult
from qutip.fastsparse import fast_csr_matrix, csr2fast
cimport numpy as cnp
cimport cython
from libc.math cimport floor, trunc
import scipy.sparse as sp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _ptrace_legacy(object rho, _sel):
    """
    Private function calculating the partial trace.
    """
    if np.prod(rho.dims[1]) == 1:
        rho = rho * rho.dag()

    cdef size_t mm, ii
    cdef int _tmp
    cdef cnp.ndarray[int, ndim=1, mode='c'] drho = np.asarray(rho.dims[0], dtype=np.int32).ravel()

    if isinstance(_sel, int):
        _sel = np.array([_sel], dtype=np.int32)
    else:
        _sel = np.asarray(_sel, dtype = np.int32)

    cdef int[::1] sel = _sel

    for mm in range(sel.shape[0]):
        if (sel[mm] < 0) or (sel[mm] >= drho.shape[0]):
            raise TypeError("Invalid selection index in ptrace.")

    cdef int[::1] rest = np.delete(np.arange(drho.shape[0],dtype=np.int32),sel)
    cdef int N = np.prod(drho)
    cdef int M = np.prod(drho.take(sel))
    cdef int R = np.prod(drho.take(rest))

    cdef int[:,::1] ilistsel = _select(sel, drho, M)
    cdef int[::1] indsel = _list2ind(ilistsel, drho)
    cdef int[:,::1] ilistrest = _select(rest, drho, R)
    cdef int[::1] indrest = _list2ind(ilistrest, drho)

    for mm in range(indrest.shape[0]):
        _tmp = indrest[mm] * N + indrest[mm]-1
        indrest[mm] = _tmp

    cdef cnp.ndarray[int, ndim=1, mode='c'] ind = np.zeros(M**2*indrest.shape[0],dtype=np.int32)
    for mm in range(M**2):
        for ii in range(indrest.shape[0]):
            ind[mm*indrest.shape[0]+ii] = indrest[ii] + \
                    N*indsel[<int>floor(mm / M)] + \
                    indsel[<int>(mm % M)]+1

    data = np.ones_like(ind,dtype=complex)
    ptr = np.arange(0,(M**2+1)*indrest.shape[0],indrest.shape[0], dtype=np.int32)
    perm = fast_csr_matrix((data,ind,ptr),shape=(M * M, N * N))
    # No need to sort here, will be sorted in reshape
    rhdata = zcsr_mult(perm, zcsr_reshape(rho.data, np.prod(rho.shape), 1), sorted=0)
    rho1_data = zcsr_reshape(rhdata, M, M)
    dims_kept0 = np.asarray(rho.dims[0], dtype=np.int32).take(sel)
    rho1_dims = [dims_kept0.tolist(), dims_kept0.tolist()]
    rho1_shape = [np.prod(dims_kept0), np.prod(dims_kept0)]
    return rho1_data, rho1_dims, rho1_shape


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[int, ndim=1, mode='c'] _list2ind(int[:,::1] ilist, int[::1] dims):
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[int, ndim=2, mode='c'] _select(int[::1] sel, int[::1] dims, int M):
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
            ilist[ii, _sel] = <int>(trunc(ii / _prd) % dims[_sel])
    return ilist


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _in(int val, int[::1] vec):
    # val in vec in pure cython
    cdef int ii
    for ii in range(vec.shape[0]):
        if val == vec[ii]:
            return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _ptrace(object rho, sel): # work for N<= 26 on 16G Ram
    cdef int[::1] _sel
    cdef object _oper
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
        _oper = (rho * rho.dag()).data
    else:
        _oper = rho.data

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
    rho1_shape = [np.prod(dims_kept0), np.prod(dims_kept0)]

    # Try to evaluate how sparse the result will be.
    if factor_keep*factor_keep > _oper.nnz:
        return csr2fast(_ptrace_core_sp(_oper, tensor_table, factor_keep)), rho1_dims, rho1_shape
    else:
        return csr2fast(_ptrace_core_dense(_oper, tensor_table, factor_keep)), rho1_dims, rho1_shape


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef object _ptrace_core_sp(rho, int[:, ::1] tensor_table, int num_sel_dims):
    cdef int p = 0, nnz = rho.nnz, ii, jj, nrow = rho.shape[0]
    cdef int[::1] pos_c = np.empty(2, dtype=np.int32)
    cdef int[::1] pos_r = np.empty(2, dtype=np.int32)
    cdef cnp.ndarray[complex, ndim=1, mode='c'] new_data = np.zeros(nnz, dtype=complex)
    cdef cnp.ndarray[int, ndim=1, mode='c'] new_col = np.zeros(nnz, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1, mode='c'] new_row = np.zeros(nnz, dtype=np.int32)
    cdef cnp.ndarray[complex, ndim=1, mode='c'] data = rho.data
    cdef cnp.ndarray[int, ndim=1, mode='c'] ptr = rho.indptr
    cdef cnp.ndarray[int, ndim=1, mode='c'] ind = rho.indices

    for ii in range(nrow):
        for jj in range(ptr[ii], ptr[ii+1]):
            _i2_k_t(ind[jj], tensor_table, pos_c)
            _i2_k_t(ii, tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                new_data[p] = data[jj]
                new_row[p] = (pos_r[0])
                new_col[p] = (pos_c[0])
                p += 1

    return sp.coo_matrix((new_data, [new_row, new_col]),
                         shape=(num_sel_dims,num_sel_dims)).tocsr()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef object _ptrace_core_dense(rho, int[:, ::1] tensor_table, int num_sel_dims):
    cdef int nnz = rho.nnz, ii, jj, nrow = rho.shape[0]
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

    return sp.coo_matrix(data_mat).tocsr()
