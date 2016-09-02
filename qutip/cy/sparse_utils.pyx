# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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
import scipy.sparse as sp
cimport numpy as np
from libc.math cimport abs
cimport cython

cdef inline int int_max(int x, int y):
    return x ^ ((x ^ y) & -(x < y))

include "parameters.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_bandwidth(
        np.ndarray[ITYPE_t, ndim=1] idx,
        np.ndarray[ITYPE_t, ndim=1] ptr,
        int nrows):
    """
    Calculates the max (mb), lower(lb), and upper(ub) bandwidths of a
    csr_matrix.
    """
    cdef int lb, ub, mb, ii, jj, ldist
    lb = -nrows
    ub = -nrows
    mb = 0

    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii + 1]):
            ldist = ii - idx[jj]
            lb = int_max(lb, ldist)
            ub = int_max(ub, -ldist)
            mb = int_max(mb, ub + lb + 1)

    return mb, lb, ub


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_profile(np.ndarray[ITYPE_t, ndim=1] idx,
        np.ndarray[ITYPE_t, ndim=1] ptr,
        int nrows):
    cdef int ii, jj, temp, ldist=0
    cdef LTYPE_t pro=0
    for ii in range(nrows):
        temp = 0
        for jj in range(ptr[ii], ptr[ii + 1]):
            ldist = idx[jj] - ii
            temp = int_max(temp, ldist)
        pro += temp
    return pro


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_permute(
        np.ndarray[cython.numeric, ndim=1] data,
        np.ndarray[ITYPE_t, ndim=1] idx,
        np.ndarray[ITYPE_t, ndim=1] ptr,
        int nrows,
        int ncols,
        np.ndarray[ITYPE_t, ndim=1] rperm,
        np.ndarray[ITYPE_t, ndim=1] cperm,
        int flag):
    """
    Permutes the rows and columns of a sparse CSR or CSC matrix according to
    the permutation arrays rperm and cperm, respectively.
    Here, the permutation arrays specify the new order of the rows and columns.
    i.e. [0,1,2,3,4] -> [3,0,4,1,2].
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[cython.numeric] new_data = np.zeros_like(data)
    cdef np.ndarray[ITYPE_t] new_idx = np.zeros_like(idx)
    cdef np.ndarray[ITYPE_t] new_ptr = np.zeros_like(ptr)
    cdef np.ndarray[ITYPE_t] perm_r
    cdef np.ndarray[ITYPE_t] perm_c
    cdef np.ndarray[ITYPE_t] inds

    if flag == 0:  # CSR matrix
        if len(rperm):
            inds = np.argsort(rperm).astype(ITYPE)
            perm_r = np.arange(len(rperm), dtype=ITYPE)[inds]

            for jj in range(nrows):
                ii = perm_r[jj]
                new_ptr[ii + 1] = ptr[jj + 1] - ptr[jj]

            for jj in range(nrows):
                new_ptr[jj + 1] = new_ptr[jj+1] + new_ptr[jj]

            for jj in range(nrows):
                k0 = new_ptr[perm_r[jj]]
                for kk in range(ptr[jj], ptr[jj + 1]):
                    new_idx[k0] = idx[kk]
                    new_data[k0] = data[kk]
                    k0 = k0 + 1

        if len(cperm):
            inds = np.argsort(cperm).astype(ITYPE)
            perm_c = np.arange(len(cperm), dtype=ITYPE)[inds]
            nnz = new_ptr[len(new_ptr) - 1]
            for jj in range(nnz):
                new_idx[jj] = perm_c[new_idx[jj]]

    elif flag == 1:  # CSC matrix
        if len(cperm):
            inds = np.argsort(cperm).astype(ITYPE)
            perm_c = np.arange(len(cperm), dtype=ITYPE)[inds]

            for jj in range(ncols):
                ii = perm_c[jj]
                new_ptr[ii + 1] = ptr[jj + 1] - ptr[jj]

            for jj in range(ncols):
                new_ptr[jj + 1] = new_ptr[jj + 1] + new_ptr[jj]

            for jj in range(ncols):
                k0 = new_ptr[perm_c[jj]]
                for kk in range(ptr[jj], ptr[jj + 1]):
                    new_idx[k0] = idx[kk]
                    new_data[k0] = data[kk]
                    k0 = k0 + 1

        if len(rperm):
            inds = np.argsort(rperm).astype(ITYPE)
            perm_r = np.arange(len(rperm), dtype=ITYPE)[inds]
            nnz = new_ptr[len(new_ptr) - 1]
            for jj in range(nnz):
                new_idx[jj] = perm_r[new_idx[jj]]

    return new_data, new_idx, new_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_reverse_permute(
        np.ndarray[cython.numeric, ndim=1] data,
        np.ndarray[ITYPE_t, ndim=1] idx,
        np.ndarray[ITYPE_t, ndim=1] ptr,
        int nrows,
        int ncols,
        np.ndarray[ITYPE_t, ndim=1] rperm,
        np.ndarray[ITYPE_t, ndim=1] cperm,
        int flag):
    """
    Reverse permutes the rows and columns of a sparse CSR or CSC matrix
    according to the original permutation arrays rperm and cperm, respectively.
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[cython.numeric, ndim=1] new_data = np.zeros_like(data)
    cdef np.ndarray[ITYPE_t, ndim=1] new_idx = np.zeros_like(idx)
    cdef np.ndarray[ITYPE_t, ndim=1] new_ptr = np.zeros_like(ptr)

    if flag == 0:  # CSR matrix
        if len(rperm):
            for jj in range(nrows):
                ii = rperm[jj]
                new_ptr[ii + 1] = ptr[jj + 1] - ptr[jj]

            for jj in range(nrows):
                new_ptr[jj + 1] = new_ptr[jj + 1] + new_ptr[jj]

            for jj in range(nrows):
                k0 = new_ptr[rperm[jj]]
                for kk in range(ptr[jj], ptr[jj + 1]):
                    new_idx[k0] = idx[kk]
                    new_data[k0] = data[kk]
                    k0 = k0 + 1

        if len(cperm):
            nnz = new_ptr[len(new_ptr) - 1]
            for jj in range(nnz):
                new_idx[jj] = cperm[new_idx[jj]]

    if flag == 1:  # CSC matrix
        if len(cperm):
            for jj in range(ncols):
                ii = cperm[jj]
                new_ptr[ii + 1] = ptr[jj + 1] - ptr[jj]

            for jj in range(ncols):
                new_ptr[jj + 1] = new_ptr[jj + 1] + new_ptr[jj]

            for jj in range(ncols):
                k0 = new_ptr[cperm[jj]]
                for kk in range(ptr[jj], ptr[jj + 1]):
                    new_idx[k0] = idx[kk]
                    new_data[k0] = data[kk]
                    k0 = k0 + 1

        if len(rperm) != 0:
            nnz = new_ptr[len(new_ptr) - 1]
            for jj in range(nnz):
                new_idx[jj] = rperm[new_idx[jj]]

    return new_data, new_idx, new_ptr

                    
@cython.boundscheck(False)
@cython.wraparound(False)
def _isdiag(np.ndarray[ITYPE_t, ndim=1] idx,
        np.ndarray[ITYPE_t, ndim=1] ptr,
        int nrows):
        
    cdef int row, num_elems
    for row in range(nrows):
        num_elems = ptr[row+1] - ptr[row]
        if num_elems > 1:
            return 0
        elif num_elems == 1:
            if idx[ptr[row]] != row:
                return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _csr_get_diag(complex[::1] data, int[::1] idx, int[::1] ptr, int k=0):
    
    cdef size_t row, jj
    cdef num_rows = ptr.shape[0]-1
    cdef int abs_k = abs(k)
    cdef int start, stop
    cdef np.ndarray[complex, ndim=1, mode='c'] out = np.zeros(num_rows-abs_k, dtype=complex)

    if k >= 0:
        start = 0
        stop = num_rows-abs_k
    else: #k < 0
        start = abs_k
        stop = num_rows

    for row in range(start, stop):
        for jj in range(ptr[row], ptr[row+1]):
            if idx[jj]-k == row:
                out[row-start] = data[jj]
                break
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _csr_kron(np.ndarray[complex, ndim=1, mode="c"] dataA, 
           np.ndarray[ITYPE_t, ndim=1, mode="c"] indsA,
           np.ndarray[ITYPE_t, ndim=1, mode="c"] indptrA,
           int rowsA, int colsA,
           np.ndarray[complex, ndim=1, mode="c"] dataB, 
           np.ndarray[ITYPE_t, ndim=1, mode="c"] indsB,
           np.ndarray[ITYPE_t, ndim=1, mode="c"] indptrB,
           int rowsB, int colsB):
    """
    Computes the kronecker product between two complex
    sparse matrices in CSR format.
    """
    cdef int out_nnz = dataA.shape[0] * dataB.shape[0]
    cdef int rows_out = rowsA * rowsB
    cdef int cols_out = colsA * colsB
    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] out_ptr = np.zeros(rows_out+1, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] out_inds = np.zeros(out_nnz, dtype=np.int32)
    cdef np.ndarray[complex, ndim=1, mode="c"] out_data = np.zeros(out_nnz, dtype = complex)
    cdef size_t ii, jj, ptrA, ptr
    cdef int row = 0
    cdef int ptr_start, ptr_end
    cdef int row_startA, row_endA, row_startB, row_endB, distA, distB, ptrB

    for ii in range(rowsA):
        row_startA = indptrA[ii]
        row_endA = indptrA[ii+1]
        distA = row_endA - row_startA

        for jj in range(rowsB):
            row_startB = indptrB[jj]
            row_endB = indptrB[jj+1]
            distB = row_endB - row_startB
    
            ptr_start = out_ptr[row]
            ptr_end = ptr_start + distB
    
            out_ptr[row+1] = out_ptr[row] + distA * distB
            row +=1
    
            for ptrA in range(row_startA, row_endA):
                ptrB = row_startB
                for ptr in range(ptr_start, ptr_end):
                    out_inds[ptr] = indsA[ptrA] * colsB + indsB[ptrB]
                    out_data[ptr] = dataA[ptrA] * dataB[ptrB]
                    ptrB += 1
        
                ptr_start += distB
                ptr_end += distB
    
    return sp.csr_matrix((out_data,out_inds,out_ptr), shape=(rows_out,cols_out))

