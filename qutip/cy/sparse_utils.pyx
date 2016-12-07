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
from qutip.fastsparse import fast_csr_matrix
cimport numpy as np
from libc.math cimport abs, sqrt
cimport cython

cdef extern from "complex.h":
    double creal(double complex x)
    double cimag(double complex x)

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
    cdef int ldist
    cdef int lb = -nrows
    cdef int ub = -nrows
    cdef int mb = 0
    cdef size_t ii, jj

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
    cdef LTYPE_t pro = 0
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
        if rperm.shape[0] != 0:
            inds = np.argsort(rperm).astype(ITYPE)
            perm_r = np.arange(rperm.shape[0], dtype=ITYPE)[inds]

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

        if cperm.shape[0] != 0:
            inds = np.argsort(cperm).astype(ITYPE)
            perm_c = np.arange(cperm.shape[0], dtype=ITYPE)[inds]
            nnz = new_ptr[new_ptr.shape[0] - 1]
            for jj in range(nnz):
                new_idx[jj] = perm_c[new_idx[jj]]

    elif flag == 1:  # CSC matrix
        if cperm.shape[0] != 0:
            inds = np.argsort(cperm).astype(ITYPE)
            perm_c = np.arange(cperm.shape[0], dtype=ITYPE)[inds]

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

        if rperm.shape[0] != 0:
            inds = np.argsort(rperm).astype(ITYPE)
            perm_r = np.arange(rperm.shape[0], dtype=ITYPE)[inds]
            nnz = new_ptr[new_ptr.shape[0] - 1]
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
        if rperm.shape[0] != 0:
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

        if cperm.shape[0] > 0:
            nnz = new_ptr[new_ptr.shape[0] - 1]
            for jj in range(nnz):
                new_idx[jj] = cperm[new_idx[jj]]

    if flag == 1:  # CSC matrix
        if cperm.shape[0] != 0:
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

        if cperm.shape[0] != 0:
            nnz = new_ptr[new_ptr.shape[0] - 1]
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
cpdef np.ndarray[complex, ndim=1, mode='c'] _csr_get_diag(complex[::1] data, 
    int[::1] idx, int[::1] ptr, int k=0):
    
    cdef size_t row, jj
    cdef int num_rows = ptr.shape[0]-1
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
@cython.cdivision(True)
def unit_row_norm(complex[::1] data, int[::1] ptr, int nrows):
    cdef size_t row, ii
    cdef double total
    for row in range(nrows):
        total = 0
        for ii in range(ptr[row], ptr[row+1]):
            total += creal(data[ii]) * creal(data[ii]) + cimag(data[ii]) * cimag(data[ii])
        total = sqrt(total)
        for ii in range(ptr[row], ptr[row+1]):
            data[ii] /= total


@cython.boundscheck(False)
@cython.wraparound(False)
def dense2D_to_fastcsr_cmode(complex[:, ::1] mat, int nrows, int ncols):
    cdef int nnz = 0
    cdef size_t ii, jj
    cdef np.ndarray[complex, ndim=1, mode='c'] data = np.zeros(nrows*ncols, dtype=complex)
    cdef np.ndarray[int, ndim=1, mode='c'] ind = np.zeros(nrows*ncols, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode='c'] ptr = np.zeros(nrows+1, dtype=np.int32)

    for ii in range(nrows):
        for jj in range(ncols):
            if mat[ii,jj] != 0:
                ind[nnz] = jj
                data[nnz] = mat[ii,jj]
                nnz += 1
        ptr[ii+1] = nnz

    if nnz < (nrows*ncols):
        return fast_csr_matrix((data[:nnz], ind[:nnz], ptr), shape=(nrows,ncols))
    else:
        return fast_csr_matrix((data, ind, ptr), shape=(nrows,ncols))


@cython.boundscheck(False)
@cython.wraparound(False)
def dense2D_to_fastcsr_fmode(complex[::1, :] mat, int nrows, int ncols):
    cdef int nnz = 0
    cdef size_t ii, jj
    cdef np.ndarray[complex, ndim=1, mode='c'] data = np.zeros(nrows*ncols, dtype=complex)
    cdef np.ndarray[int, ndim=1, mode='c'] ind = np.zeros(nrows*ncols, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode='c'] ptr = np.zeros(nrows+1, dtype=np.int32)

    for ii in range(nrows):
        for jj in range(ncols):
            if mat[ii,jj] != 0:
                ind[nnz] = jj
                data[nnz] = mat[ii,jj]
                nnz += 1
        ptr[ii+1] = nnz

    if nnz < (nrows*ncols):
        return fast_csr_matrix((data[:nnz], ind[:nnz], ptr), shape=(nrows,ncols))
    else:
        return fast_csr_matrix((data, ind, ptr), shape=(nrows,ncols))
