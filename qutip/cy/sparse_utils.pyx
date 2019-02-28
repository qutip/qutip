#!python
#cython: language_level=3
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
cimport numpy as cnp
from libc.math cimport abs, fabs, sqrt
from libcpp cimport bool
cimport cython
cnp.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)


cdef extern from "<complex>" namespace "std" nogil:
    double abs(double complex x)
    double real(double complex x)
    double imag(double complex x)

cdef extern from "<complex>" namespace "std" nogil:
    double cabs "abs" (double complex x)

cdef inline int int_max(int x, int y):
    return x ^ ((x ^ y) & -(x < y))

include "parameters.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_bandwidth(
        int[::1] idx,
        int[::1] ptr,
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
def _sparse_profile(int[::1] idx,
        int[::1] ptr,
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
        cnp.ndarray[cython.numeric, ndim=1] data,
        int[::1] idx,
        int[::1] ptr,
        int nrows,
        int ncols,
        cnp.ndarray[ITYPE_t, ndim=1] rperm,
        cnp.ndarray[ITYPE_t, ndim=1] cperm,
        int flag):
    """
    Permutes the rows and columns of a sparse CSR or CSC matrix according to
    the permutation arrays rperm and cperm, respectively.
    Here, the permutation arrays specify the new order of the rows and columns.
    i.e. [0,1,2,3,4] -> [3,0,4,1,2].
    """
    cdef int ii, jj, kk, k0, nnz
    cdef cnp.ndarray[cython.numeric] new_data = np.zeros_like(data)
    cdef cnp.ndarray[ITYPE_t] new_idx = np.zeros_like(idx)
    cdef cnp.ndarray[ITYPE_t] new_ptr = np.zeros_like(ptr)
    cdef cnp.ndarray[ITYPE_t] perm_r
    cdef cnp.ndarray[ITYPE_t] perm_c
    cdef cnp.ndarray[ITYPE_t] inds

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
        cnp.ndarray[cython.numeric, ndim=1] data,
        int[::1] idx,
        int[::1] ptr,
        int nrows,
        int ncols,
        cnp.ndarray[ITYPE_t, ndim=1] rperm,
        cnp.ndarray[ITYPE_t, ndim=1] cperm,
        int flag):
    """
    Reverse permutes the rows and columns of a sparse CSR or CSC matrix
    according to the original permutation arrays rperm and cperm, respectively.
    """
    cdef int ii, jj, kk, k0, nnz
    cdef cnp.ndarray[cython.numeric, ndim=1] new_data = np.zeros_like(data)
    cdef cnp.ndarray[ITYPE_t, ndim=1] new_idx = np.zeros_like(idx)
    cdef cnp.ndarray[ITYPE_t, ndim=1] new_ptr = np.zeros_like(ptr)

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
def _isdiag(int[::1] idx,
        int[::1] ptr,
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
cpdef cnp.ndarray[complex, ndim=1, mode='c'] _csr_get_diag(complex[::1] data,
    int[::1] idx, int[::1] ptr, int k=0):

    cdef size_t row, jj
    cdef int num_rows = ptr.shape[0]-1
    cdef int abs_k = abs(k)
    cdef int start, stop
    cdef cnp.ndarray[complex, ndim=1, mode='c'] out = np.zeros(num_rows-abs_k, dtype=complex)

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
            total += real(data[ii]) * real(data[ii]) + imag(data[ii]) * imag(data[ii])
        total = sqrt(total)
        for ii in range(ptr[row], ptr[row+1]):
            data[ii] /= total



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double zcsr_one_norm(complex[::1] data, int[::1] ind, int[::1] ptr,
                     int nrows, int ncols):

    cdef int k
    cdef size_t ii, jj
    cdef double * col_sum = <double *>PyDataMem_NEW_ZEROED(ncols, sizeof(double))
    cdef double max_col = 0
    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii+1]):
            k = ind[jj]
            col_sum[k] += cabs(data[jj])
    for ii in range(ncols):
        if col_sum[ii] > max_col:
            max_col = col_sum[ii]
    PyDataMem_FREE(col_sum)
    return max_col


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double zcsr_inf_norm(complex[::1] data, int[::1] ind, int[::1] ptr,
                     int nrows, int ncols):

    cdef int k
    cdef size_t ii, jj
    cdef double * row_sum = <double *>PyDataMem_NEW_ZEROED(nrows, sizeof(double))
    cdef double max_row = 0
    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii+1]):
            row_sum[ii] += cabs(data[jj])
    for ii in range(nrows):
        if row_sum[ii] > max_row:
            max_row = row_sum[ii]
    PyDataMem_FREE(row_sum)
    return max_row


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool cy_tidyup(complex[::1] data, double atol, unsigned int nnz):
    """
    Performs an in-place tidyup of CSR matrix data
    """
    cdef size_t kk
    cdef double re, im
    cdef bool re_flag, im_flag, out_flag = 0
    for kk in range(nnz):
        re_flag = 0
        im_flag = 0
        re = real(data[kk])
        im = imag(data[kk])
        if fabs(re) < atol:
            re = 0
            re_flag = 1
        if fabs(im) < atol:
            im = 0
            im_flag = 1

        if re_flag or im_flag:
            data[kk] = re + 1j*im

        if re_flag and im_flag:
            out_flag = 1
    return out_flag
