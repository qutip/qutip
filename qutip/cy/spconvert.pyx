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
from qutip.fastsparse import fast_csr_matrix
cimport numpy as np
cimport cython

include "sparse_struct.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def coo2fast(object A):
    cdef int nnz = A.nnz
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]
    cdef complex[::1] data = A.data
    cdef int[::1] rows = A.rows
    cdef int[::1] cols = A.cols
    cdef CSR_Matrix out
    init_CSR(&out, nnz, nrows, ncols)
    
    cdef int i, j, iad, j0
    cdef double complex val
    cdef size_t kk
    # Determine row lengths
    for kk in range(nnz):
        out.indptr[rows[kk]] = out.indptr[rows[kk]] + 1
    # Starting position of rows
    j = 0
    for kk in range(nrows):
        j0 = out.indptr[kk]
        out.indptr[kk] = j
        j += j0
    #Do the data
    for kk in range(nnz):
        i = rows[kk]
        j = cols[kk]
        val = data[kk]
        iad = out.indptr[i]
        out.data[iad] = val
        out.indices[iad] = j
        out.indptr[i] = iad+1
    # Shift back
    for kk in range(nrows,0,-1):
        out.indptr[kk] = out.indptr[kk-1]
    out.indptr[0] = 0
    
    return CSR_to_scipy(&out)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
def arr_coo2fast(complex[::1] data, int[::1] rows, int[::1] cols, int nrows, int ncols):
    cdef int nnz = data.shape[0]
    cdef CSR_Matrix out
    init_CSR(&out, nnz, nrows, ncols)

    cdef int i, j, iad, j0
    cdef double complex val
    cdef size_t kk
    # Determine row lengths
    for kk in range(nnz):
        out.indptr[rows[kk]] = out.indptr[rows[kk]] + 1
    # Starting position of rows
    j = 0
    for kk in range(nrows):
        j0 = out.indptr[kk]
        out.indptr[kk] = j
        j += j0
    #Do the data
    for kk in range(nnz):
        i = rows[kk]
        j = cols[kk]
        val = data[kk]
        iad = out.indptr[i]
        out.data[iad] = val
        out.indices[iad] = j
        out.indptr[i] = iad+1
    # Shift back
    for kk in range(nrows,0,-1):
        out.indptr[kk] = out.indptr[kk-1]
    out.indptr[0] = 0

    return CSR_to_scipy(&out)
    
    
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

