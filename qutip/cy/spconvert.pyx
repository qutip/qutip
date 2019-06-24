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
from qutip.fastsparse import fast_csr_matrix
cimport numpy as cnp
cimport cython
from libc.stdlib cimport div, malloc, free

cdef extern from "stdlib.h":
    ctypedef struct div_t:
        int quot
        int rem

include "sparse_routines.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def arr_coo2fast(complex[::1] data, int[::1] rows, int[::1] cols, int nrows, int ncols):
    """
    Converts a set of ndarrays (data, rows, cols) that specify a COO sparse matrix
    to CSR format.
    """
    cdef int nnz = data.shape[0]
    cdef COO_Matrix mat
    mat.data = &data[0]
    mat.rows = &rows[0]
    mat.cols = &cols[0]
    mat.nrows = nrows
    mat.ncols = ncols
    mat.nnz = nnz
    mat.is_set = 1
    mat.max_length = nnz

    cdef CSR_Matrix out
    COO_to_CSR(&out, &mat)
    return CSR_to_scipy(&out)


@cython.boundscheck(False)
@cython.wraparound(False)
def dense2D_to_fastcsr_cmode(complex[:, ::1] mat, int nrows, int ncols):
    """
    Converts a dense c-mode complex ndarray to a sparse CSR matrix.

    Parameters
    ----------
    mat : ndarray
        Input complex ndarray
    nrows : int
        Number of rows in matrix.
    ncols : int
        Number of cols in matrix.

    Returns
    -------
    out : fast_csr_matrix
        Output matrix in CSR format.
    """
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
def dense1D_to_fastcsr_ket(complex[::1] vec):
    """
    Converts a dense c-mode complex ndarray to a sparse CSR matrix.

    Parameters
    ----------
    mat : ndarray
        Input complex ndarray

    Returns
    -------
    out : fast_csr_matrix
        Output matrix in CSR format.
    """
    cdef int nnz = 0
    cdef size_t ii, nrows = vec.shape[0]
    cdef np.ndarray[complex, ndim=1, mode='c'] data = np.zeros(nrows, dtype=complex)
    cdef np.ndarray[int, ndim=1, mode='c'] ind = np.zeros(nrows, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode='c'] ptr = np.zeros(nrows+1, dtype=np.int32)

    for ii in range(nrows):
        if vec[ii] != 0:
            data[nnz] = vec[ii]
            nnz += 1
        ptr[ii+1] = nnz

    if nnz < (nrows):
        return fast_csr_matrix((data[:nnz], ind[:nnz], ptr), shape=(nrows,1))
    else:
        return fast_csr_matrix((data, ind, ptr), shape=(nrows,1))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fdense2D_to_CSR(complex[::1, :] mat, CSR_Matrix * out,
                                unsigned int nrows, unsigned int ncols):
    """
    Converts a dense complex ndarray to a CSR matrix struct.

    Parameters
    ----------
    mat : ndarray
        Input complex ndarray
    nrows : int
        Number of rows in matrix.
    ncols : int
        Number of cols in matrix.

    Returns
    -------
    out : CSR_Matrix
        Output matrix as CSR struct.
    """
    cdef int nnz = 0
    cdef size_t ii, jj
    init_CSR(out, nrows*ncols, nrows, ncols, nrows*ncols)

    for ii in range(nrows):
        for jj in range(ncols):
            if mat[ii,jj] != 0:
                out.indices[nnz] = jj
                out.data[nnz] = mat[ii,jj]
                nnz += 1
        out.indptr[ii+1] = nnz

    if nnz < (nrows*ncols):
        shorten_CSR(out, nnz)


@cython.boundscheck(False)
@cython.wraparound(False)
def dense2D_to_fastcsr_fmode(complex[::1, :] mat, int nrows, int ncols):
    """
    Converts a dense fortran-mode complex ndarray to a sparse CSR matrix.

    Parameters
    ----------
    mat : ndarray
        Input complex ndarray
    nrows : int
        Number of rows in matrix.
    ncols : int
        Number of cols in matrix.

    Returns
    -------
    out : fast_csr_matrix
        Output matrix in CSR format.
    """
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
def zcsr_reshape(object A not None, int new_rows, int new_cols):
    """
    Reshapes a complex CSR matrix.

    Parameters
    ----------
    A : fast_csr_matrix
        Input CSR matrix.
    new_rows : int
        Number of rows in reshaped matrix.
    new_cols : int
        Number of cols in reshaped matrix.

    Returns
    -------
    out : fast_csr_matrix
        Reshaped CSR matrix.

    Notes
    -----
    This routine does not need to make a temp. copy of the matrix.
    """
    cdef CSR_Matrix inmat = CSR_from_scipy(A)
    cdef COO_Matrix mat
    CSR_to_COO(&mat, &inmat)
    cdef CSR_Matrix out
    cdef div_t new_inds
    cdef size_t kk

    if (mat.nrows * mat.ncols) != (new_rows * new_cols):
        raise Exception('Total size of array must be unchanged.')

    for kk in range(mat.nnz):
        new_inds = div(mat.ncols*mat.rows[kk]+mat.cols[kk], new_cols)
        mat.rows[kk] = new_inds.quot
        mat.cols[kk] = new_inds.rem

    mat.nrows = new_rows
    mat.ncols = new_cols

    COO_to_CSR_inplace(&out, &mat)
    sort_indices(&out)
    return CSR_to_scipy(&out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_index_permute(int [::1] idx_arr,
                     int [::1] dims,
                     int [::1] order):

    cdef int ndims = dims.shape[0]
    cdef int ii, n, dim, idx, orderr

    #the fastest way to allocate memory for a temporary array
    cdef int * multi_idx = <int*> malloc(sizeof(int) * ndims)

    try:
        for ii from 0 <= ii < idx_arr.shape[0]:
            idx = idx_arr[ii]

            #First, decompose long index into multi-index
            for n from ndims > n >= 0:
                dim = dims[n]
                multi_idx[n] = idx % dim
                idx = idx // dim

            #Finally, assemble new long index from reordered multi-index
            dim = 1
            idx = 0
            for n from ndims > n >= 0:
                orderr = order[n]
                idx += multi_idx[orderr] * dim
                dim *= dims[orderr]

            idx_arr[ii] = idx
    finally:
        free(multi_idx)
