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
cimport numpy as cnp
cimport cython
cimport libc.math
from libcpp cimport bool

cdef extern from "src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows)

include "complex_math.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv(
        object super_op,
        complex[::1] vec):
    """
    Sparse matrix, dense vector multiplication.
    Here the vector is assumed to have one-dimension.
    Matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    super_op : csr matrix
    vec : array
        Dense vector for multiplication.  Must be one-dimensional.

    Returns
    -------
    out : array
        Returns dense array.

    """
    return spmv_csr(super_op.data, super_op.indices, super_op.indptr, vec)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv_csr(complex[::1] data,
            int[::1] ind, int[::1] ptr, complex[::1] vec):
    """
    Sparse matrix, dense vector multiplication.
    Here the vector is assumed to have one-dimension.
    Matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    data : array
        Data for sparse matrix.
    idx : array
        Indices for sparse matrix data.
    ptr : array
        Pointers for sparse matrix data.
    vec : array
        Dense vector for multiplication.  Must be one-dimensional.

    Returns
    -------
    out : array
        Returns dense array.

    """
    cdef unsigned int num_rows = ptr.shape[0] - 1
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros((num_rows), dtype=np.complex)
    zspmvpy(&data[0], &ind[0], &ptr[0], &vec[0], 1.0, &out[0], num_rows)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def spmvpy_csr(complex[::1] data,
            int[::1] ind, int[::1] ptr, complex[::1] vec,
            complex alpha, complex[::1] out):
    """
    Sparse matrix, dense vector multiplication.
    Here the vector is assumed to have one-dimension.
    Matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    data : array
        Data for sparse matrix.
    idx : array
        Indices for sparse matrix data.
    ptr : array
        Pointers for sparse matrix data.
    vec : array
        Dense vector for multiplication.  Must be one-dimensional.
    alpha : complex
        Numerical coefficient for sparse matrix.
    out: array
        Output array

    """
    cdef unsigned int num_rows = vec.shape[0]
    zspmvpy(&data[0], &ind[0], &ptr[0], &vec[0], alpha, &out[0], num_rows)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void spmvpy(complex* data, int* ind, int* ptr,
            complex* vec,
            complex a,
            complex* out,
            unsigned int nrows):

    zspmvpy(data, ind, ptr, vec, a, out, nrows)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _spmm_c_py(complex* data, int* ind, int* ptr,
            complex* mat, complex a, complex* out,
            unsigned int sp_rows, unsigned int nrows, unsigned int ncols):
    """
    sparse*dense "C" ordered.
    """
    cdef int row, col, ii, jj, row_start, row_end
    for row from 0 <= row < sp_rows :
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            for col in range(ncols):
                out[row * ncols + col] += a*data[jj]*mat[ind[jj] * ncols + col]


cpdef void spmmpy_c(complex[::1] data, int[::1] ind, int[::1] ptr,
             complex[:,::1] M, complex a, complex[:,::1] out):
    """
    Sparse matrix, c ordered dense matrix multiplication.
    The sparse matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    data : array
        Data for sparse matrix.
    idx : array
        Indices for sparse matrix data.
    ptr : array
        Pointers for sparse matrix data.
    mat : array 2d
        Dense matrix for multiplication.  Must be in c mode.
    alpha : complex
        Numerical coefficient for sparse matrix.
    out: array
        Output array. Must be in c mode.

    """
    cdef unsigned int sp_rows = ptr.shape[0]-1
    cdef unsigned int nrows = M.shape[0]
    cdef unsigned int ncols = M.shape[1]
    _spmm_c_py(&data[0], &ind[0], &ptr[0], &M[0,0], 1.,
               &out[0,0], sp_rows, nrows, ncols)


cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmmc(object sparse,
                                                   complex[:,::1] mat):
    """
    Sparse matrix, c ordered dense matrix multiplication.
    The sparse matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    sparse : csr matrix
    mat : array 2d
        Dense matrix for multiplication. Must be in c mode.

    Returns
    -------
    out : array
         Keep input ordering
    """
    cdef unsigned int sp_rows = sparse.indptr.shape[0]-1
    cdef unsigned int ncols = mat.shape[1]
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out = \
                     np.zeros((sp_rows, ncols), dtype=complex)
    spmmpy_c(sparse.data, sparse.indices, sparse.indptr,
             mat, 1., out)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _spmm_f_py(complex* data, int* ind, int* ptr,
            complex* mat, complex a, complex* out,
            unsigned int sp_rows, unsigned int nrows, unsigned int ncols):
    """
    sparse*dense "F" ordered.
    """
    cdef int col
    for col in range(ncols):
        spmvpy(data, ind, ptr, mat+nrows*col, a, out+sp_rows*col, sp_rows)


cpdef void spmmpy_f(complex[::1] data, int[::1] ind, int[::1] ptr,
         complex[::1,:] mat, complex a, complex[::1,:] out):
    """
    Sparse matrix, fortran ordered dense matrix multiplication.
    The sparse matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    data : array
        Data for sparse matrix.
    idx : array
        Indices for sparse matrix data.
    ptr : array
        Pointers for sparse matrix data.
    mat : array 2d
        Dense matrix for multiplication.  Must be in fortran mode.
    alpha : complex
        Numerical coefficient for sparse matrix.
    out: array
        Output array. Must be in fortran mode.

    """
    cdef unsigned int sp_rows = ptr.shape[0]-1
    cdef unsigned int nrows = mat.shape[0]
    cdef unsigned int ncols = mat.shape[1]
    _spmm_f_py(&data[0], &ind[0], &ptr[0], &mat[0,0], 1.,
               &out[0,0], sp_rows, nrows, ncols)


cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmmf(object sparse,
                                                   complex[::1,:] mat):
    """
    Sparse matrix, fortran ordered dense matrix multiplication.
    The sparse matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    sparse : csr matrix
    mat : array 2d
        Dense matrix for multiplication. Must be in fortran mode.

    Returns
    -------
    out : array
    Keep input ordering
    """
    cdef unsigned int sp_rows = sparse.indptr.shape[0]-1
    cdef unsigned int ncols = mat.shape[1]
    cdef cnp.ndarray[complex, ndim=2, mode="fortran"] out = \
                     np.zeros((sp_rows, ncols), dtype=complex, order="F")
    spmmpy_f(sparse.data, sparse.indices, sparse.indptr,
             mat, 1., out)
    return out


cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmm(object sparse,
                                            cnp.ndarray[complex, ndim=2] mat):
    """
    Sparse matrix, dense matrix multiplication.
    The sparse matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    sparse : csr matrix
    mat : array 2d
    Dense matrix for multiplication. Can be in c or fortran mode.

    Returns
    -------
    out : array
    Keep input ordering
    """
    if mat.flags["F_CONTIGUOUS"]:
        return spmmf(sparse, mat)
    else:
        return spmmc(sparse, mat)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs(
        double t,
        complex[::1] rho,
        complex[::1] data,
        int[::1] ind,
        int[::1] ptr):

    cdef unsigned int nrows = rho.shape[0]
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = \
        np.zeros(nrows, dtype=complex)
    zspmvpy(&data[0], &ind[0], &ptr[0], &rho[0], 1.0, &out[0], nrows)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args):

    H = H_func(t, args).data
    return -1j * spmv_csr(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td_with_state(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args):

    H = H_func(t, psi, args)
    return -1j * spmv_csr(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rho_func_td(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] rho,
        object L0,
        object L_func,
        object args):
    cdef object L
    L = L0 + L_func(t, args).data
    return spmv_csr(L.data, L.indices, L.indptr, rho)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_psi(object A, complex[::1] vec, bool isherm):

    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr

    cdef size_t row, jj
    cdef int nrows = vec.shape[0]
    cdef complex expt = 0, temp, cval

    for row in range(nrows):
        cval = conj(vec[row])
        temp = 0
        for jj in range(ptr[row], ptr[row+1]):
            temp += data[jj]*vec[ind[jj]]
        expt += cval*temp

    if isherm :
        return real(expt)
    else:
        return expt


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_psi_csr(complex[::1] data,
                        int[::1] ind,
                        int[::1] ptr,
                        complex[::1] vec,
                        bool isherm):

    cdef size_t row, jj
    cdef int nrows = vec.shape[0]
    cdef complex expt = 0, temp, cval

    for row in range(nrows):
        cval = conj(vec[row])
        temp = 0
        for jj in range(ptr[row], ptr[row+1]):
            temp += data[jj]*vec[ind[jj]]
        expt += cval*temp

    if isherm :
        return real(expt)
    else:
        return expt


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_rho_vec(object super_op,
                        complex[::1] rho_vec,
                        int herm):

    return cy_expect_rho_vec_csr(super_op.data,
                                 super_op.indices,
                                 super_op.indptr,
                                 rho_vec,
                                 herm)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_rho_vec_csr(complex[::1] data,
                             int[::1] idx,
                             int[::1] ptr,
                             complex[::1] rho_vec,
                             int herm):

    cdef size_t row
    cdef int jj,row_start,row_end
    cdef int num_rows = rho_vec.shape[0]
    cdef int n = <int>libc.math.sqrt(num_rows)
    cdef complex dot = 0.0

    for row from 0 <= row < num_rows by n+1:
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            dot += data[jj]*rho_vec[idx[jj]]

    if herm == 0:
        return dot
    else:
        return real(dot)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_spmm_tr(object op1, object op2, int herm):

    cdef size_t row
    cdef complex tr = 0.0

    cdef int col1, row1_idx_start, row1_idx_end
    cdef complex[::1] data1 = op1.data
    cdef int[::1] idx1 = op1.indices
    cdef int[::1] ptr1 = op1.indptr

    cdef int col2, row2_idx_start, row2_idx_end
    cdef complex[::1] data2 = op2.data
    cdef int[::1] idx2 = op2.indices
    cdef int[::1] ptr2 = op2.indptr

    cdef int num_rows = ptr1.shape[0]-1

    for row in range(num_rows):

        row1_idx_start = ptr1[row]
        row1_idx_end = ptr1[row + 1]
        for row1_idx from row1_idx_start <= row1_idx < row1_idx_end:
            col1 = idx1[row1_idx]

            row2_idx_start = ptr2[col1]
            row2_idx_end = ptr2[col1 + 1]
            for row2_idx from row2_idx_start <= row2_idx < row2_idx_end:
                col2 = idx2[row2_idx]

                if col2 == row:
                    tr += data1[row1_idx] * data2[row2_idx]
                    break

    if herm == 0:
        return tr
    else:
        return real(tr)



@cython.boundscheck(False)
@cython.wraparound(False)
def expect_csr_ket(object A, object B, int isherm):

    cdef complex[::1] Adata = A.data
    cdef int[::1] Aind = A.indices
    cdef int[::1] Aptr = A.indptr
    cdef complex[::1] Bdata = B.data
    cdef int[::1] Bptr = B.indptr
    cdef int nrows = A.shape[0]

    cdef int j
    cdef size_t ii, jj
    cdef double complex cval=0, row_sum, expt = 0

    for ii in range(nrows):
        if (Bptr[ii+1] - Bptr[ii]) != 0:
            cval = conj(Bdata[Bptr[ii]])
            row_sum = 0
            for jj in range(Aptr[ii], Aptr[ii+1]):
                j = Aind[jj]
                if (Bptr[j+1] - Bptr[j]) != 0:
                    row_sum += Adata[jj]*Bdata[Bptr[j]]
            expt += cval*row_sum
    if isherm:
        return real(expt)
    else:
        return expt



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double complex zcsr_mat_elem(object A, object left, object right, bool bra_ket=1):
    """
    Computes the matrix element for an operator A and left and right vectors.
    right must be a ket, but left can be a ket or bra vector.  If left
    is bra then bra_ket = 1, else set bra_ket = 0.
    """
    cdef complex[::1] Adata = A.data
    cdef int[::1] Aind = A.indices
    cdef int[::1] Aptr = A.indptr
    cdef int nrows = A.shape[0]

    cdef complex[::1] Ldata = left.data
    cdef int[::1] Lind = left.indices
    cdef int[::1] Lptr = left.indptr
    cdef int Lnnz = Lind.shape[0]

    cdef complex[::1] Rdata = right.data
    cdef int[::1] Rind = right.indices
    cdef int[::1] Rptr = right.indptr

    cdef int j, go, head=0
    cdef size_t ii, jj, kk
    cdef double complex cval=0, row_sum, mat_elem=0

    for ii in range(nrows):
        row_sum = 0
        go = 0
        if bra_ket:
            for kk in range(head, Lnnz):
                if Lind[kk] == ii:
                    cval = Ldata[kk]
                    head = kk
                    go = 1
        else:
            if (Lptr[ii] - Lptr[ii+1]) != 0:
                cval = conj(Ldata[Lptr[ii]])
                go = 1

        if go:
            for jj in range(Aptr[ii], Aptr[ii+1]):
                j = Aind[jj]
                if (Rptr[j] - Rptr[j+1]) != 0:
                    row_sum += Adata[jj]*Rdata[Rptr[j]]
            mat_elem += cval*row_sum

    return mat_elem
