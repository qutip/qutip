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
cimport numpy as np
cimport cython
cimport libc.math

cdef extern from "src/zspmv.h" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec, 
                double complex a, double complex *out, int nrows)

include "complex_math.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[complex, ndim=1, mode="c"] spmv(
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
cpdef np.ndarray[complex, ndim=1, mode="c"] spmv_csr(complex[::1] data,
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
    cdef int num_rows = ptr.shape[0] - 1
    cdef np.ndarray[complex, ndim=1, mode="c"] out = np.zeros((num_rows), dtype=np.complex)
    zspmvpy(&data[0], &ind[0], &ptr[0], &vec[0], 1.0, &out[0], num_rows)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline void spmvpy(complex[::1] data,
            int[::1] ind,
            int[::1] ptr,
            complex[::1] vec,
            complex a,
            complex[::1] out):
    
    zspmvpy(&data[0], &ind[0], &ptr[0], &vec[0], a, &out[0], vec.shape[0])



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs(
        double t, 
        complex[::1] rho,
        complex[::1] data,
        int[::1] ind,
        int[::1] ptr):

    cdef int num_rows = rho.shape[0]
    cdef np.ndarray[complex, ndim=1, mode="c"] out = \
        np.zeros((num_rows), dtype=complex)
    zspmvpy(&data[0], &ind[0], &ptr[0], &rho[0], 1.0, &out[0], num_rows)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] cy_ode_psi_func_td(
        double t, 
        np.ndarray[CTYPE_t, ndim=1, mode="c"] psi, 
        object H_func,
        object args):

    H = H_func(t, args).data
    return -1j * spmv_csr(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] cy_ode_psi_func_td_with_state(
        double t,
        np.ndarray[CTYPE_t, ndim=1, mode="c"] psi,
        object H_func,
        object args):

    H = H_func(t, psi, args)
    return -1j * spmv_csr(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] cy_ode_rho_func_td(
        double t,
        np.ndarray[CTYPE_t, ndim=1, mode="c"] rho,
        object L0,
        object L_func,
        object args):
    cdef object L
    L = L0 + L_func(t, args).data
    return spmv_csr(L.data, L.indices, L.indptr, rho)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_psi(object op,
                    complex[::1] state,
                    int isherm):

    cdef complex[::1] y = spmv_csr(op.data, op.indices, op.indptr, state)
    cdef int row, num_rows = state.shape[0]
    cdef complex dot = 0
    for row from 0 <= row < num_rows:
        dot += conj(state[row]) * y[row]

    if isherm:
        return float(dot.real)
    else:
        return complex(dot)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_psi_csr(complex[::1] data,
                        int[::1] idx,
                        int[::1] ptr, 
                        complex[::1] state,
                        int isherm):

    cdef complex [::1] y = spmv_csr(data,idx,ptr,state)
    cdef int row, num_rows = state.shape[0]
    cdef complex dot = 0

    for row from 0 <= row < num_rows:
        dot += conj(state[row])*y[row]

    if isherm:
        return <double>dot
    else:
        return dot


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
        return <double>dot



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
        return <double>tr




