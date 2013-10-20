# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np
cimport numpy as np
cimport cython
cimport libc.math


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] spmv(object super_op,
									   np.ndarray[CTYPE_t, ndim=1] vec):
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
cpdef np.ndarray[CTYPE_t, ndim=1] spmv_csr(np.ndarray[CTYPE_t, ndim=1] data,
                                       np.ndarray[int] idx,
                                       np.ndarray[int] ptr,
                                       np.ndarray[CTYPE_t, ndim=1] vec):
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
    cdef Py_ssize_t row
    cdef int jj,row_start,row_end
    cdef int num_rows = ptr.shape[0]-1
    cdef CTYPE_t dot
    cdef np.ndarray[CTYPE_t, ndim=1] out = np.zeros((num_rows), dtype=np.complex)
    for row in range(num_rows):
        dot=0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            dot+=data[jj]*vec[idx[jj]]
        out[row]=dot
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef spmvpy(np.ndarray[CTYPE_t, ndim=1] data,
             np.ndarray[int] idx,np.ndarray[int] ptr,
             np.ndarray[CTYPE_t, ndim=1] vec,
             CTYPE_t a, np.ndarray[CTYPE_t, ndim=1] out):
    """
    Sparse matrix time vector plus vector function:
    out = out + a * (data, idx, ptr) * vec
    """
    cdef Py_ssize_t row
    cdef int jj, row_start, row_end
    cdef int num_rows = vec.shape[0]
    cdef CTYPE_t dot
    for row in range(num_rows):
        dot = 0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start, row_end):
            dot = dot + data[jj] * vec[idx[jj]]
        out[row] = out[row] + a * dot
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_ode_rhs(double t, 
                                             np.ndarray[CTYPE_t, ndim=1] rho,
                                             np.ndarray[CTYPE_t, ndim=1] data,
                                             np.ndarray[int] idx,
                                             np.ndarray[int] ptr):
    cdef int row, jj, row_start, row_end
    cdef int num_rows = rho.shape[0]
    cdef CTYPE_t dot
    cdef np.ndarray[CTYPE_t, ndim=1] out = np.zeros((num_rows),dtype=np.complex)
    for row from 0 <= row < num_rows:
        dot = 0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            dot = dot + data[jj] * rho[idx[jj]]
        out[row] = dot
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_ode_psi_func_td(double t, 
                                                     np.ndarray[CTYPE_t, ndim=1] psi, 
                                                     object H_func,
                                                     object args):
    H = H_func(t, args)
    return -1j * spmv_csr(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_ode_psi_func_td_with_state(double t,
                                                                np.ndarray[CTYPE_t, ndim=1] psi,
                                                                object H_func,
                                                                object args):
    H = H_func(t, psi, args)
    return -1j * spmv_csr(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_ode_rho_func_td(double t,
                                                     np.ndarray[CTYPE_t, ndim=1] rho,
                                                     object L0,
                                                     object L_func,
                                                     object args):
    L = L0 + L_func(t, args)
    return spmv_csr(L.data, L.indices, L.indptr, rho)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] spmv_dia(np.ndarray[CTYPE_t, ndim=2] data,
                                           np.ndarray[int] offsets, 
                                           int num_rows, int num_diags,
                                           np.ndarray[CTYPE_t, ndim=1] vec,
                                           np.ndarray[CTYPE_t, ndim=1] ret,
                                           int N):
    """DIA sparse matrix-vector product
    """
    cdef int ii, jj,i0,i1,i2
    cdef CTYPE_t dot
    for ii in range(num_diags):
        i0=-offsets[ii]
        if i0>0:i1=i0
        else:i1=0
        if num_rows<num_rows+i0:
            i2=num_rows
        else:
            i2=num_rows+i0
        dot=0.0j
        for jj in range(i1,i2):
            dot+=data[ii,jj-i0]*vec[jj-i0]
        ret[jj]=dot
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_psi(object op,
                    np.ndarray[CTYPE_t, ndim=1] state,
                    int isherm):
    cdef np.ndarray[CTYPE_t, ndim=1] y = spmv_csr(op.data, op.indices, op.indptr, state)
    cdef np.ndarray[CTYPE_t, ndim=1] x = state.conj()
    cdef int row, num_rows = state.shape[0]
    cdef CTYPE_t dot = 0.0j
    for row from 0 <= row < num_rows:
        dot += x[row] * y[row]

    if isherm:
        return float(dot.real)
    else:
        return complex(dot)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_psi_csr(np.ndarray[CTYPE_t, ndim=1] data,
                        np.ndarray[int] idx,
                        np.ndarray[int] ptr, 
                        np.ndarray[CTYPE_t, ndim=1] state,
                        int isherm):
    cdef np.ndarray[CTYPE_t, ndim=1] y = spmv_csr(data,idx,ptr,state)
    cdef np.ndarray[CTYPE_t, ndim=1] x = state.conj()
    cdef int row, num_rows = state.shape[0]
    cdef CTYPE_t dot = 0.0j
    for row from 0 <= row < num_rows:
        dot+=x[row]*y[row]

    if isherm:
        return float(dot.real)
    else:
        return complex(dot)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_rho_vec(object super_op,
                        np.ndarray[CTYPE_t, ndim=1] rho_vec,
                        int herm):

    return cy_expect_rho_vec_csr(super_op.data, super_op.indices, super_op.indptr, rho_vec, herm)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_rho_vec_csr(np.ndarray[CTYPE_t, ndim=1] data,
                             np.ndarray[int] idx,
                             np.ndarray[int] ptr,
                             np.ndarray[CTYPE_t, ndim=1] rho_vec,
                             int herm):
    
    cdef Py_ssize_t row
    cdef int jj,row_start,row_end
    cdef int num_rows = rho_vec.shape[0]
    cdef int n = <int>libc.math.sqrt(num_rows)
    cdef CTYPE_t dot = 0.0
    for row from 0 <= row < num_rows by n+1:
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            dot += data[jj]*rho_vec[idx[jj]]
 
    if herm == 0:
        return dot
    else:
        return float(dot.real)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_spmm_tr(object op1, object op2, int herm):
    
    cdef Py_ssize_t row
    cdef CTYPE_t tr = 0.0

    cdef int col1, row1_idx_start, row1_idx_end
    cdef np.ndarray[CTYPE_t, ndim=1] data1 = op1.data
    cdef np.ndarray[int] idx1 = op1.indices
    cdef np.ndarray[int] ptr1 = op1.indptr

    cdef int col2, row2_idx_start, row2_idx_end
    cdef np.ndarray[CTYPE_t, ndim=1] data2 = op2.data
    cdef np.ndarray[int] idx2 = op2.indices
    cdef np.ndarray[int] ptr2 = op2.indptr

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
 
    if herm == 0:
        return tr
    else:
        return float(tr.real)




