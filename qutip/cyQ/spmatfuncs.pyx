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
ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=2] spmv(np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr,np.ndarray[CTYPE_t, ndim=2] vec):
    """
    Sparse matrix, dense vector multiplication.  
    Here the vector is assumed to have two-dimensions.
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
        Dense vector for multiplication.  Must be two-dimensional.
    
    Returns
    -------
    out : array
        Returns dense, two-dimensional, array.
    
    """
    cdef Py_ssize_t row
    cdef int jj,row_start,row_end
    cdef int num_rows=len(vec)
    cdef CTYPE_t dot
    cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)
    for row in range(num_rows):
        dot=0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            dot+=data[jj]*vec[idx[jj],0]
        out[row,0]=dot
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=2] spmv1d(np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr,np.ndarray[CTYPE_t, ndim=1] vec):
    """
    Sparse matrix, dense vector multiplication.  
    Here the vector is assumed to have two-dimensions.
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
        Returns dense, two-dimensional, array.
    
    """
    cdef Py_ssize_t row
    cdef int jj,row_start,row_end
    cdef int num_rows=len(vec)
    cdef CTYPE_t dot
    cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)
    for row in range(num_rows):
        dot=0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            dot+=data[jj]*vec[idx[jj]]
        out[row,0]=dot
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=2] cy_ode_rhs(double t, np.ndarray[CTYPE_t, ndim=1] rho, np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr):
    cdef int row, jj, row_start, row_end
    cdef int num_rows=len(rho)
    cdef CTYPE_t dot
    cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)
    for row from 0 <= row < num_rows:
        dot = 0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            dot = dot + data[jj] * rho[idx[jj]]
        out[row,0] = dot
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_ode_psi_func_td(double t, np.ndarray[CTYPE_t, ndim=1] psi, object H_func, object args):
    H = H_func(t, args)
    return -1j * spmv1d(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_ode_psi_func_td_with_state(double t, np.ndarray[CTYPE_t, ndim=1] psi, object H_func, object args):
    H = H_func(t, psi, args)
    return -1j * spmv1d(H.data, H.indices, H.indptr, psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=2] spmv_dia(np.ndarray[CTYPE_t, ndim=2] data, np.ndarray[int] offsets,int num_rows,int num_diags, np.ndarray[CTYPE_t, ndim=2] vec, np.ndarray[CTYPE_t, ndim=2] ret,int N):
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
            dot+=data[ii,jj-i0]*vec[jj-i0,0]
        ret[jj,0]=dot
    return ret



@cython.boundscheck(False)
@cython.wraparound(False)
def cy_expect(np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr,int isherm, np.ndarray[CTYPE_t, ndim=2] state):
    cdef np.ndarray[CTYPE_t, ndim=2] y = spmv(data,idx,ptr,state)
    cdef np.ndarray[CTYPE_t, ndim=2] x = state.conj().transpose()
    cdef int num_rows = len(state)
    cdef CTYPE_t dot = 0.0j
    for row in range(num_rows):
        dot+=x[0,row]*y[row,0]
    if isherm:
        return np.real(dot)
    else:
        return dot
















