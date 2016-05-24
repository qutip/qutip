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
from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] parallel_spmv_csr(
        np.ndarray[CTYPE_t, ndim=1, mode="c"] data,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] idx,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] ptr,
        np.ndarray[CTYPE_t, ndim=1, mode="c"] vec,
        int num_threads):
        
    cdef Py_ssize_t row
    cdef int jj,row_start,row_end
    cdef int num_rows = ptr.shape[0]-1
    cdef np.ndarray[CTYPE_t, ndim=1, mode="c"] out = np.zeros((num_rows), dtype=np.complex)
    
    for row in prange(num_rows, nogil=True, num_threads=num_threads):
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            out[row] = out[row] + data[jj]*vec[idx[jj]]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] parallel_spmvpy(
        np.ndarray[CTYPE_t, ndim=1, mode="c"] data,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] idx,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] ptr,
        np.ndarray[CTYPE_t, ndim=1, mode="c"] vec,
        CTYPE_t a, 
        np.ndarray[CTYPE_t, ndim=1, mode="c"] out,
        int num_threads):
    """
    Sparse matrix time vector plus vector function:
    out = out + a * (data, idx, ptr) * vec
    """
    cdef Py_ssize_t row
    cdef int jj, row_start, row_end
    cdef int num_rows = vec.shape[0]
    cdef CTYPE_t dot

    for row in prange(num_rows, nogil=True, num_threads=num_threads):
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            out[row] = out[row] + a*data[jj]*vec[idx[jj]]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] parallel_ode_rhs(
        double t, 
        np.ndarray[CTYPE_t, ndim=1, mode="c"] vec,
        np.ndarray[CTYPE_t, ndim=1, mode="c"] data,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] idx,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] ptr,
        int num_threads):

    cdef int row, jj, row_start, row_end
    cdef int num_rows = ptr.shape[0]-1
    cdef CTYPE_t dot
    cdef np.ndarray[CTYPE_t, ndim=1, mode="c"] out = \
        np.zeros((num_rows), dtype=np.complex)

    for row in prange(num_rows, nogil=True, num_threads=num_threads):
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            out[row] = out[row] + data[jj]*vec[idx[jj]]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] parallel_ode_psi_func_td(
        double t, 
        np.ndarray[CTYPE_t, ndim=1, mode="c"] psi, 
        object H_func,
        object args,
        int num_threads):

    H = H_func(t, args)
    return -1j * parallel_spmv_csr(H.data, H.indices, H.indptr, psi, num_threads)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] parallel_ode_psi_func_td_with_state(
        double t,
        np.ndarray[CTYPE_t, ndim=1, mode="c"] psi,
        object H_func,
        object args,
        int num_threads):

    H = H_func(t, psi, args)
    return -1j * parallel_spmv_csr(H.data, H.indices, H.indptr, psi, num_threads)