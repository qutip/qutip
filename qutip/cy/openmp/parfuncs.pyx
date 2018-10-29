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

cdef extern from "src/zspmv_openmp.hpp" nogil:
    void zspmvpy_openmp(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows, int nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv_openmp(
        object super_op,
        complex[::1] vec,
        unsigned int nthr):
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
    return spmv_csr_openmp(super_op.data, super_op.indices, super_op.indptr, vec, nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv_csr_openmp(complex[::1] data,
            int[::1] ind, int[::1] ptr, complex[::1] vec, unsigned int nthr):
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
    cdef unsigned int num_rows = vec.shape[0]
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros(num_rows, dtype=complex)
    zspmvpy_openmp(&data[0], &ind[0], &ptr[0], &vec[0], 1.0, &out[0], num_rows, nthr)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void spmvpy_openmp(complex * data, int * ind, int * ptr,
            complex * vec,
            complex a,
            complex * out,
            unsigned int nrows,
            unsigned int nthr):

    zspmvpy_openmp(data, ind, ptr, vec, a, out, nrows, nthr)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs_openmp(
        double t,
        complex[::1] rho,
        complex[::1] data,
        int[::1] ind,
        int[::1] ptr,
        unsigned int nthr):

    cdef unsigned int nrows = rho.shape[0]
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = \
        np.zeros((nrows), dtype=complex)
    zspmvpy_openmp(&data[0], &ind[0], &ptr[0], &rho[0], 1.0, &out[0], nrows, nthr)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td_openmp(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args,
        unsigned int nthr):

    H = H_func(t, args).data
    return -1j * spmv_csr_openmp(H.data, H.indices, H.indptr, psi, nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td_with_state_openmp(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args,
        unsigned int nthr):

    H = H_func(t, psi, args)
    return -1j * spmv_csr_openmp(H.data, H.indices, H.indptr, psi, nthr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rho_func_td_openmp(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] rho,
        object L0,
        object L_func,
        object args,
        unsigned int nthr):
    cdef object L
    L = L0 + L_func(t, args).data
    return spmv_csr_openmp(L.data, L.indices, L.indptr, rho, nthr)
