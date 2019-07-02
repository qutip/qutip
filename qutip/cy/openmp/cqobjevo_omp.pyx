#!python
#cython: language_level=3
# distutils: language = c++
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
"""
Parallel version of the cqobjevo's
See ../cqobjevo.pyx for more details
"""
from qutip.cy.cqobjevo cimport CQobjCte, CQobjEvoTd, CQobjEvoTdMatched
from qutip.cy.openmp.parfuncs cimport spmvpy_openmp
import numpy as np
import scipy.sparse as sp
cimport numpy as np
import cython
cimport cython
from cython.parallel import prange

include "../complex_math.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _spmmcpy_par(complex* data, int* ind, int* ptr, complex* mat,
                      complex a, complex* out, int sp_rows,
                      unsigned int nrows, unsigned int ncols, int nthr):
    """
    sparse*dense "C" ordered.
    """
    cdef int row, col, ii, jj, row_start, row_end
    for row in prange(sp_rows, nogil=True, num_threads=nthr):
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            for col in range(ncols):
                out[row * ncols + col] += a*data[jj]*mat[ind[jj] * ncols + col]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _spmmfpy_omp(complex* data, int* ind, int* ptr, complex* mat,
                  complex a, complex* out, unsigned int sp_rows,
                  unsigned int nrows, unsigned int ncols, int nthr):
    """
    sparse*dense "F" ordered.
    """
    cdef int col
    for col in range(ncols):
        spmvpy_openmp(data, ind, ptr, mat+nrows*col, a,
                      out+sp_rows*col, sp_rows, nthr)


cdef class CQobjCteOmp(CQobjCte):
    cdef int nthr

    def set_threads(self, nthr):
        self.nthr = nthr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        spmvpy_openmp(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1.,
               out, self.shape0, self.nthr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec):
        cdef complex[::1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy_openmp(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1.,
               &y[0], self.shape0, self.nthr)
        cdef int row
        cdef complex dot = 0
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row])*y[row]
        return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        _spmmfpy_omp(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol, self.nthr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        _spmmcpy_par(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol, self.nthr)


cdef class CQobjEvoTdOmp(CQobjEvoTd):
    cdef int nthr

    def set_threads(self, nthr):
        self.nthr = nthr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        cdef int[2] shape
        shape[0] = self.shape1
        shape[1] = 1
        self._factor_dyn(t, vec, shape)
        cdef int i
        spmvpy_openmp(self.cte.data, self.cte.indices, self.cte.indptr, vec,
               1., out, self.shape0, self.nthr)
        for i in range(self.num_ops):
            spmvpy_openmp(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                   vec, self.coeff_ptr[i], out, self.shape0, self.nthr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        cdef int i
        _spmmfpy_omp(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol, self.nthr)
        for i in range(self.num_ops):
             _spmmfpy_omp(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                 mat, self.coeff_ptr[i], out, self.shape0, nrow, ncol, self.nthr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        cdef int i
        _spmmcpy_par(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol, self.nthr)
        for i in range(self.num_ops):
             _spmmcpy_par(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                 mat, self.coeff_ptr[i], out, self.shape0, nrow, ncol, self.nthr)


cdef class CQobjEvoTdMatchedOmp(CQobjEvoTdMatched):
    cdef int nthr

    def set_threads(self, nthr):
        self.nthr = nthr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        cdef int[2] shape
        shape[0] = self.shape1
        shape[1] = 1
        self._factor_dyn(t, vec, shape)
        self._call_core(self.data_t, self.coeff_ptr)
        spmvpy_openmp(self.data_ptr, &self.indices[0], &self.indptr[0], vec,
               1., out, self.shape0, self.nthr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        self._call_core(self.data_t, self.coeff_ptr)
        _spmmfpy_omp(self.data_ptr, &self.indices[0], &self.indptr[0], mat, 1.,
               out, self.shape0, nrow, ncol, self.nthr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        self._call_core(self.data_t, self.coeff_ptr)
        _spmmcpy_par(self.data_ptr, &self.indices[0], &self.indptr[0], mat, 1.,
               out, self.shape0, nrow, ncol, self.nthr)
