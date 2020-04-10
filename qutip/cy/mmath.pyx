# cython: language_level=3
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

from qutip.cy.spmatfuncs cimport spmvpy as spmvpy_
from qutip.cy.spmatfuncs cimport _spmm_c_py, _spmm_f_py

# Link the omp version if available at compilation if available.
# Allows the same code to be used for both serial and openmp.

IF HAVE_OPENMP:
    from qutip.cy.openmp.parfuncs cimport (spmvpy_openmp, spmmcpy_par,
                                           spmmfpy_omp)

    cdef void spmvpy(complex * data, int * ind, int *  ptr,
                     complex * vec, complex a, complex * out,
                     unsigned int nrows, unsigned int nthr):
        if nthr > 1:
            spmvpy_openmp(data, ind, ptr, vec, a, out, nrows, nthr)
        else:
            spmvpy_(data, ind, ptr, vec, a, out, nrows)

    cdef void spmmcpy(complex* data, int* ind, int* ptr,
                      complex* mat, complex a, complex* out,
                      int sp_rows, unsigned int nrows,
                      unsigned int ncols, int nthr):
        if nthr > 1:
            spmmcpy_par(data, ind, ptr, mat, a, out,
                        sp_rows, nrows, ncols, nthr)
        else:
            _spmm_c_py(data, ind, ptr, mat, a, out,
                       sp_rows, nrows, ncols)

    cdef void spmmfpy(complex* data, int* ind, int* ptr,
                      complex* mat, complex a, complex* out,
                      int sp_rows, unsigned int nrows,
                      unsigned int ncols, int nthr):
        if nthr > 1:
            spmmfpy_omp(data, ind, ptr, mat, a, out,
                        sp_rows, nrows, ncols, nthr)
        else:
            _spmm_f_py(data, ind, ptr, mat, a, out,
                       sp_rows, nrows, ncols)

ELSE:
    cdef void spmvpy(complex * data, int * ind, int *  ptr,
                     complex * vec, complex a, complex * out,
                     unsigned int nrows, unsigned int nthr):
        spmvpy_(data, ind, ptr, vec, a, out, nrows)

    cdef void spmmcpy(complex* data, int* ind, int* ptr,
                      complex* mat, complex a, complex* out,
                      int sp_rows, unsigned int nrows,
                      unsigned int ncols, int nthr):
        _spmm_c_py(data, ind, ptr, mat, a, out, sp_rows, nrows, ncols)

    cdef void spmmfpy(complex* data, int* ind, int* ptr,
                      complex* mat, complex a, complex* out,
                      int sp_rows, unsigned int nrows,
                      unsigned int ncols, int nthr):
        _spmm_f_py(data, ind, ptr, mat, a, out, sp_rows, nrows, ncols)
