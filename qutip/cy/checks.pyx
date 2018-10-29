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
import scipy.sparse as sp
from qutip.fastsparse import fast_csr_matrix
cimport numpy as cnp
cimport cython

include "sparse_routines.pxi"


def _test_coo2csr_struct(object A):
    cdef COO_Matrix mat = COO_from_scipy(A)
    cdef CSR_Matrix out
    COO_to_CSR(&out, &mat)
    return CSR_to_scipy(&out)


def _test_sorting(object A):
    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]

    cdef CSR_Matrix out

    out.data = &data[0]
    out.indices = &ind[0]
    out.indptr = &ptr[0]
    out.nrows = nrows
    out.ncols = ncols
    out.is_set = 1
    out.numpy_lock = 0
    sort_indices(&out)


def _test_coo2csr_inplace_struct(object A, int sorted = 0):
    cdef complex[::1] data = A.data
    cdef int[::1] rows = A.row
    cdef int[::1] cols = A.col
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]
    cdef int nnz = data.shape[0]
    cdef size_t kk
    #We need to make copies here to test the inplace conversion
    #as we cannot use numpy data due to ownership issues.
    cdef complex * _data = <complex *>PyDataMem_NEW(nnz * sizeof(complex))
    cdef int * _rows = <int *>PyDataMem_NEW(nnz * sizeof(int))
    cdef int * _cols = <int *>PyDataMem_NEW(nnz * sizeof(int))
    for kk in range(nnz):
        _data[kk] = data[kk]
        _rows[kk] = rows[kk]
        _cols[kk] = cols[kk]

    cdef COO_Matrix mat
    mat.data = _data
    mat.rows = _rows
    mat.cols = _cols
    mat.nrows = nrows
    mat.ncols = ncols
    mat.nnz = nnz
    mat.max_length = mat.nnz
    mat.is_set = 1
    mat.numpy_lock = 0

    cdef CSR_Matrix out

    COO_to_CSR_inplace(&out, &mat)
    if sorted:
        sort_indices(&out)
    return CSR_to_scipy(&out)


def _test_csr2coo_struct(object A):
    cdef CSR_Matrix mat = CSR_from_scipy(A)
    cdef COO_Matrix out
    CSR_to_COO(&out, &mat)
    return COO_to_scipy(&out)
