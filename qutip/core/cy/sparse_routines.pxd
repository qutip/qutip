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

from .sparse_structs cimport CSR_Matrix, COO_Matrix

cdef void raise_error_CSR(int E, CSR_Matrix * C=*)
cdef void raise_error_COO(int E, COO_Matrix * C=*)

cdef void init_CSR(
    CSR_Matrix * mat,
    int nnz,
    int nrows,
    int ncols=*,
    int max_length=*,
    int init_zeros=*,
)
cdef void copy_CSR(CSR_Matrix * out, CSR_Matrix * mat)
cdef void init_COO(
    COO_Matrix * mat,
    int nnz,
    int nrows,
    int ncols=*,
    int max_length=*,
    int init_zeros=*,
)

cdef void free_CSR(CSR_Matrix * mat)
cdef void free_COO(COO_Matrix * mat)
cdef void shorten_CSR(CSR_Matrix * mat, int N)
cdef void expand_CSR(CSR_Matrix * mat, int init_zeros=*)

cdef void sort_indices(CSR_Matrix * mat)

cdef void COO_to_CSR(CSR_Matrix * out, COO_Matrix * mat)
cdef void CSR_to_COO(COO_Matrix * out, CSR_Matrix * mat)
cdef void COO_to_CSR_inplace(CSR_Matrix * out, COO_Matrix * mat)

cdef CSR_Matrix CSR_from_scipy(object A)
cdef void CSR_from_scipy_inplace(object A, CSR_Matrix * mat)
cdef COO_Matrix COO_from_scipy(object A)
cdef void identity_CSR(CSR_Matrix * mat, unsigned int nrows)
