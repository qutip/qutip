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

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_pad_csr(object A, int row_scale, int col_scale, int insertrow=0, int insertcol=0):
    cdef int nrowin = A.shape[0]
    cdef int ncolin = A.shape[1]
    cdef int nnz = A.indptr[nrowin]
    cdef int nrowout = nrowin*row_scale
    cdef int ncolout = ncolin*col_scale
    cdef size_t kk
    cdef int temp, temp2
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr_in = A.indptr
    cdef cnp.ndarray[int, ndim=1, mode='c'] ptr_out = np.zeros(nrowout+1,dtype=np.int32)

    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        temp = insertcol*ncolin
        for kk in range(nnz):
            ind[kk] += temp
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")


    if insertrow == 0:
        temp = ptr_in[nrowin]
        for kk in range(nrowin):
            ptr_out[kk] = ptr_in[kk]
        for kk in range(nrowin, nrowout+1):
            ptr_out[kk] = temp

    elif insertrow == row_scale-1:
        temp = (row_scale - 1) * nrowin
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = ptr_in[kk-temp]

    elif insertrow > 0 and insertrow < row_scale - 1:
        temp = insertrow*nrowin
        for kk in range(temp, temp+nrowin):
            ptr_out[kk] = ptr_in[kk-temp]
        temp = kk+1
        temp2 = ptr_in[nrowin]
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = temp2
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    A.indptr = ptr_out

    return A
