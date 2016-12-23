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
cimport numpy as np
cimport cython

cdef extern from "complex.h" nogil:
    double complex conj(double complex x)

include "sparse_struct.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def zcsr_add(complex[::1] dataA, int[::1] indsA, int[::1] indptrA,
             complex[::1] dataB, int[::1] indsB, int[::1] indptrB, 
             int nrows, int ncols,
             int Annz, int Bnnz,
             double complex alpha = 1):
    
    """
    Adds two sparse CSR matries. Like SciPy, we assume the worse case
    for the fill A.nnz + B.nnz.
    """
    cdef int worse_fill = Annz + Bnnz
    cdef int nnz
    #Both matrices are zero mats
    if Annz == 0 and Bnnz == 0:
        return fast_csr_matrix(([], [], []), shape=(nrows,ncols))
    #A is the zero matrix
    elif Annz == 0:
        return fast_csr_matrix((alpha*np.asarray(dataB), indsB, indptrB), 
                            shape=(nrows,ncols))
    #B is the zero matrix
    elif Bnnz == 0:
        return fast_csr_matrix((dataA, indsA, indptrA), 
                            shape=(nrows,ncols))
    # Out CSR_Matrix
    cdef CSR_Matrix out
    init_CSR(&out, worse_fill, nrows, worse_fill)
    out.ncols = ncols
    nnz = _zcsr_add_core(&dataA[0], &indsA[0], &indptrA[0], 
                     &dataB[0], &indsB[0], &indptrB[0],
                     alpha,
                     &out,       
                     nrows, ncols)
    #Shorten data and indices if needed
    if out.nnz > nnz:
        shorten_CSR(&out, nnz)
    return CSR_to_scipy(&out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _zcsr_add_core(double complex * Adata, int * Aind, int * Aptr, 
                     double complex * Bdata, int * Bind, int * Bptr,
                     double complex alpha,
                     CSR_Matrix * C,       
                     int nrows, int ncols) nogil:
    
    cdef int j1, j2, kc = 0 
    cdef int ka, kb, ka_max, kb_max
    cdef size_t ii
    cdef double complex tmp
    C.indptr[0] = 0
    if alpha != 1:
        for ii in range(nrows):
            ka = Aptr[ii]
            kb = Bptr[ii]
            ka_max = Aptr[ii+1]-1
            kb_max = Bptr[ii+1]-1
            while (ka <= ka_max) or (kb <= kb_max):
                if ka <= ka_max:
                    j1 = Aind[ka]
                else:
                    j1 = ncols+1

                if kb <= kb_max:
                    j2 = Bind[kb]
                else:
                    j2 = ncols+1

                if j1 == j2:
                    tmp = Adata[ka] + alpha*Bdata[kb]
                    if tmp != 0:
                        C.data[kc] = tmp
                        C.indices[kc] = j1
                        kc += 1
                    ka += 1
                    kb += 1   
                elif j1 < j2:
                    C.data[kc] = Adata[ka]
                    C.indices[kc] = j1
                    ka += 1
                    kc += 1
                elif j1 > j2:
                    C.data[kc] = alpha*Bdata[kb]
                    C.indices[kc] = j2
                    kb += 1
                    kc += 1

            C.indptr[ii+1] = kc
    else:
        for ii in range(nrows):
            ka = Aptr[ii]
            kb = Bptr[ii]
            ka_max = Aptr[ii+1]-1
            kb_max = Bptr[ii+1]-1
            while (ka <= ka_max) or (kb <= kb_max):
                if ka <= ka_max:
                    j1 = Aind[ka]
                else:
                    j1 = ncols+1

                if kb <= kb_max:
                    j2 = Bind[kb]
                else:
                    j2 = ncols+1

                if j1 == j2:
                    tmp = Adata[ka] + Bdata[kb]
                    if tmp != 0:
                        C.data[kc] = tmp
                        C.indices[kc] = j1
                        kc += 1
                    ka += 1
                    kb += 1   
                elif j1 < j2:
                    C.data[kc] = Adata[ka]
                    C.indices[kc] = j1
                    ka += 1
                    kc += 1
                elif j1 > j2:
                    C.data[kc] = Bdata[kb]
                    C.indices[kc] = j2
                    kb += 1
                    kc += 1

            C.indptr[ii+1] = kc
    return kc



@cython.boundscheck(False)
@cython.wraparound(False)
def zcsr_mult(complex [::1] dataA, int[::1] indsA, int[::1] indptrA,
             complex [::1] dataB, int[::1] indsB, int[::1] indptrB, 
             int nrows, int ncols,
             int Annz, int Bnnz):
    #Both matrices are zero mats
    if Annz == 0 or Bnnz == 0:
        return fast_csr_matrix(([], [], []), shape=(nrows,ncols))
    
    cdef int nnz
    cdef CSR_Matrix out
    
    nnz = _zcsr_mult_pass1(&dataA[0], &indsA[0], &indptrA[0], 
                     &dataB[0], &indsB[0], &indptrB[0],  
                     nrows, ncols)
    
    init_CSR(&out, nnz, nrows, 0)
    _zcsr_mult_pass2(&dataA[0], &indsA[0], &indptrA[0], 
                     &dataB[0], &indsB[0], &indptrB[0],
                     &out,       
                     nrows, ncols)
    return CSR_to_scipy(&out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _zcsr_mult_pass1(double complex * Adata, int * Aind, int * Aptr, 
                     double complex * Bdata, int * Bind, int * Bptr,       
                     int nrows, int ncols) nogil:

    cdef int j, k, row_nnz, nnz = 0
    cdef size_t ii,jj,kk
    #Setup mask array
    cdef int * mask = <int *>PyDataMem_NEW_ZEROED(ncols, sizeof(int))
    for ii in range(ncols):
        mask[ii] = -1
    #Pass 1
    for ii in range(nrows):
        row_nnz = 0
        for jj in range(Aptr[ii], Aptr[ii+1]):
            j = Aind[jj]
            for kk in range(Bptr[j], Bptr[j+1]):
                k = Bind[kk]
                if mask[k] != ii:
                    mask[k] = ii
                    row_nnz += 1
        nnz += row_nnz
    PyDataMem_FREE(mask)
    return nnz


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _zcsr_mult_pass2(double complex * Adata, int * Aind, int * Aptr, 
                     double complex * Bdata, int * Bind, int * Bptr,
                     CSR_Matrix * C,
                     int nrows, int ncols) nogil:  

    cdef int head, length, temp, j, k, nnz = 0
    cdef size_t ii,jj,kk
    cdef double complex val
    cdef double complex * sums = <double complex *>PyDataMem_NEW_ZEROED(ncols, sizeof(double complex))
    cdef int * nxt = <int *>PyDataMem_NEW_ZEROED(ncols, sizeof(int))
    for ii in range(ncols):
        nxt[ii] = -1

    C.indptr[0] = 0
    for ii in range(nrows):
        head = -2
        length = 0
        for jj in range(Aptr[ii], Aptr[ii+1]):
            j = Aind[jj]
            val = Adata[jj]
            for kk in range(Bptr[j], Bptr[j+1]):
                k = Bind[kk]
                sums[k] += val*Bdata[kk]
                if nxt[k] == -1:
                    nxt[k] = head
                    head = k
                    length += 1

        for jj in range(length):
            if sums[head] != 0:
                C.indices[nnz] = head
                C.data[nnz] = sums[head]
                nnz += 1
            temp = head
            head = nxt[head]
            nxt[temp] = -1
            sums[temp] = 0
    
        C.indptr[ii+1] = nnz

    #Free temp arrays
    PyDataMem_FREE(sums)
    PyDataMem_FREE(nxt)


@cython.boundscheck(False)
@cython.wraparound(False)
def zcsr_kron(object A, object B):
    """
    Computes the kronecker product between two complex
    sparse matrices in CSR format.
    """
    cdef complex[::1] dataA = A.data
    cdef int[::1] indsA = A.indices
    cdef int[::1] indptrA = A.indptr
    cdef int rowsA = A.shape[0]
    cdef int colsA = A.shape[1]

    cdef complex[::1] dataB = B.data
    cdef int[::1] indsB = B.indices
    cdef int[::1] indptrB = B.indptr
    cdef int rowsB = B.shape[0]
    cdef int colsB = B.shape[1]

    cdef int out_nnz = dataA.shape[0] * dataB.shape[0]
    cdef int rows_out = rowsA * rowsB
    cdef int cols_out = colsA * colsB

    cdef CSR_Matrix out
    init_CSR(&out, out_nnz, rows_out)
    out.ncols = cols_out

    _zcsr_kron_core(&dataA[0], &indsA[0], &indptrA[0], 
                    &dataB[0], &indsB[0], &indptrB[0],
                    &out, 
                    rowsA, rowsB, colsB)
    return CSR_to_scipy(&out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _zcsr_kron_core(double complex * dataA, int * indsA, int * indptrA, 
                     double complex * dataB, int * indsB, int * indptrB,
                     CSR_Matrix * out,       
                     int rowsA, int rowsB, int colsB) nogil:
    cdef size_t ii, jj, ptrA, ptr
    cdef int row = 0
    cdef int ptr_start, ptr_end
    cdef int row_startA, row_endA, row_startB, row_endB, distA, distB, ptrB

    for ii in range(rowsA):
        row_startA = indptrA[ii]
        row_endA = indptrA[ii+1]
        distA = row_endA - row_startA

        for jj in range(rowsB):
            row_startB = indptrB[jj]
            row_endB = indptrB[jj+1]
            distB = row_endB - row_startB

            ptr_start = out.indptr[row]
            ptr_end = ptr_start + distB

            out.indptr[row+1] = out.indptr[row] + distA * distB
            row += 1

            for ptrA in range(row_startA, row_endA):
                ptrB = row_startB
                for ptr in range(ptr_start, ptr_end):
                    out.indices[ptr] = indsA[ptrA] * colsB + indsB[ptrB]
                    out.data[ptr] = dataA[ptrA] * dataB[ptrB]
                    ptrB += 1

                ptr_start += distB
                ptr_end += distB


@cython.boundscheck(False)
@cython.wraparound(False)
def zcsr_transpose(object A):
    """
    Transpose of a sparse matrix in CSR format.
    """
    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]

    cdef CSR_Matrix out
    init_CSR(&out, data.shape[0], ncols)
    out.ncols = nrows

    _zcsr_trans_core(&data[0], &ind[0], &ptr[0], 
                    &out, nrows, ncols)
    return CSR_to_scipy(&out)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _zcsr_trans_core(double complex * data, int * ind, int * ptr, 
                     CSR_Matrix * out,       
                     int nrows, int ncols) nogil:
    
    cdef int k, nxt
    cdef size_t ii, jj
    
    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii+1]):
            k = ind[jj] + 1
            out.indptr[k] += 1
    
    for ii in range(ncols):
        out.indptr[ii+1] += out.indptr[ii]
    
    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii+1]):
            k = ind[jj]
            nxt = out.indptr[k]
            out.data[nxt] = data[jj]
            out.indices[nxt] = ii
            out.indptr[k] = nxt + 1
    
    for ii in range(ncols,0,-1):
        out.indptr[ii] = out.indptr[ii-1]
    
    out.indptr[0] = 0



@cython.boundscheck(False)
@cython.wraparound(False)
def zcsr_adjoint(object A):
    """
    Adjoint of a sparse matrix in CSR format.
    """
    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]

    cdef CSR_Matrix out
    init_CSR(&out, data.shape[0], ncols)
    out.ncols = nrows

    _zcsr_adjoint_core(&data[0], &ind[0], &ptr[0], 
                        &out, nrows, ncols)
    return CSR_to_scipy(&out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _zcsr_adjoint_core(double complex * data, int * ind, int * ptr, 
                     CSR_Matrix * out,       
                     int nrows, int ncols) nogil:

    cdef int k, nxt
    cdef size_t ii, jj

    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii+1]):
            k = ind[jj] + 1
            out.indptr[k] += 1

    for ii in range(ncols):
        out.indptr[ii+1] += out.indptr[ii]

    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii+1]):
            k = ind[jj]
            nxt = out.indptr[k]
            out.data[nxt] = conj(data[jj])
            out.indices[nxt] = ii
            out.indptr[k] = nxt + 1

    for ii in range(ncols,0,-1):
        out.indptr[ii] = out.indptr[ii-1]

    out.indptr[0] = 0