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

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_bandwidth(np.ndarray[int] idx, np.ndarray[int] ptr, int nrows):
    """
    Calculates the max (mb), lower(lb), and upper(ub) bandwidths of a csr_matrix.
    """
    cdef int lb, ub, mb, ii, jj, ldist
    lb=-nrows
    ub=-nrows
    mb=0
    for ii in range(nrows):
        for jj in range(ptr[ii],ptr[ii+1]):
            ldist=ii-idx[jj]
            lb=max(lb,ldist)
            ub=max(ub,-ldist)
            mb=max(mb,ub+lb+1)
    return mb, lb, ub


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_permute_complex(np.ndarray[CTYPE_t] data, np.ndarray[int] idx, np.ndarray[int] ptr, int nrows,
                    int ncols, np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm, int flag):
    """
    Permutes the rows and columns of a sparse CSR or CSC matrix according to the permutation
    arrays rperm and cperm, respectively.  Here, the permutation arrays specify the 
    new order of the rows and columns. i.e. [0,1,2,3,4] -> [3,0,4,1,2].
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[CTYPE_t, ndim=1] new_data = np.zeros(len(data),dtype=np.complex)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)
    cdef np.ndarray[np.intp_t] perm_r
    cdef np.ndarray[np.intp_t] perm_c
    cdef np.ndarray[np.intp_t] inds
    if flag==0: #for CSR matricies
        if len(rperm)!=0:
            inds=np.argsort(rperm)
            perm_r=np.arange(len(rperm))[inds]
    
            for jj in range(nrows):
               ii=perm_r[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]
    
            for jj in range(nrows): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]
    
            for jj in range(nrows): 
                k0=new_ptr[perm_r[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(cperm)!=0:
            inds =np.argsort(cperm)
            perm_c=np.arange(len(cperm))[inds]
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=perm_c[new_idx[jj]]
    
    elif flag==1: #for CSC matricies
        if len(cperm)!=0:
            inds=np.argsort(cperm)
            perm_c=np.arange(len(cperm))[inds]
    
            for jj in range(ncols):
               ii=perm_c[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]
    
            for jj in range(ncols): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]
    
            for jj in range(ncols): 
                k0=new_ptr[perm_c[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(rperm)!=0:
            inds =np.argsort(rperm)
            perm_r=np.arange(len(rperm))[inds]
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=perm_r[new_idx[jj]]
    
    return new_data, new_idx, new_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_permute_float(np.ndarray[DTYPE_t] data, np.ndarray[int] idx, np.ndarray[int] ptr, int nrows,
                    int ncols, np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm, int flag):
    """
    Permutes the rows and columns of a sparse CSR or CSC matrix according to the permutation
    arrays rperm and cperm, respectively.  Here, the permutation arrays specify the 
    new order of the rows and columns. i.e. [0,1,2,3,4] -> [3,0,4,1,2].
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[DTYPE_t, ndim=1] new_data = np.zeros(len(data),dtype=float)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)
    cdef np.ndarray[np.intp_t] perm_r
    cdef np.ndarray[np.intp_t] perm_c
    cdef np.ndarray[np.intp_t] inds
    if flag==0: #for CSR matricies
        if len(rperm)!=0:
            inds=np.argsort(rperm)
            perm_r=np.arange(len(rperm))[inds]

            for jj in range(nrows):
               ii=perm_r[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(nrows): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(nrows): 
                k0=new_ptr[perm_r[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(cperm)!=0:
            inds =np.argsort(cperm)
            perm_c=np.arange(len(cperm))[inds]
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=perm_c[new_idx[jj]]

    elif flag==1: #for CSC matricies
        if len(cperm)!=0:
            inds=np.argsort(cperm)
            perm_c=np.arange(len(cperm))[inds]

            for jj in range(ncols):
               ii=perm_c[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(ncols): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(ncols): 
                k0=new_ptr[perm_c[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(rperm)!=0:
            inds =np.argsort(rperm)
            perm_r=np.arange(len(rperm))[inds]
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=perm_r[new_idx[jj]]
    return new_data, new_idx, new_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_permute_int(np.ndarray[np.intp_t] data, np.ndarray[int] idx, np.ndarray[int] ptr, int nrows,
                    int ncols, np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm, int flag):
    """
    Permutes the rows and columns of a sparse CSR or CSC matrix according to the permutation
    arrays rperm and cperm, respectively.  Here, the permutation arrays specify the 
    new order of the rows and columns. i.e. [0,1,2,3,4] -> [3,0,4,1,2].
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[np.intp_t, ndim=1] new_data = np.zeros(len(data),dtype=int)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)
    cdef np.ndarray[np.intp_t] perm_r
    cdef np.ndarray[np.intp_t] perm_c
    cdef np.ndarray[np.intp_t] inds
    if flag==0: #for CSR matricies
        if len(rperm)!=0:
            inds=np.argsort(rperm)
            perm_r=np.arange(len(rperm))[inds]

            for jj in range(nrows):
               ii=perm_r[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(nrows): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(nrows): 
                k0=new_ptr[perm_r[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(cperm)!=0:
            inds =np.argsort(cperm)
            perm_c=np.arange(len(cperm))[inds]
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=perm_c[new_idx[jj]]

    elif flag==1: #for CSC matricies
        if len(cperm)!=0:
            inds=np.argsort(cperm)
            perm_c=np.arange(len(cperm))[inds]

            for jj in range(ncols):
               ii=perm_c[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(ncols): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(ncols): 
                k0=new_ptr[perm_c[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(rperm)!=0:
            inds =np.argsort(rperm)
            perm_r=np.arange(len(rperm))[inds]
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=perm_r[new_idx[jj]]
    return new_data, new_idx, new_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_reverse_permute_complex(np.ndarray[CTYPE_t] data, np.ndarray[int] idx, 
                            np.ndarray[int] ptr, int nrows, int ncols, 
                            np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm, int flag):
    """
    Reverse permutes the rows and columns of a sparse CSR or CSC matrix according to the 
    original permutation arrays rperm and cperm, respectively.
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[CTYPE_t, ndim=1] new_data = np.zeros(len(data),dtype=np.complex)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)
    if flag==0: #CSR matrix
        if len(rperm)!=0:
            for jj in range(nrows):
               ii=rperm[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(nrows): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(nrows): 
                k0=new_ptr[rperm[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(cperm)!=0:
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=cperm[new_idx[jj]]
    if flag==1: #CSC matrix
        if len(cperm)!=0:
            for jj in range(ncols):
               ii=cperm[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(ncols): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(ncols): 
                k0=new_ptr[cperm[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(rperm)!=0:
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=rperm[new_idx[jj]]
    return new_data, new_idx, new_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_reverse_permute_float(np.ndarray[DTYPE_t] data, np.ndarray[int] idx, 
                            np.ndarray[int] ptr, int nrows, int ncols, 
                            np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm, int flag):
    """
    Reverse permutes the rows and columns of a sparse CSR or CSC matrix according to the 
    original permutation arrays rperm and cperm, respectively.
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[DTYPE_t, ndim=1] new_data = np.zeros(len(data),dtype=float)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)
    if flag==0: #CSR matrix
        if len(rperm)!=0:
            for jj in range(nrows):
               ii=rperm[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(nrows): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(nrows): 
                k0=new_ptr[rperm[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(cperm)!=0:
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=cperm[new_idx[jj]]
    if flag==1: #CSC matrix
        if len(cperm)!=0:
            for jj in range(ncols):
               ii=cperm[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(ncols): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(ncols): 
                k0=new_ptr[cperm[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(rperm)!=0:
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=rperm[new_idx[jj]]
    return new_data, new_idx, new_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_reverse_permute_int(np.ndarray[np.intp_t] data, np.ndarray[int] idx, 
                            np.ndarray[int] ptr, int nrows, int ncols, 
                            np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm, int flag):
    """
    Reverse permutes the rows and columns of a sparse CSR or CSC matrix according to the 
    original permutation arrays rperm and cperm, respectively.
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[np.intp_t, ndim=1] new_data = np.zeros(len(data),dtype=int)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)
    if flag==0: #CSR matrix
        if len(rperm)!=0:
            for jj in range(nrows):
               ii=rperm[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(nrows): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(nrows): 
                k0=new_ptr[rperm[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(cperm)!=0:
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=cperm[new_idx[jj]]
    if flag==1: #CSC matrix
        if len(cperm)!=0:
            for jj in range(ncols):
               ii=cperm[jj]
               new_ptr[ii+1]=ptr[jj+1]-ptr[jj]

            for jj in range(ncols): 
                new_ptr[jj+1]=new_ptr[jj+1]+new_ptr[jj]

            for jj in range(ncols): 
                k0=new_ptr[cperm[jj]]
                for kk in range(ptr[jj],ptr[jj+1]):
                    new_idx[k0]=idx[kk]
                    new_data[k0]=data[kk]
                    k0=k0+1
        if len(rperm)!=0:
            nnz=new_ptr[len(new_ptr)-1]
            for jj in range(nnz):
                new_idx[jj]=rperm[new_idx[jj]]
    return new_data, new_idx, new_ptr


