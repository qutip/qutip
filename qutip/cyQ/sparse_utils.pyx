import numpy as np
cimport numpy as np
cimport cython

ctypedef np.complex128_t CTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_permute(np.ndarray[CTYPE_t] data, np.ndarray[int] idx, np.ndarray[int] ptr, int nrows, 
                    np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm):
    """
    Permutes the rows and columns of a sparse CSR matrix according to the permutation
    arrays rperm and cperm, respectively.  Here, the permutation arrays specify the 
    new order of the rows and columns. i.e. [0,1,2,3,4] -> [3,0,4,1,2]
    """
    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[CTYPE_t, ndim=1] new_data = np.zeros(len(data),dtype=np.complex)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)
    cdef np.ndarray[np.intp_t] perm_r
    cdef np.ndarray[np.intp_t] perm_c
    cdef np.ndarray[np.intp_t] inds
    
    
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
        nnz=new_ptr[nrows]
        
        for jj in range(nnz):
            new_idx[jj]=perm_c[new_idx[jj]]
    
    return new_data, new_idx, new_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
def _sparse_reverse_permute(np.ndarray[CTYPE_t] data, np.ndarray[int] idx, np.ndarray[int] ptr, int nrows, 
                    np.ndarray[np.intp_t] rperm, np.ndarray[np.intp_t] cperm):

    cdef int ii, jj, kk, k0, nnz
    cdef np.ndarray[CTYPE_t, ndim=1] new_data = np.zeros(len(data),dtype=np.complex)
    cdef np.ndarray[np.intp_t] new_idx = np.zeros(len(idx),dtype=int)
    cdef np.ndarray[np.intp_t] new_ptr = np.zeros(len(ptr),dtype=int)



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
        nnz=new_ptr[nrows]
        for jj in range(nnz):
            new_idx[jj]=cperm[new_idx[jj]]
    
        return new_data, new_idx, new_ptr






