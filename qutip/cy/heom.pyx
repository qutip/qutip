#cython: language_level=3

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
