import numpy as np
cimport numpy as np
cimport cython
#from cython.parallel cimport prange

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def spmv_csr_serial(np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr,np.ndarray[DTYPE_t, ndim=2] vec):
    cdef Py_ssize_t row
    cdef int jj,row_start,row_end
    cdef int num_rows=len(vec)
    cdef complex dot
    cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)
    for row in range(num_rows):
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            dot+=data[jj]*vec[idx[jj],0]
        out[row,0]=dot
    return out


#@cython.boundscheck(False)
#@cython.wraparound(False)
#def spmv_csr_parallel(np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr,np.ndarray[DTYPE_t, ndim=2] vec):
    #cdef Py_ssize_t row
    #cdef int jj,row_start,row_end,num_rows=len(vec)
    #cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)
    #for row in prange(num_rows,nogil=True):
        #row_start = ptr[row]
        #row_end = ptr[row+1]
        #for jj in range(row_start,row_end):
            #out[row,0]=out[row,0]+data[jj]*vec[idx[jj],0]
    #return out

