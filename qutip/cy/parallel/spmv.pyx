import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t

CTYPE = np.int64
ctypedef np.int64_t LTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] parallel_spmv_csr(
        np.ndarray[CTYPE_t, ndim=1, mode="c"] data,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] idx,
        np.ndarray[ITYPE_t, ndim=1, mode="c"] ptr,
        np.ndarray[CTYPE_t, ndim=1, mode="c"] vec,
        int num_threads):
        
    cdef Py_ssize_t row
    cdef int jj,row_start,row_end
    cdef int num_rows = ptr.shape[0]-1
    cdef np.ndarray[CTYPE_t, ndim=1, mode="c"] out = np.zeros((num_rows), dtype=np.complex)
    for row in prange(num_rows, nogil=True, num_threads=num_threads):
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj in range(row_start,row_end):
            out[row] = out[row]+data[jj]*vec[idx[jj]]
    return out