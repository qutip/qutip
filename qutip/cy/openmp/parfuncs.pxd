#cython: language_level=3

cimport numpy as cnp
cimport cython

cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv_csr_openmp(complex[::1] data,
                int[::1] ind, int[::1] ptr, complex[::1] vec, unsigned int nthr)


cdef void spmvpy_openmp(complex * data,
                int * ind,
                int *  ptr,
                complex * vec,
                complex a,
                complex * out,
                unsigned int nrows,
                unsigned int nthr)
