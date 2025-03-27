#cython: language_level=3

cimport numpy as cnp
cimport cython

cdef void spmvpy_openmp(
    complex *data, int *ind, int *ptr, complex * vec, complex a, complex * out,
    unsigned int nrows, unsigned int nthr,
)
