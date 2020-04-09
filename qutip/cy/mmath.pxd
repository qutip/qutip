#cython: language_level=3


cdef void spmvpy(complex * data, int * ind, int *  ptr,
                 complex * vec, complex a, complex * out,
                 unsigned int nrows, unsigned int nthr)

cdef void spmmcpy(complex* data,
                       int* ind,
                       int* ptr,
                       complex* mat,
                       complex a,
                       complex* out,
                       int sp_rows,
                       unsigned int nrows,
                       unsigned int ncols,
                       int nthr)

cdef void spmmfpy(complex* data, int* ind, int* ptr,
                       complex* mat,
                       complex a,
                       complex* out,
                       int sp_rows,
                       unsigned int nrows,
                       unsigned int ncols,
                       int nthr)
