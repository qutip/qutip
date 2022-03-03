#cython: language_level=3

cdef struct _csr_mat:
    double complex * data
    int * indices
    int * indptr
    int nnz
    int nrows
    int ncols
    int is_set
    int max_length
    int numpy_lock

cdef struct _coo_mat:
    double complex * data
    int * rows
    int * cols
    int nnz
    int nrows
    int ncols
    int is_set
    int max_length
    int numpy_lock

ctypedef _csr_mat CSR_Matrix
ctypedef _coo_mat COO_Matrix
