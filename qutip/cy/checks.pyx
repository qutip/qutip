#cython: language_level=3

import numpy as np
import scipy.sparse as sp
from qutip.fastsparse import fast_csr_matrix
cimport numpy as cnp
cimport cython

include "sparse_routines.pxi"


def _test_coo2csr_struct(object A):
    cdef COO_Matrix mat = COO_from_scipy(A)
    cdef CSR_Matrix out
    COO_to_CSR(&out, &mat)
    return CSR_to_scipy(&out)


def _test_sorting(object A):
    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]

    cdef CSR_Matrix out

    out.data = &data[0]
    out.indices = &ind[0]
    out.indptr = &ptr[0]
    out.nrows = nrows
    out.ncols = ncols
    out.is_set = 1
    out.numpy_lock = 0
    sort_indices(&out)


def _test_coo2csr_inplace_struct(object A, int sorted = 0):
    cdef complex[::1] data = A.data
    cdef int[::1] rows = A.row
    cdef int[::1] cols = A.col
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]
    cdef int nnz = data.shape[0]
    cdef size_t kk
    #We need to make copies here to test the inplace conversion
    #as we cannot use numpy data due to ownership issues.
    cdef complex * _data = <complex *>PyDataMem_NEW(nnz * sizeof(complex))
    cdef int * _rows = <int *>PyDataMem_NEW(nnz * sizeof(int))
    cdef int * _cols = <int *>PyDataMem_NEW(nnz * sizeof(int))
    for kk in range(nnz):
        _data[kk] = data[kk]
        _rows[kk] = rows[kk]
        _cols[kk] = cols[kk]

    cdef COO_Matrix mat
    mat.data = _data
    mat.rows = _rows
    mat.cols = _cols
    mat.nrows = nrows
    mat.ncols = ncols
    mat.nnz = nnz
    mat.max_length = mat.nnz
    mat.is_set = 1
    mat.numpy_lock = 0

    cdef CSR_Matrix out

    COO_to_CSR_inplace(&out, &mat)
    if sorted:
        sort_indices(&out)
    return CSR_to_scipy(&out)


def _test_csr2coo_struct(object A):
    cdef CSR_Matrix mat = CSR_from_scipy(A)
    cdef COO_Matrix out
    CSR_to_COO(&out, &mat)
    return COO_to_scipy(&out)
