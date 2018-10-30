#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
from scipy.sparse import coo_matrix
from qutip.fastsparse import fast_csr_matrix
cimport numpy as np
cimport cython
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from qutip.cy.sparse_structs cimport CSR_Matrix, COO_Matrix
np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)


#Struct used for CSR indices sorting
cdef struct _data_ind_pair:
    double complex data
    int ind

ctypedef _data_ind_pair data_ind_pair
ctypedef int (*cfptr)(data_ind_pair, data_ind_pair)


cdef void raise_error_CSR(int E, CSR_Matrix * C = NULL):
    if not C.numpy_lock and C != NULL:
        free_CSR(C)
    if E == -1:
        raise MemoryError('Could not allocate memory.')
    elif E == -2:
        raise Exception('Error manipulating CSR_Matrix structure.')
    elif E == -3:
        raise Exception('CSR_Matrix is not initialized.')
    elif E == -4:
        raise Exception('NumPy already has lock on data.')
    elif E == -5:
        raise Exception('Cannot expand data structures past max_length.')
    elif E == -6:
        raise Exception('CSR_Matrix cannot be expanded.')
    elif E == -7:
        raise Exception('Data length cannot be larger than max_length')
    else:
        raise Exception('Error in Cython code.')


cdef void raise_error_COO(int E, COO_Matrix * C = NULL):
    if not C.numpy_lock and C != NULL:
        free_COO(C)
    if E == -1:
        raise MemoryError('Could not allocate memory.')
    elif E == -2:
        raise Exception('Error manipulating COO_Matrix structure.')
    elif E == -3:
        raise Exception('COO_Matrix is not initialized.')
    elif E == -4:
        raise Exception('NumPy already has lock on data.')
    elif E == -5:
        raise Exception('Cannot expand data structures past max_length.')
    elif E == -6:
        raise Exception('COO_Matrix cannot be expanded.')
    elif E == -7:
        raise Exception('Data length cannot be larger than max_length')
    else:
        raise Exception('Error in Cython code.')


cdef inline int int_min(int a, int b) nogil:
    return b if b < a else a

cdef inline int int_max(int a, int b) nogil:
    return a if a > b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void init_CSR(CSR_Matrix * mat, int nnz, int nrows, int ncols = 0,
                int max_length = 0, int init_zeros = 1):
    """
    Initialize CSR_Matrix struct. Matrix is assumed to be square with
    shape nrows x nrows.  Manually set mat.ncols otherwise

    Parameters
    ----------
    mat : CSR_Matrix *
        Pointer to struct.
    nnz : int
        Length of data and indices arrays. Also number of nonzero elements
    nrows : int
        Number of rows in matrix. Also gives length
        of indptr array (nrows+1).
    ncols : int (default = 0)
        Number of cols in matrix. Default is ncols = nrows.
    max_length : int (default = 0)
        Maximum length of data and indices arrays.  Used for resizing.
        Default value of zero indicates no resizing.
    """
    if max_length == 0:
        max_length = nnz
    if nnz > max_length:
        raise_error_CSR(-7, mat)
    if init_zeros:
        mat.data = <double complex *>PyDataMem_NEW_ZEROED(nnz, sizeof(double complex))
    else:
        mat.data = <double complex *>PyDataMem_NEW(nnz * sizeof(double complex))
    if mat.data == NULL:
        raise_error_CSR(-1, mat)
    if init_zeros:
        mat.indices = <int *>PyDataMem_NEW_ZEROED(nnz, sizeof(int))
        mat.indptr = <int *>PyDataMem_NEW_ZEROED((nrows+1), sizeof(int))
    else:
        mat.indices = <int *>PyDataMem_NEW(nnz * sizeof(int))
        mat.indptr = <int *>PyDataMem_NEW((nrows+1) * sizeof(int))
    mat.nnz = nnz
    mat.nrows = nrows
    if ncols == 0:
        mat.ncols = nrows
    else:
        mat.ncols = ncols
    mat.is_set = 1
    mat.max_length = max_length
    mat.numpy_lock = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy_CSR(CSR_Matrix * out, CSR_Matrix * mat):
    """
    Copy a CSR_Matrix.
    """
    cdef size_t kk
    if not mat.is_set:
        raise_error_CSR(-3)
    elif out.is_set:
        raise_error_CSR(-2)
    init_CSR(out, mat.nnz, mat.nrows, mat.nrows, mat.max_length)
    # We cannot use memcpy here since there are issues with
    # doing so on Win with the GCC compiler
    for kk in range(mat.nnz):
        out.data[kk] = mat.data[kk]
        out.indices[kk] = mat.indices[kk]
    for kk in range(mat.nrows+1):
        out.indptr[kk] = mat.indptr[kk]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void init_COO(COO_Matrix * mat, int nnz, int nrows, int ncols = 0,
                int max_length = 0, int init_zeros = 1):
    """
    Initialize COO_Matrix struct. Matrix is assumed to be square with
    shape nrows x nrows.  Manually set mat.ncols otherwise

    Parameters
    ----------
    mat : COO_Matrix *
        Pointer to struct.
    nnz : int
        Number of nonzero elements.
    nrows : int
        Number of rows in matrix.
    nrows : int (default = 0)
        Number of cols in matrix. Default is ncols = nrows.
    max_length : int (default = 0)
        Maximum length of arrays.  Used for resizing.
        Default value of zero indicates no resizing.
    """
    if max_length == 0:
        max_length = nnz
    if nnz > max_length:
        raise_error_COO(-7, mat)
    if init_zeros:
        mat.data = <double complex *>PyDataMem_NEW_ZEROED(nnz, sizeof(double complex))
    else:
        mat.data = <double complex *>PyDataMem_NEW(nnz * sizeof(double complex))
    if mat.data == NULL:
        raise_error_COO(-1, mat)
    if init_zeros:
        mat.rows = <int *>PyDataMem_NEW_ZEROED(nnz, sizeof(int))
        mat.cols = <int *>PyDataMem_NEW_ZEROED(nnz, sizeof(int))
    else:
        mat.rows = <int *>PyDataMem_NEW(nnz * sizeof(int))
        mat.cols = <int *>PyDataMem_NEW(nnz * sizeof(int))
    mat.nnz = nnz
    mat.nrows = nrows
    if ncols == 0:
        mat.ncols = nrows
    else:
        mat.ncols = ncols
    mat.is_set = 1
    mat.max_length = max_length
    mat.numpy_lock = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void free_CSR(CSR_Matrix * mat):
    """
    Manually free CSR_Matrix data structures if
    data is not locked by NumPy.
    """
    if not mat.numpy_lock and mat.is_set:
        if mat.data != NULL:
            PyDataMem_FREE(mat.data)
        if mat.indices != NULL:
            PyDataMem_FREE(mat.indices)
        if mat.indptr != NULL:
            PyDataMem_FREE(mat.indptr)
        mat.is_set = 0
    else:
        raise_error_CSR(-2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void free_COO(COO_Matrix * mat):
    """
    Manually free COO_Matrix data structures if
    data is not locked by NumPy.
    """
    if not mat.numpy_lock and mat.is_set:
        if mat.data != NULL:
            PyDataMem_FREE(mat.data)
        if mat.rows != NULL:
            PyDataMem_FREE(mat.rows)
        if mat.cols != NULL:
            PyDataMem_FREE(mat.cols)
        mat.is_set = 0
    else:
        raise_error_COO(-2)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void shorten_CSR(CSR_Matrix * mat, int N):
    """
    Shortends the length of CSR data and indices arrays.
    """
    if (not mat.numpy_lock) and mat.is_set:
        mat.data = <double complex *>PyDataMem_RENEW(mat.data, N * sizeof(double complex))
        mat.indices = <int *>PyDataMem_RENEW(mat.indices, N * sizeof(int))
        mat.nnz = N
    else:
        if mat.numpy_lock:
            raise_error_CSR(-4, mat)
        elif not mat.is_set:
            raise_error_CSR(-3, mat)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void expand_CSR(CSR_Matrix * mat, int init_zeros=0):
    """
    Expands the length of CSR data and indices arrays to accomodate
    more nnz.  THIS IS CURRENTLY NOT USED
    """
    cdef size_t ii
    cdef int new_size
    if mat.nnz == mat.max_length:
        raise_error_CSR(-5, mat) #Cannot expand data past max_length.
    elif (not mat.numpy_lock) and mat.is_set:
        new_size = int_min(2*mat.nnz, mat.max_length)
        new_data = <double complex *>PyDataMem_RENEW(mat.data, new_size * sizeof(double complex))
        if new_data == NULL:
            raise_error_CSR(-1, mat)
        else:
            mat.data = new_data
            if init_zeros == 1:
                for ii in range(mat.nnz, new_size):
                    mat.data[ii] = 0

        new_ind = <int *>PyDataMem_RENEW(mat.indices, new_size * sizeof(int))
        mat.indices = new_ind
        if init_zeros == 1:
            for ii in range(mat.nnz, new_size):
                mat.indices[ii] = 0
        mat.nnz = new_size
    else:
        if mat.numpy_lock:
            raise_error_CSR(-4, mat)
        elif not mat.is_set:
            raise_error_CSR(-3, mat)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object CSR_to_scipy(CSR_Matrix * mat):
    """
    Converts a CSR_Matrix struct to a SciPy csr_matrix class object.
    The NumPy arrays are generated from the pointers, and the lifetime
    of the pointer memory is tied to that of the NumPy array
    (i.e. automatic garbage cleanup.)

    Parameters
    ----------
    mat : CSR_Matrix *
        Pointer to CSR_Matrix.
    """
    cdef np.npy_intp dat_len, ptr_len
    cdef np.ndarray[complex, ndim=1] _data
    cdef np.ndarray[int, ndim=1] _ind, _ptr
    if (not mat.numpy_lock) and mat.is_set:
        dat_len = mat.nnz
        ptr_len = mat.nrows+1
        _data = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_COMPLEX128, mat.data)
        PyArray_ENABLEFLAGS(_data, np.NPY_OWNDATA)

        _ind = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_INT32, mat.indices)
        PyArray_ENABLEFLAGS(_ind, np.NPY_OWNDATA)

        _ptr = np.PyArray_SimpleNewFromData(1, &ptr_len, np.NPY_INT32, mat.indptr)
        PyArray_ENABLEFLAGS(_ptr, np.NPY_OWNDATA)
        mat.numpy_lock = 1
        return fast_csr_matrix((_data, _ind, _ptr), shape=(mat.nrows,mat.ncols))
    else:
        if mat.numpy_lock:
            raise_error_CSR(-4)
        elif not mat.is_set:
            raise_error_CSR(-3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object COO_to_scipy(COO_Matrix * mat):
    """
    Converts a COO_Matrix struct to a SciPy coo_matrix class object.
    The NumPy arrays are generated from the pointers, and the lifetime
    of the pointer memory is tied to that of the NumPy array
    (i.e. automatic garbage cleanup.)

    Parameters
    ----------
    mat : COO_Matrix *
        Pointer to COO_Matrix.
    """
    cdef np.npy_intp dat_len
    cdef np.ndarray[complex, ndim=1] _data
    cdef np.ndarray[int, ndim=1] _row, _col
    if (not mat.numpy_lock) and mat.is_set:
        dat_len = mat.nnz
        _data = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_COMPLEX128, mat.data)
        PyArray_ENABLEFLAGS(_data, np.NPY_OWNDATA)

        _row = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_INT32, mat.rows)
        PyArray_ENABLEFLAGS(_row, np.NPY_OWNDATA)

        _col = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_INT32, mat.cols)
        PyArray_ENABLEFLAGS(_col, np.NPY_OWNDATA)
        mat.numpy_lock = 1
        return coo_matrix((_data, (_row, _col)), shape=(mat.nrows,mat.ncols))
    else:
        if mat.numpy_lock:
            raise_error_COO(-4)
        elif not mat.is_set:
            raise_error_COO(-3)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void COO_to_CSR(CSR_Matrix * out, COO_Matrix * mat):
    """
    Conversion from COO to CSR. Not in place,
    but result is sorted correctly.
    """
    cdef int i, j, iad, j0
    cdef double complex val
    cdef size_t kk
    init_CSR(out, mat.nnz, mat.nrows, mat.ncols, max_length=0, init_zeros=1)
    # Determine row lengths
    for kk in range(mat.nnz):
        out.indptr[mat.rows[kk]] = out.indptr[mat.rows[kk]] + 1
    # Starting position of rows
    j = 0
    for kk in range(mat.nrows):
        j0 = out.indptr[kk]
        out.indptr[kk] = j
        j += j0
    #Do the data
    for kk in range(mat.nnz):
        i = mat.rows[kk]
        j = mat.cols[kk]
        val = mat.data[kk]
        iad = out.indptr[i]
        out.data[iad] = val
        out.indices[iad] = j
        out.indptr[i] = iad+1
    # Shift back
    for kk in range(mat.nrows,0,-1):
        out.indptr[kk] = out.indptr[kk-1]
    out.indptr[0] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void CSR_to_COO(COO_Matrix * out, CSR_Matrix * mat):
    """
    Converts a CSR_Matrix to a COO_Matrix.
    """
    cdef int k1, k2
    cdef size_t jj, kk
    init_COO(out, mat.nnz, mat.nrows, mat.ncols)
    for kk in range(mat.nnz):
        out.data[kk] = mat.data[kk]
        out.cols[kk] = mat.indices[kk]
    for kk in range(mat.nrows-1,0,-1):
        k1 = mat.indptr[kk+1]
        k2 = mat.indptr[kk]
        for jj in range(k2, k1):
            out.rows[jj] = kk



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void COO_to_CSR_inplace(CSR_Matrix * out, COO_Matrix * mat):
    """
    In place conversion from COO to CSR. In place, but not sorted.
    The length of the COO (data,rows,cols) must be equal to the NNZ
    in the final matrix (i.e. no padded zeros on ends of arrays).
    """
    cdef size_t kk
    cdef int i, j, init, inext, jnext, ipos
    cdef int * _tmp_rows
    cdef complex val, val_next
    cdef int * work = <int *>PyDataMem_NEW_ZEROED(mat.nrows+1, sizeof(int))
    # Determine output indptr array
    for kk in range(mat.nnz):
        i = mat.rows[kk]
        work[i+1] += 1
    work[0] = 0
    for kk in range(mat.nrows):
        work[kk+1] += work[kk]

    if mat.nnz < (mat.nrows+1):
        _tmp_rows = <int *>PyDataMem_RENEW(mat.rows, (mat.nrows+1) * sizeof(int))
        mat.rows = _tmp_rows
    init = 0
    while init < mat.nnz:
        while (mat.rows[init] < 0):
            init += 1
        val = mat.data[init]
        i = mat.rows[init]
        j = mat.cols[init]
        mat.rows[init] = -1
        while 1:
            ipos = work[i]
            val_next = mat.data[ipos]
            inext = mat.rows[ipos]
            jnext = mat.cols[ipos]

            mat.data[ipos] = val
            mat.cols[ipos] = j
            mat.rows[ipos] = -1
            work[i] += 1
            if inext < 0:
                break
            val = val_next
            i = inext
            j = jnext
        init += 1

    for kk in range(mat.nrows):
        mat.rows[kk+1] = work[kk]
    mat.rows[0] = 0

    if mat.nnz > (mat.nrows+1):
        _tmp_rows = <int *>PyDataMem_RENEW(mat.rows, (mat.nrows+1) * sizeof(int))
        mat.rows = _tmp_rows
    #Free working array
    PyDataMem_FREE(work)
    #Set CSR pointers to original COO data.
    out.data = mat.data
    out.indices = mat.cols
    out.indptr = mat.rows
    out.nrows = mat.nrows
    out.ncols = mat.ncols
    out.nnz = mat.nnz
    out.max_length = mat.nnz
    out.is_set = 1
    out.numpy_lock = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ind_sort(data_ind_pair x, data_ind_pair y):
    return x.ind < y.ind


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sort_indices(CSR_Matrix * mat):
    """
    Sorts the indices of a CSR_Matrix inplace.
    """
    cdef size_t ii, jj
    cdef vector[data_ind_pair] pairs
    cdef cfptr cfptr_ = &ind_sort
    cdef int row_start, row_end, length

    for ii in range(mat.nrows):
        row_start = mat.indptr[ii]
        row_end = mat.indptr[ii+1]
        length = row_end - row_start
        pairs.resize(length)

        for jj in range(length):
            pairs[jj].data = mat.data[row_start+jj]
            pairs[jj].ind = mat.indices[row_start+jj]

        sort(pairs.begin(),pairs.end(),cfptr_)

        for jj in range(length):
            mat.data[row_start+jj] = pairs[jj].data
            mat.indices[row_start+jj] = pairs[jj].ind



@cython.boundscheck(False)
@cython.wraparound(False)
cdef CSR_Matrix CSR_from_scipy(object A):
    """
    Converts a SciPy CSR sparse matrix to a
    CSR_Matrix struct.
    """
    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]
    cdef int nnz = ptr[nrows]

    cdef CSR_Matrix mat

    mat.data = &data[0]
    mat.indices = &ind[0]
    mat.indptr = &ptr[0]
    mat.nrows = nrows
    mat.ncols = ncols
    mat.nnz = nnz
    mat.max_length = nnz
    mat.is_set = 1
    mat.numpy_lock = 1

    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void CSR_from_scipy_inplace(object A, CSR_Matrix* mat):
    """
    Converts a SciPy CSR sparse matrix to a
    CSR_Matrix struct.
    """
    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]
    cdef int nnz = ptr[nrows]

    mat.data = &data[0]
    mat.indices = &ind[0]
    mat.indptr = &ptr[0]
    mat.nrows = nrows
    mat.ncols = ncols
    mat.nnz = nnz
    mat.max_length = nnz
    mat.is_set = 1
    mat.numpy_lock = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef COO_Matrix COO_from_scipy(object A):
    """
    Converts a SciPy COO sparse matrix to a
    COO_Matrix struct.
    """
    cdef complex[::1] data = A.data
    cdef int[::1] rows = A.row
    cdef int[::1] cols = A.col
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]
    cdef int nnz = data.shape[0]

    cdef COO_Matrix mat
    mat.data = &data[0]
    mat.rows = &rows[0]
    mat.cols = &cols[0]
    mat.nrows = nrows
    mat.ncols = ncols
    mat.nnz = nnz
    mat.max_length = nnz
    mat.is_set = 1
    mat.numpy_lock = 1

    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void identity_CSR(CSR_Matrix * mat, unsigned int nrows):
    cdef size_t kk
    init_CSR(mat, nrows, nrows, nrows, 0, 0)
    for kk in range(nrows):
        mat.data[kk] = 1
        mat.indices[kk] = kk
        mat.indptr[kk] = kk
    mat.indptr[nrows] = nrows
