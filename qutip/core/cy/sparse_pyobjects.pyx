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
cimport numpy as np
cimport cython
from scipy.sparse import coo_matrix
from qutip.core.fastsparse import fast_csr_matrix
from qutip.core.cy.sparse_structs cimport CSR_Matrix, COO_Matrix
from qutip.core.cy.sparse_routines cimport raise_error_CSR, raise_error_COO
np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

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

