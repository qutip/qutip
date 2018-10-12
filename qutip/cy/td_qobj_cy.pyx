# distutils: language = c++
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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
import scipy.sparse as sp
cimport numpy as np
import cython
cimport cython
from qutip.qobj import Qobj
from qutip.cy.spmath cimport _zcsr_add_core
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.spmath import zcsr_add
from qutip.cy.td_qobj_factor cimport coeffFunc
#from libc.stdlib cimport malloc, free
cimport libc.math

include "complex_math.pxi"
include "sparse_routines.pxi"

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void spmmCpy(complex * data, int * ind, int * ptr,
            complex * mat, complex a, complex * out,
            unsigned int sp_rows, unsigned int nrows, unsigned int ncols):
    """
    sparse*dense "C" ordered.
    """
    cdef int row, col, ii
    for row from 0 <= row < sp_rows :
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            for col in range(ncols):
                out[row * ncols + col] += a*data[jj]*mat[ind[jj] * ncols + col]

def spmmc(complex[::1] data, int[::1] ind, int[::1] ptr, complex[:,::1] A):
    cdef unsigned int sp_rows = ptr.shape[0]-1
    cdef unsigned int nrows = A.shape[0]
    cdef unsigned int ncols = A.shape[1]
    cdef np.ndarray[complex, ndim=2, mode="c"] out = np.zeros((sp_rows,ncols), dtype=complex)
    spmmCpy(&data[0], &ind[0], &ptr[0], &A[0,0], 1., &out[0,0], sp_rows, nrows, ncols)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void spmmFpy(complex * data, int * ind, int * ptr,
            complex * mat, complex a, complex * out,
            unsigned int sp_rows, unsigned int nrows, unsigned int ncols):
    cdef int col
    for col in range(ncols):
        spmvpy(data, ind, ptr, mat+nrows*col, a, out+sp_rows*col, sp_rows)


def spmmf(complex[::1] data, int[::1] ind, int[::1] ptr, complex[::1,:] A):
    cdef unsigned int sp_rows = ptr.shape[0]-1
    cdef unsigned int nrows = A.shape[0]
    cdef unsigned int ncols = A.shape[1]
    cdef np.ndarray[complex, ndim=2, mode="fortran"] out = np.zeros((sp_rows,ncols), dtype=complex, order="F")
    spmmFpy(&data[0], &ind[0], &ptr[0], &A[0,0], 1., &out[0,0], sp_rows, nrows, ncols)
    return out

"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef complex _expect_super(self, double t, complex* vec, int isherm):
    cdef int row, i
    cdef int jj, row_start, row_end
    cdef int num_rows = self.shape0
    cdef int n = <int>libc.math.sqrt(num_rows)
    cdef complex dot = 0.0
    cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                      dtype=complex)
    self.factor(t, &coeff[0])

    for row from 0 <= row < num_rows by n+1:
        row_start = self.cte.indptr[row]
        row_end = self.cte.indptr[row+1]
        for jj from row_start <= jj < row_end:
            dot += self.cte.data[jj]*vec[self.cte.indices[jj]]
    for i in range(self.N_ops):
        for row from 0 <= row < num_rows by n+1:
            row_start = self.ops[i].indptr[row]
            row_end = self.ops[i].indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.ops[i].data[jj] * \
                      vec[self.ops[i].indices[jj]] * coeff[i]
    if isherm:
        return real(dot)
    else:
        return dot
"""
def zcsr_match(sparses_list):
    """
    For a list of csr sparse matrice A,
    set them so the their indptr and indices be all equal.
    Require keeping 0s in the data, but summation can be done in vector form.
    """
    full_shape = sparses_list[0].copy()
    for sparse_elem in sparses_list[1:]:
        full_shape.data *= 0.
        full_shape.data += 1.
        if sparse_elem.indptr[-1] != 0:
            full_shape = zcsr_add(
                      full_shape.data, full_shape.indices, full_shape.indptr,
                      sparse_elem.data, sparse_elem.indices, sparse_elem.indptr,
                      full_shape.shape[0], full_shape.shape[1],
                      full_shape.indptr[-1], sparse_elem.indptr[-1], 0.)
    out = []
    for sparse_elem in sparses_list[:]:
        full_shape.data *= 0.
        if sparse_elem.indptr[-1] != 0:
            out.append(zcsr_add(
                      full_shape.data, full_shape.indices, full_shape.indptr,
                      sparse_elem.data, sparse_elem.indices, sparse_elem.indptr,
                      full_shape.shape[0], full_shape.shape[1],
                      full_shape.indptr[-1], sparse_elem.indptr[-1], 1.))
        else:
            out.append(full_shape.copy())
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef shallow_get_state(CSR_Matrix* mat):
    """
    Converts a CSR sparse matrix to a tuples for pickling.
    No deep copy of the data, pointer are passed.
    """
    long_data = PyLong_FromVoidPtr(<void *>&mat.data[0])
    long_indices = PyLong_FromVoidPtr(<void *>&mat.indices[0])
    long_indptr = PyLong_FromVoidPtr(<void *>&mat.indptr[0])
    return (long_data,  long_indices,  long_indptr,
            mat.nrows, mat.ncols, mat.nnz, mat.max_length,
            mat.is_set, mat.numpy_lock)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef shallow_set_state(CSR_Matrix* mat, state):
    """
    Converts a CSR sparse matrix to a tuples for pickling.
    No deep copy of the data, pointer are passed.
    """
    mat.data = <complex*>PyLong_AsVoidPtr(state[0])
    mat.indices = <int*>PyLong_AsVoidPtr(state[1])
    mat.indptr = <int*>PyLong_AsVoidPtr(state[2])
    mat.nrows = state[3]
    mat.ncols = state[4]
    mat.nnz = state[5]
    mat.max_length = state[6]
    mat.is_set = state[7]
    mat.numpy_lock = state[8]


cdef class cy_qobj:
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        """self * vec"""
        pass

    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                    int nrow, int ncols):
        """self * dense mat fortran ordered """
        pass

    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                    int nrow, int ncols):
        """self * dense mat c ordered"""
        pass

    cdef complex _expect(self, double t, complex* vec, int isherm):
        """<vec| self |vec>"""
        return 0.

    cdef complex _expect_super(self, double t, complex* rho, int isherm):
        """tr( self * rho )"""
        return 0.

    def set_factor(self, func=None, ptr=False, obj=None):
        cdef void* f_ptr
        self.factor_use_ptr = 0
        self.factor_use_cobj = 0
        if ptr:
            self.factor_use_ptr = 1
            f_ptr = PyLong_AsVoidPtr(ptr)
            self.factor_ptr = <void(*)(double, complex*)> f_ptr
        elif func is not None:
            self.factor_func = func
        elif obj is not None:
            self.factor_use_cobj = 1
            self.factor_cobj = obj
        else:
            raise Exception("Could not set coefficient function")

    cdef void factor(self, double t):
        cdef int i
        if self.factor_use_ptr:
            self.factor_ptr(t, self.coeff_ptr)
        elif self.factor_use_cobj:
            self.factor_cobj._call_core(t, self.coeff_ptr)
        else:
            coeff = self.factor_func(t)
            for i in range(self.N_ops):
                self.coeff_ptr[i] = coeff[i]



    def mul_vec(self, double t, complex[::1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0,
                                                        dtype=complex)
        self._mul_vec(t, &vec[0], &out[0])
        return out

    def mul_mat(self, double t, np.ndarray[complex, ndim=2] mat):
        cdef np.ndarray[complex, ndim=2] out
        cdef unsigned int sp_rows = self.shape0
        cdef unsigned int nrows = mat.shape[0]
        cdef unsigned int ncols = mat.shape[1]
        if mat.flags["F_CONTIGUOUS"]:
            out = np.zeros((sp_rows,ncols), dtype=complex, order="F")
            self._mul_matf(t,&mat[0,0],&out[0,0],nrows,ncols)
        else:
            out = np.zeros((sp_rows,ncols), dtype=complex)
            self._mul_matc(t,&mat[0,0],&out[0,0],nrows,ncols)
        return out

    def expect(self, double t, complex[::1] vec, int isherm):
        if self.super:
            return self._expect_super(t, &vec[0], isherm)
        else:
            return self._expect(t, &vec[0], isherm)

    def ode_mul_mat_F_vec(self, double t, complex[::1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape1*self.shape1,
                                                      dtype=complex)
        self._mul_matf(t, &vec[0], &out[0], self.shape1, self.shape1)
        return out

    def call(self, double t, int data=0):
        return None

    def call_with_coeff(self, double t, complex[::1] coeff, int data=0):
        return None

    """def set_factor(self, func=None, ptr=False):
        pass"""

    def set_data(self, cte):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass


cdef class cy_cte_qobj(cy_qobj):
    def set_data(self, cte):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.cte = CSR_from_scipy(cte.data)
        self.total_elem = cte.data.data.shape[0]
        self.super = cte.issuper

    def __getstate__(self):
        CSR_info = shallow_get_state(&self.cte)
        return (self.shape0, self.shape1, self.dims,
                self.total_elem, self.super, CSR_info)

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.total_elem = state[3]
        self.super = state[4]
        shallow_set_state(&self.cte, state[5])

    def call(self, double t, int data=0):
        cdef CSR_Matrix out
        out.is_set = 0
        copy_CSR(&out, &self.cte)
        scipy_obj = CSR_to_scipy(&out)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj,dims = self.dims)

    def call_with_coeff(self, double t, complex[::1] coeff, int data=0):
        cdef CSR_Matrix out
        out.is_set = 0
        copy_CSR(&out, &self.cte)
        scipy_obj = CSR_to_scipy(&out)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1.,
               out, self.shape0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        spmmFpy(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        spmmCpy(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec, int isherm):
        cdef complex[::1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1.,
               &y[0], self.shape0)
        cdef int row
        cdef complex dot = 0
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row])*y[row]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec, int isherm):
        cdef int row
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0

        for row from 0 <= row < num_rows by n+1:
            row_start = self.cte.indptr[row]
            row_end = self.cte.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.cte.data[jj]*vec[self.cte.indices[jj]]

        if isherm:
            return real(dot)
        else:
            return dot


cdef class cy_cte_qobj_dense(cy_qobj):
    def set_data(self, cte):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.cte = cte.data.toarray()
        self.super = cte.issuper

    def __getstate__(self):
        return (self.shape0, self.shape1, self.dims,
                self.super, np.array(self.cte))

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.super = state[3]
        self.cte = state[4]

    def call(self, double t, int data=0):
        if data:
            return sp.csr_matrix(self.cte, dtype=complex, copy=True)
        else:
            return Qobj(self.cte, dims = self.dims)

    def call_with_coeff(self, double t, complex[::1] coeff, int data=0):
        if data:
            return sp.csr_matrix(self.cte, dtype=complex, copy=True)
        else:
            return Qobj(self.cte, dims = self.dims)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        cdef int i,j
        cdef complex* ptr
        for i in range(self.shape0):
            ptr = &self.cte[i,0]
            for j in range(self.shape1):
                out[i] += ptr[j]*vec[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i,j,k
        cdef complex* ptr = &self.cte[0,0]
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i+j*self.shape0] += ptr[i*nrow+k]*mat[k+j*nrow]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i,j,k
        cdef complex* ptr = &self.cte[0,0]
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i*ncol+j] += ptr[i*nrow+k]*mat[k*ncol+j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec, int isherm):
        cdef int i,j
        cdef complex dot = 0
        for i in range(self.shape0):
          for j in range(self.shape1):
            dot += conj(vec[i])*self.cte[i,j]*vec[j]

        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec, int isherm):
        cdef int row, i
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0
        for row from 0 <= row < num_rows by n+1:
          for i in range(self.shape1):
            dot += self.cte[row,i]*vec[i]

        if isherm:
            return real(dot)
        else:
            return dot


cdef class cy_td_qobj(cy_qobj):
    def __init__(self):
        self.N_ops = 0
        self.ops = <CSR_Matrix**> PyDataMem_NEW(0 * sizeof(CSR_Matrix*))

    def __del__(self):
        for i in range(self.N_ops):
            PyDataMem_FREE(self.ops[i])
        PyDataMem_FREE(self.ops)

    def set_data(self, cte, ops):
        cdef int i
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.cte = CSR_from_scipy(cte.data)
        cummulative_op = cte.data
        self.super = cte.issuper

        self.N_ops = len(ops)
        self.coeff = np.empty((self.N_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]
        PyDataMem_FREE(self.ops)
        self.ops = <CSR_Matrix**> PyDataMem_NEW(self.N_ops * sizeof(CSR_Matrix*))
        self.sum_elem = np.zeros(self.N_ops, dtype=int)
        for i, op in enumerate(ops):
            self.ops[i] = <CSR_Matrix*> PyDataMem_NEW(sizeof(CSR_Matrix))
            CSR_from_scipy_inplace(op[0].data, self.ops[i])
            cummulative_op += op[0].data
            self.sum_elem[i] = cummulative_op.data.shape[0]

        self.total_elem = self.sum_elem[self.N_ops-1]



    def __getstate__(self):
        cte_info = shallow_get_state(&self.cte)
        ops_info = ()
        sum_elem = ()
        for i in range(self.N_ops):
            ops_info += (shallow_get_state(self.ops[i]),)
            sum_elem += (self.sum_elem[i],)

        factor_ptr = PyLong_FromVoidPtr(<void*>self.factor_ptr)
        factor_func = PyLong_FromVoidPtr(<void*>self.factor_func)
        return (self.shape0, self.shape1, self.dims, self.total_elem, self.super,
                self.factor_use_ptr, factor_ptr, factor_func, self.N_ops,
                sum_elem, cte_info, ops_info)

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.total_elem = state[3]
        self.super = state[4]
        self.factor_use_ptr = state[5]
        self.factor_ptr = <void(*)(double, complex*)> PyLong_AsVoidPtr(state[6])
        self.factor_func = <object> PyLong_AsVoidPtr(state[7])
        self.N_ops = state[8]
        shallow_set_state(&self.cte, state[10])
        self.sum_elem = np.zeros(self.N_ops, dtype=int)
        self.ops = <CSR_Matrix**> PyDataMem_NEW(self.N_ops * sizeof(CSR_Matrix*))
        for i in range(self.N_ops):
            self.ops[i] = <CSR_Matrix*> PyDataMem_NEW(sizeof(CSR_Matrix))
            self.sum_elem[i] = state[9][i]
            shallow_set_state(self.ops[i], state[11][i])
        self.coeff = np.empty((self.N_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, CSR_Matrix * out,
                                    complex* coeff):
        cdef int i
        cdef CSR_Matrix previous, next

        if(self.N_ops ==1):
            _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                         self.ops[0].data, self.ops[0].indices,
                         self.ops[0].indptr,
                         coeff[0], out, self.shape0, self.shape1)
        else:
            # Ugly with a loop for 1 to N-2...
            # It save the copy of data from cte and out
            # no init/free to cte, out
            init_CSR(&next, self.sum_elem[0], self.shape0, self.shape1)
            _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                           self.ops[0].data,
                           self.ops[0].indices,
                           self.ops[0].indptr,
                           coeff[0], &next, self.shape0, self.shape1)
            #previous = next
            #next = CSR_Matrix()
            previous, next = next, previous

            for i in range(1,self.N_ops-1):
                init_CSR(&next, self.sum_elem[i], self.shape0, self.shape1)
                _zcsr_add_core(previous.data, previous.indices,
                               previous.indptr,
                               self.ops[i].data,
                               self.ops[i].indices,
                               self.ops[i].indptr,
                               coeff[i], &next, self.shape0, self.shape1)
                free_CSR(&previous)
                #previous = next
                #next = CSR_Matrix()
                previous, next = next, previous

            _zcsr_add_core(previous.data, previous.indices, previous.indptr,
                           self.ops[self.N_ops-1].data,
                           self.ops[self.N_ops-1].indices,
                           self.ops[self.N_ops-1].indptr,
                           coeff[self.N_ops-1], out, self.shape0, self.shape1)
            free_CSR(&previous)

    def call(self, double t, int data=0):
        cdef CSR_Matrix out
        init_CSR(&out, self.total_elem, self.shape0, self.shape1)
        self.factor(t)
        self._call_core(t, &out, self.coeff_ptr)
        scipy_obj = CSR_to_scipy(&out)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj,dims = self.dims)

    def call_with_coeff(self, double t, complex[::1] coeff, int data=0):
        cdef CSR_Matrix out
        init_CSR(&out, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out, &coeff[0])
        scipy_obj = CSR_to_scipy(&out)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        self.factor(t)
        cdef int i
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec,
               1., out, self.shape0)
        for i in range(self.N_ops):
            spmvpy(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                   vec, self.coeff_ptr[i], out, self.shape0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        self.factor(t)
        cdef int i
        spmmFpy(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)
        for i in range(self.N_ops):
             spmmFpy(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                 mat, self.coeff_ptr[i], out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        self.factor(t)
        cdef int i
        spmmCpy(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)
        for i in range(self.N_ops):
             spmmCpy(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                 mat, self.coeff_ptr[i], out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec, int isherm):
        cdef complex [::1] y = np.zeros(self.shape0, dtype=complex)
        cdef int row
        cdef complex dot = 0
        self._mul_vec(t, &vec[0], &y[0])
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row]) * y[row]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec, int isherm):
        cdef int row, i
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0
        self.factor(t)

        for row from 0 <= row < num_rows by n+1:
            row_start = self.cte.indptr[row]
            row_end = self.cte.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.cte.data[jj]*vec[self.cte.indices[jj]]
        for i in range(self.N_ops):
            for row from 0 <= row < num_rows by n+1:
                row_start = self.ops[i].indptr[row]
                row_end = self.ops[i].indptr[row+1]
                for jj from row_start <= jj < row_end:
                    dot += self.ops[i].data[jj] * \
                          vec[self.ops[i].indices[jj]] * self.coeff_ptr[i]
        if isherm:
            return real(dot)
        else:
            return dot


cdef class cy_td_qobj_dense(cy_qobj):
    def set_data(self, cte, ops):
        cdef int i,j,k
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.super = cte.issuper
        self.N_ops = len(ops)
        self.cte = cte.data.toarray()
        self.ops = np.zeros((self.N_ops, self.shape0, self.shape1),
                            dtype=complex)
        self.data_t = np.empty((self.shape0, self.shape1), dtype=complex)
        self.data_ptr = &self.data_t[0,0]
        self.coeff = np.empty((self.N_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]
        for i, op in enumerate(ops):
          oparray = op[0].data.toarray()
          for j in range(self.shape0):
            for k in range(self.shape1):
              self.ops[i,j,k] = oparray[j,k]

    """def set_factor(self, func=None, ptr=False):
        cdef void* f_ptr
        if ptr:
            self.factor_use_ptr = 1
            f_ptr = PyLong_AsVoidPtr(ptr)
            self.factor_ptr = <void(*)(double, complex*)> f_ptr
        else:
            self.factor_use_ptr = 0
            self.factor_func = func"""

    def __getstate__(self):
        factor_ptr = PyLong_FromVoidPtr(<void*>self.factor_ptr)
        factor_func = PyLong_FromVoidPtr(<void*>self.factor_func)
        return (self.shape0, self.shape1, self.dims, self.super,
                self.factor_use_ptr, factor_ptr, factor_func, self.N_ops,
                np.array(self.cte), np.array(self.ops))

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.super = state[3]
        self.factor_use_ptr = state[4]
        self.factor_ptr = <void(*)(double, complex*)> PyLong_AsVoidPtr(state[5])
        self.factor_func = <object> PyLong_AsVoidPtr(state[6])
        self.N_ops = state[7]
        self.cte = state[8]
        self.ops = state[9]
        self.data_t = np.empty((self.shape0, self.shape1), dtype=complex)
        self.data_ptr = &self.data_t[0,0]
        self.coeff = np.empty((self.N_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]

    """cdef void factor(self, double t):
        cdef int i
        if self.factor_use_ptr:
            self.factor_ptr(t, self.coeff_ptr)
        else:
            coeff = self.factor_func(t)
            for i in range(self.N_ops):
                self.coeff_ptr[i] = coeff[i]"""

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, complex[:,::1] out, complex* coeff):
        cdef int i,j
        cdef complex* ptr
        cdef complex* out_ptr
        #copy(self.cte, out)
        ptr = &self.cte[0,0]
        out_ptr = &out[0,0]
        for i in range(self.shape0 * self.shape0):
            out_ptr[i] = ptr[i]
        for i in range(self.N_ops):
            ptr = &self.ops[i,0,0]
            for j in range(self.shape0 * self.shape0):
                out_ptr[j] += ptr[j]*coeff[i]

    def call(self, double t, int data=0):
        cdef np.ndarray[complex, ndim=2] data_t = \
                  np.empty((self.shape0, self.shape1), dtype=complex)
        self.factor(t)
        self._call_core(t, data_t, self.coeff_ptr)

        if data:
            return sp.csr_matrix(data_t, dtype=complex, copy=True)
        else:
            return Qobj(data_t, dims = self.dims)

    def call_with_coeff(self, double t, complex[::1] coeff, int data=0):
        cdef np.ndarray[complex, ndim=2] data_t = \
                    np.empty((self.shape0, self.shape1), dtype=complex)
        self._call_core(t, data_t, &coeff[0])
        if data:
            return sp.csr_matrix(data_t, dtype=complex, copy=True)
        else:
            return Qobj(data_t, dims = self.dims)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)

        cdef int i,j
        for i in range(self.shape0):
            for j in range(self.shape1):
                out[i] += self.data_t[i,j]*vec[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i,j,k
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i+j*self.shape0] += self.data_ptr[i*nrow+k]*mat[k+j*nrow]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i,j,k
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i*ncol+j] += self.data_ptr[i*nrow+k]*mat[k*ncol+j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec, int isherm):
        cdef int row
        cdef complex dot = 0
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)
        for i in range(self.shape0):
          for j in range(self.shape1):
            dot += conj(vec[i])*self.data_t[i,j]*vec[j]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec, int isherm):
        cdef int row, i
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)

        for row from 0 <= row < num_rows by n+1:
          for i in range(self.shape1):
            dot += self.data_t[row,i]*vec[i]

        if isherm:
            return real(dot)
        else:
            return dot


cdef class cy_td_qobj_matched(cy_qobj):
    def set_data(self, cte, ops):
        cdef int i,j
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.super = cte.issuper
        self.N_ops = len(ops)
        self.coeff = np.zeros((self.N_ops), dtype=complex)
        self.coeff_ptr = &self.coeff[0]

        sparse_list = []
        for op in ops:
            sparse_list.append(op[0].data)
        sparse_list += [cte.data]
        matched = zcsr_match(sparse_list)

        self.indptr = matched[0].indptr
        self.indices = matched[0].indices
        self.cte = matched[0].data
        self.nnz = len(self.cte)
        self.data_t = np.zeros((self.nnz), dtype=complex)
        self.data_ptr = &self.data_t[0]

        self.ops = np.zeros((self.N_ops, self.nnz), dtype=complex)
        for i, op in enumerate(matched[1:]):
          for j in range(self.nnz):
            self.ops[i,j] = op.data[j]

    """def set_factor(self, func=None, ptr=False):
        cdef void* f_ptr
        if ptr:
            self.factor_use_ptr = 1
            f_ptr = PyLong_AsVoidPtr(ptr)
            self.factor_ptr = <void(*)(double, complex*)> f_ptr
        else:
            self.factor_use_ptr = 0
            self.factor_func = func"""

    def __getstate__(self):
        factor_ptr = PyLong_FromVoidPtr(<void*>self.factor_ptr)
        factor_func = PyLong_FromVoidPtr(<void*>self.factor_func)
        return (self.shape0, self.shape1, self.dims, self.nnz, self.super,
                self.factor_use_ptr, factor_ptr, factor_func, self.N_ops,
                np.array(self.indptr), np.array(self.indices),
                np.array(self.cte), np.array(self.ops))

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.nnz = state[3]
        self.super = state[4]
        self.factor_use_ptr = state[5]
        self.factor_ptr = <void(*)(double, complex*)> PyLong_AsVoidPtr(state[6])
        self.factor_func = <object> PyLong_AsVoidPtr(state[7])
        self.N_ops = state[8]

        self.indptr = state[9]
        self.indices = state[10]
        self.cte = state[11]
        self.ops = state[12]
        self.coeff = np.zeros((self.N_ops), dtype=complex)
        self.coeff_ptr = &self.coeff[0]
        self.data_t = np.zeros((self.nnz), dtype=complex)
        self.data_ptr = &self.data_t[0]

    """cdef void factor(self, double t):
        cdef int i
        if self.factor_use_ptr:
            self.factor_ptr(t, self.coeff_ptr)
        else:
            coeff = self.factor_func(t)
            for i in range(self.N_ops):
                self.coeff_ptr[i] = coeff[i]"""

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, complex[::1] out, complex* coeff):
        cdef int i,j
        cdef complex * ptr
        ptr = &self.cte[0]
        for j in range(self.nnz):
            out[j] = ptr[j]
        for i in range(self.N_ops):
            ptr = &self.ops[i,0]
            for j in range(self.nnz):
                out[j] += ptr[j]*coeff[i]

    def call(self, double t, int data=0):
        cdef int i
        cdef complex[::1] data_t = np.empty(self.nnz, dtype=complex)
        self.factor(t)
        self._call_core(t, data_t, self.coeff_ptr)

        cdef CSR_Matrix out_csr
        init_CSR(&out_csr, self.nnz, self.shape0, self.shape1)
        for i in range(self.nnz):
            out_csr.data[i] = data_t[i]
            out_csr.indices[i] = self.indices[i]
        for i in range(self.shape0+1):
            out_csr.indptr[i] = self.indptr[i]
        scipy_obj = CSR_to_scipy(&out_csr)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj,dims = self.dims)

    def call_with_coeff(self, double t, complex[::1] coeff, int data=0):
        cdef complex[::1] out = np.empty(self.nnz, dtype=complex)
        self._call_core(t, out, &coeff[0])
        cdef CSR_Matrix out_csr
        init_CSR(&out_csr, self.nnz, self.shape0, self.shape1)
        for i in range(self.nnz):
            out_csr.data[i] = out[i]
            out_csr.indices[i] = self.indices[i]
        for i in range(self.shape0+1):
            out_csr.indptr[i] = self.indptr[i]
        scipy_obj = CSR_to_scipy(&out_csr)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj,dims = self.dims)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)
        spmvpy(self.data_ptr, &self.indices[0], &self.indptr[0], vec,
               1., out, self.shape0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)
        spmmFpy(self.data_ptr, &self.indices[0], &self.indptr[0], mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)
        spmmCpy(self.data_ptr, &self.indices[0], &self.indptr[0], mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec, int isherm):
        cdef complex [::1] y = np.zeros(self.shape0, dtype=complex)
        cdef int row
        cdef complex dot = 0
        self._mul_vec(t, &vec[0], &y[0])
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row]) * y[row]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec, int isherm):
        cdef int row, i
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0

        self.factor(t)
        self._call_core(t, self.data_t, self.coeff_ptr)

        for row from 0 <= row < num_rows by n+1:
            row_start = self.indptr[row]
            row_end = self.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.data_ptr[jj]*vec[self.indices[jj]]

        if isherm:
            return real(dot)
        else:
            return dot
