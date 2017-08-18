# distutils: language = c++
import numpy as np
cimport numpy as np
import cython
cimport cython
from qutip.qobj import Qobj
from qutip.cy.spmath cimport _zcsr_add_core
from qutip.cy.inter cimport zinterpolate, interpolate
from qutip.cy.spmatfuncs cimport spmvpy

include 'complex_math.pxi'
include 'sparse_routines.pxi'

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef class cy_compiled_td_qobj:
    cdef int total_elem
    cdef int shape0, shape1
    cdef int super
    cdef CSR_Matrix cte
    cdef double w

    def __init__(self):
        pass

    def set_data(self, cte):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.cte = CSR_from_scipy(cte.data)
        self.total_elem = cte.data.data.shape[0]
        self.super = cte.issuper

    def call(self, double t, int data=0):
        cdef CSR_Matrix out
        copy_CSR(&out, &self.cte)
        scipy_obj = CSR_to_scipy(&out)
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1., out, self.shape0)

    def rhs(self, double t, np.ndarray[complex, ndim=1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0, dtype=complex)
        self._rhs_mat(t, &vec[0], &out[0])
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_psi(self, double t, complex* vec, int isherm):
        cdef np.ndarray[complex, ndim=1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1., &y[0], self.shape0)
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
    cdef complex _expect_mat_rho(self, double t, complex* vec, int isherm):
        cdef int row
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>np.sqrt(num_rows)
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

    def expect(self, double t, np.ndarray[complex, ndim=1] vec, int isherm):
        if self.super:
            return self._expect_mat_rho(t, &vec[0], isherm)
        else:
            return self._expect_mat_psi(t, &vec[0], isherm)

#ptr and object all target the same ctdqo, so not usefull here
cdef cy_compiled_td_qobj ctdqo = cy_compiled_td_qobj()

def get_object():
    return ctdqo

cdef void rhs_mat(double t, complex* vec, complex* out):
    ctdqo._rhs_mat(t, vec, out)

cdef complex _expect_mat_psi(double t, complex* vec, int isherm):
    return ctdqo._expect_mat_psi(t, vec, isherm)

cdef complex _expect_mat_rho(double t, complex* vec, int isherm):
    return ctdqo._expect_mat_rho(t, vec, isherm)

def get_ptr():
    cdef void * ptr1 = <void*>rhs_mat
    cdef void * ptr2
    if ctdqo.super:
        ptr2 = <void*>_expect_mat_rho
    else:
        ptr2 = <void*>_expect_mat_psi
    return PyLong_FromVoidPtr(ptr1), PyLong_FromVoidPtr(ptr2)
