# distutils: language = c++
import numpy as np
cimport numpy as np
import cython
cimport cython
from qutip.qobj import Qobj
#from scipy import sparse.csr_matrix as csr
from qutip.cy.spmath cimport _zcsr_add_core
from qutip.cy.inter cimport zinterpolate, interpolate
from qutip.cy.spmatfuncs cimport spmvpy

include '/home/eric/anaconda3/lib/python3.5/site-packages/qutip-4.3.0.dev0+089c19c-py3.5-linux-x86_64.egg/qutip/cy/complex_math.pxi'
include '/home/eric/anaconda3/lib/python3.5/site-packages/qutip-4.3.0.dev0+089c19c-py3.5-linux-x86_64.egg/qutip/cy/sparse_routines.pxi'

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef class cy_compiled_td_qobj:
    cdef int total_elem
    cdef int shape0, shape1

    #pointer to data
    cdef CSR_Matrix cte
    cdef CSR_Matrix op0
    cdef int op0_sum_elem
    #args
    cdef double w


    def __init__(self):
        pass


    def set_data(self, cte, ops):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]

        self.cte = CSR_from_scipy(cte.data)
        cummulative_op = cte.data
        self.op0 = CSR_from_scipy(ops[0][0].data)
        cummulative_op += ops[0][0].data
        self.op0_sum_elem = cummulative_op.data.shape[0]

        self.total_elem = self.op0_sum_elem


    def set_args(self, args, str_args, tlist):
         self.w = args['w']


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void factor(self, t, complex* out):
        cdef double w = self.w

        out[0] = cos(w*t)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, CSR_Matrix * out):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(1, dtype=complex)
        self.factor(t, &coeff[0])

        _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                       self.op0.data, self.op0.indices, self.op0.indptr, coeff[0],
                       out, self.shape0, self.shape1)


    def call(self, double t, int data=0):
        cdef CSR_Matrix out
        init_CSR(&out, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out)
        scipy_obj = CSR_to_scipy(&out)
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr, vec, 1., out, self.shape0)

    def rhs(self, double t, np.ndarray[complex, ndim=1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0, dtype=complex)
        self._rhs_mat(t, &vec[0], &out[0])
        return out

    def rhs_ptr(self):
        cdef void * ptr = <void*>self._rhs_mat
        return PyLong_FromVoidPtr(ptr)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)
        cdef np.ndarray[complex, ndim=1] y = np.empty(self.shape0, dtype=complex)
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr, vec, 1., &y[0], self.shape0)
        cdef int row
        cdef complex dot = 0

        for row from 0 <= row < self.shape0:
            dot += conj(vec[row])*y[row]

        if isherm:
            return real(dot)
        else:
            return dot

    def expect(self, double t, np.ndarray[complex, ndim=1] vec, int isherm):
        return self._expect_mat(t, &vec[0], isherm)

    def expect_ptr(self):
        cdef void * ptr = <void*>self._expect_mat
        return PyLong_FromVoidPtr(ptr)
