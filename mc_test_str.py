import numpy as np
cimport numpy as cnp
import cython
from qutip import qobj
from scipy import sparse.csr_matrix as csr
from qutip.cy.spmath import _zcsr_add_core
from qutip.cy.inter import zinterpolate, interpolate

include "sparse_routines.pxi"
include "complex_math.pxi"


cdef extern from "src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows)


cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)


cdef void split_qobj(object obj, complex*, int*, int*):
    cdef cnp.ndarray[complex, ndim=1] data = obj.data.data
    cdef cnp.ndarray[complex, ndim=1] ptr = obj.data.indptr
    cdef cnp.ndarray[complex, ndim=1] ind = obj.data.indices
    return &ptr[0], &ind[0], &data[0]


cdef class cy_compiled_td_qobj:
    cdef int total_elem
    cdef int shape0, shape1

    #pointer to data
    cdef CSR_Matrix cte_obj
    cdef CSR_Matrix * cte
    cdef CSR_Matrix op0_obj
    cdef CSR_Matrix * op0
    cdef int op0_sum_elem
    cdef CSR_Matrix op1_obj
    cdef CSR_Matrix * op1
    cdef int op1_sum_elem

    #args
    cdef complex* str_array_0
    cdef double dt
    cdef int N
    cdef double w

    def __init__(self):
        pass

    def set_args(self, args, tlist):
        self.dt = tlist[-1] / (tlist.shape[0]-1)
        self.N = args["str_array_0"].shape[0]
        self.w = args['w']

    def set_data(self, cte, ops):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]

        self.cte_obj = CSR_from_scipy(cte)
        self.cte = &self.cte_obj
        cummulative_op = cte

        self.op0_obj = CSR_from_scipy(ops[0][0])
        self.op0 = &self.op0_obj
        cummulative_op += ops[0][0]
        op0_sum_elem = cummulative_op.data.shape[0]

        self.op1_obj = CSR_from_scipy(ops[1][0])
        self.op1 = &self.op1_obj
        cummulative_op += ops[1][0]
        op1_sum_elem = cummulative_op.data.shape[0]

        total_elem = op1_sum_elem

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void factor(self, t, complex* out):
        cdef complex* str_array_0 = self.str_array_0
        cdef double dt = self.dt
        cdef int N = self.N
        cdef double w = self.w

        factor[0] = interpolate(t, str_array_0, N, dt)
        factor[1] = sin(w*t)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, CSR_Matrix * out):
        cdef CSR_Matrix * cummulative_0
        cummulative_0 = new CSR_Matrix()
        init_CSR(cummulative_0, self.op0_sum_elem, self.shape0, self.shape1)
        self.factor(t, coeff)
        _zcsr_add_core(cte.data, cte.ind, cte.ptr,
                       op0.data, op0.ind, op0.ptr, coeff[0]
                       cummulative_0, self.shape0, self.shape1)

        _zcsr_add_core(cummulative_0.data, cummulative_0.ind, cummulative_0.ptr,
                       op1.ptr, op1.ptr, op1.data, coeff[1]
                       out, self.shape0, self.shape1)

    #cdef void _c_call(self, double t, complex * out, int * ind, int * ptr):

    def call(self, double t, bool data=False):
        cdef CSR_Matrix * out
        out = new CSR_Matrix()
        init_CSR(out, self.total_elem, self.shape0, self.shape1)
        self._c_call(t, out)
        scipy_obj = CSR_to_scipy(out)
        if data:
            return scipy_obj
        else:
            return qobj(scipy_obj)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        cdef CSR_Matrix * out_mat
        out_mat = new CSR_Matrix()
        init_CSR(out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, out_mat)
        zspmvpy(out_mat.data, out_mat.ind, out_mat.ptr, vec, 1., out, self.shape0)

    def rhs(self, double t, np.ndarray[complex, ndim=1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0)
        self._rhs_mat(t, vec, out)
        return out

    def rhs_ptr():
        void * ptr = <void*>self._rhs_mat
        return PyLong_FromVoidPtr(ptr)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat(self, double t, complex* vec, int herm):
        cdef CSR_Matrix * out_mat
        out_mat = new CSR_Matrix()
        init_CSR(out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, out_mat)
        return cy_expect_psi_csr(out_mat.data, out_mat.ind, out_mat.ptr, vec, herm)

    def expect(self, double t, np.ndarray[complex, ndim=1] vec, int isherm):
        return self._expect_mat(t, vec, isherm)

    def expect_ptr():
        void * ptr = <void*>self._expect_mat
        return PyLong_FromVoidPtr(ptr)
