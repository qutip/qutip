# distutils: language = c++
import numpy as np
cimport numpy as np
import cython
cimport cython
from qutip.qobj import Qobj
from qutip.cy.spmath cimport _zcsr_add_core
from qutip.cy.inter cimport zinterpolate, interpolate
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.spmatfuncs cimport cy_expect_rho_vec_csr, cy_expect_psi_csr
from libc.stdlib cimport malloc, free

include "complex_math.pxi"
include "sparse_routines.pxi"

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef class cy_compiled_td_qobj:
    cdef int total_elem
    cdef int shape0, shape1
    cdef int super
    cdef void(factor_ptr*)(double, complex*)
    cdef object factor_func
    cdef int factor_use_ptr

    #pointer to data
    cdef CSR_Matrix cte

    cdef void ** ops
    cdef object op_list
    cdef int[::1] sum_elem
    cdef int N_ops

    #cdef CSR_Matrix op0
    #cdef int op0_sum_elem
    #cdef CSR_Matrix op1
    #cdef int op1_sum_elem
    #args
    cdef double dt dt_times
    cdef int N N_times
    cdef double w
    cdef double * str_array_0

    def __init__(self):
        self.ops = malloc(0 * sizeof(void*))

    def __del__(self):
        free(self.ops)

    def set_data(self, cte, ops):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.cte = CSR_from_scipy(cte.data)
        cummulative_op = cte.data
        self.super = cte.issuper

        self.N_ops = len(ops)
        free(self.ops)
        self.ops = malloc(self.N_ops * sizeof(void*))
        self.sum_elem = np.zeros(self.N_ops, dtype=int)
        self.op_list = []

        for op in ops:
            cummulative_op += ops[0].data
            self.op_list.append(CSR_from_scipy(ops[0][0].data))
            self.sum_elem.append(cummulative_op.data.shape[0])
            self.ops = <void*>self.op_list[-1]

        self.total_elem = self.sum_elem[self.N_ops-1]

    def set_factor_f(self, func, ptr=False):
        if ptr:
            self.factor_use_ptr = 1
            cdef void* f_ptr = PyLong_AsVoidPtr(func)
            self.factor_ptr = <void(*)(double, complex*)> f_ptr
        else:
            self.factor_use_ptr = 0
            self.factor_func = func

    cdef void factor(self, double t, complex* out):
        if self.factor_use_ptr:
            self.factor_ptr(t,out)
        else:
            coeff = self.factor_func(t)
            for i in range(self.N_ops):
                out[i] = coeff[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, double t, CSR_Matrix * out):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        cdef int i
        cdef CSR_Matrix previous, now, next

        if(self.N_ops ==1):
            now = <CSR_Matrix> self.ops[0]
            _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                         now.data, now.indices, now.indptr,
                         coeff[0], &out, self.shape0, self.shape1)
        else:
            #Ugly with a loop for 1 to N-2...
            #It save the copy of cte and out
            previous = self.cte
            init_CSR(&next, self.sum_elem[0], self.shape0, self.shape1)
            now = <CSR_Matrix> self.ops[0]
            _zcsr_add_core(previous.data, previous.indices, previous.indptr,
                           now.data, now.indices, now.indptr,
                           coeff[0], &next, self.shape0, self.shape1)
            previous = next
            next = CSR_Matrix()

            for i in range(1,self.N_ops-1):
                init_CSR(&next, self.sum_elem[i], self.shape0, self.shape1)
                now = <CSR_Matrix> self.ops[i]
                _zcsr_add_core(previous.data, previous.indices, previous.indptr,
                               now.data, now.indices, now.indptr,
                               coeff[i], &next, self.shape0, self.shape1)
                free_CSR(&previous)
                previous = next
                next = CSR_Matrix()

            now = <CSR_Matrix> self.ops[self.N_ops-1]
            _zcsr_add_core(previous.data, previous.indices, previous.indptr,
                           now.data, now.indices, now.indptr,
                           coeff[self.N_ops-1], &out, self.shape0, self.shape1)
            free_CSR(&previous)

    def call(self, double t, int data=0):
        cdef CSR_Matrix out
        init_CSR(&out, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out)
        scipy_obj = CSR_to_scipy(&out)
        #free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat_sum(self, double t, complex* vec, complex* out):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr, vec, 1., out, self.shape0)
        free_CSR(&out_mat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        cdef int i
        cdef CSR_Matrix now
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec,
               1., out, self.shape0)
        for i in range(self.N_ops):
            now = <CSR_Matrix> self.ops[i]
            spmvpy(now.data, now.indices, now.indptr, vec,
                   coeff[i], out, self.shape0)


    def rhs_sum(self, double t, complex[::1] vec):
        cdef complex[::1] out = np.zeros(self.shape0, dtype=complex)
        self._rhs_mat_sum(t, &vec[0], &out[0])
        return out

    def rhs(self, double t, complex[::1] vec):
        cdef complex[::1] out = np.zeros(self.shape0, dtype=complex)
        self._rhs_mat(t, &vec[0], &out[0])
        return out


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_sum1(self, double t, complex* vec, int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)
        cdef np.ndarray[complex, ndim=1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr, vec, 1., &y[0], self.shape0)
        cdef int row
        cdef complex dot = 0
        free_CSR(&out_mat)
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row])*y[row]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_sum2(self, double t, complex* vec, int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)
        expect = cy_expect_psi_csr(out_mat.data, out_mat.indices, out_mat.indptr, vec, isherm)
        free_CSR(&out_mat)
        return expect

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm):
        cdef np.ndarray[complex, ndim=1] y = np.zeros(self.shape0, dtype=complex)
        self._rhs_mat(t, &vec[0], &y[0])
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
    cdef complex _expect_mat_last(self, double t, complex* vec, int isherm):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        cdef int i
        cdef CSR_Matrix now
        expect = cy_expect_psi_csr(self.cte.data, self.cte.indices,
                                   self.cte.indptr, vec, 0)
        for i in range(self.N_ops):
            now = <CSR_Matrix> self.ops[i]
            expect += cy_expect_psi_csr(now.data, now.indices,
                                        now.indptr, vec, 0) * coeff[i]
        if isherm:
            return real(expect)
        else:
            return expect


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_super_sum(self, double t, complex* vec, int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        self._call_core(t, &out_mat)

        expect = cy_expect_rho_vec_csr(out_mat.data, out_mat.indices,
                                   out_mat.indptr, vec, isherm)
        free_CSR(&out_mat)
        return expect

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm):
        cdef int row, i
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>np.sqrt(num_rows)
        cdef complex dot = 0.0

        for row from 0 <= row < num_rows by n+1:
            row_start = self.cte.indptr[row]
            row_end = self.cte.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.cte.data[jj]*vec[self.cte.indices[jj]]
        for i in range(self.N_ops):
            now = <CSR_Matrix> self.ops[i]
            for row from 0 <= row < num_rows by n+1:
                row_start = now.indptr[row]
                row_end = now.indptr[row+1]
                for jj from row_start <= jj < row_end:
                    dot += now.data[jj]*vec[now.indices[jj]]*coeff[i]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_super_last(self, double t, complex* vec, int isherm):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        cdef int i
        cdef CSR_Matrix now
        expect = cy_expect_rho_vec_csr(self.cte.data, self.cte.indices,
                                   self.cte.indptr, vec, 0)
        for i in range(self.N_ops):
            now = <CSR_Matrix> self.ops[i]
            expect += cy_expect_rho_vec_csr(now.data, now.indices,
                                        now.indptr, vec, 0) * coeff[i]
        if isherm:
            return real(expect)
        else:
            return expect

    def expect(self, double t, complex[::1] vec, int isherm, type = 1):
        if self.super:
          if type == 1:
            return self._expect_mat_super_sum(t, &vec[0], isherm)
          elif type == 2:
            return self._expect_mat_super(t, &vec[0], isherm)
          else:
            return self._expect_mat_super_last(t, &vec[0], isherm)
        else:
          if type == 1:
            return self._expect_mat(t, &vec[0], isherm)
          elif type == 2:
            return self._expect_mat_sum1(t, &vec[0], isherm)
          elif type == 3:
            return self._expect_mat_sum2(t, &vec[0], isherm)
          else:
            return self._expect_mat_last(t, &vec[0], isherm)
