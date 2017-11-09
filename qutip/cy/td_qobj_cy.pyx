# distutils: language = c++
import numpy as np
cimport numpy as np
import cython
cimport cython
from qutip.qobj import Qobj
from qutip.cy.spmath cimport _zcsr_add_core
from qutip.cy.inter cimport zinterpolate, interpolate
from qutip.cy.spmatfuncs cimport spmvpy
from libc.stdlib cimport malloc, free
cimport libc.math

include "complex_math.pxi"
include "sparse_routines.pxi"


cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)


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
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        pass
    cdef complex _expect_mat(self, double t, complex* vec, int isherm):
        return 0.
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm):
        return 0.


cdef class cy_cte_qobj(cy_qobj):
    def __init__(self):
        pass

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

    def get_ptr(self):
        return PyLong_FromVoidPtr(<void*> self)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1.,
               out, self.shape0)

    def rhs(self, double t, complex[::1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0,
                                                        dtype=complex)
        self._rhs_mat(t, &vec[0], &out[0])
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm):
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
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm):
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

    def expect(self, double t, complex[::1] vec, int isherm):
        if self.super:
            return self._expect_mat_super(t, &vec[0], isherm)
        else:
            return self._expect_mat(t, &vec[0], isherm)


cdef class cy_td_qobj(cy_qobj):
    def __init__(self):
        self.N_ops = 0
        self.ops = <CSR_Matrix**> malloc(0 * sizeof(CSR_Matrix*))

    def __del__(self):
        for i in range(self.N_ops):
            free(self.ops[i])
        free(self.ops)

    def set_data(self, cte, ops):
        cdef int i
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.cte = CSR_from_scipy(cte.data)
        cummulative_op = cte.data
        self.super = cte.issuper

        self.N_ops = len(ops)
        free(self.ops)
        self.ops = <CSR_Matrix**> malloc(self.N_ops * sizeof(CSR_Matrix*))
        self.sum_elem = np.zeros(self.N_ops, dtype=int)
        for i, op in enumerate(ops):
            self.ops[i] = <CSR_Matrix*> malloc(sizeof(CSR_Matrix))
            CSR_from_scipy_inplace(op[0].data, self.ops[i])
            cummulative_op += op[0].data
            self.sum_elem[i] = cummulative_op.data.shape[0]

        self.total_elem = self.sum_elem[self.N_ops-1]

    def set_factor(self, func=None, ptr=False):
        cdef void* f_ptr
        if ptr:
            self.factor_use_ptr = 1
            f_ptr = PyLong_AsVoidPtr(ptr)
            self.factor_ptr = <void(*)(double, complex*)> f_ptr
        else:
            self.factor_use_ptr = 0
            self.factor_func = func

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
        self.ops = <CSR_Matrix**> malloc(self.N_ops * sizeof(CSR_Matrix*))
        for i in range(self.N_ops):
            self.ops[i] = <CSR_Matrix*> malloc(sizeof(CSR_Matrix))
            self.sum_elem[i] = state[9][i]
            shallow_set_state(self.ops[i], state[11][i])

    cdef void factor(self, double t, complex* out):
        cdef int i
        if self.factor_use_ptr:
            self.factor_ptr(t, out)
        else:
            coeff = self.factor_func(t)
            for i in range(self.N_ops):
                out[i] = coeff[i]

    def get_ptr(self):
        return PyLong_FromVoidPtr(<void*> self)

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
            previous = next
            next = CSR_Matrix()

            for i in range(1,self.N_ops-1):
                init_CSR(&next, self.sum_elem[i], self.shape0, self.shape1)
                _zcsr_add_core(previous.data, previous.indices,
                               previous.indptr,
                               self.ops[i].data,
                               self.ops[i].indices,
                               self.ops[i].indptr,
                               coeff[i], &next, self.shape0, self.shape1)
                free_CSR(&previous)
                previous = next
                next = CSR_Matrix()

            _zcsr_add_core(previous.data, previous.indices, previous.indptr,
                           self.ops[self.N_ops-1].data,
                           self.ops[self.N_ops-1].indices,
                           self.ops[self.N_ops-1].indptr,
                           coeff[self.N_ops-1], out, self.shape0, self.shape1)
            free_CSR(&previous)

    def call(self, double t, int data=0):
        cdef CSR_Matrix out
        init_CSR(&out, self.total_elem, self.shape0, self.shape1)
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        self._call_core(t, &out, &coeff[0])
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
    cdef void _rhs_mat_sum(self, double t, complex* vec, complex* out):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        self._call_core(t, &out_mat, &coeff[0])
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr,
               vec, 1., out, self.shape0)
        free_CSR(&out_mat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        cdef int i

        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec,
               1., out, self.shape0)
        for i in range(self.N_ops):
            spmvpy(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                   vec, coeff[i], out, self.shape0)

    def rhs_sum(self, double t, complex[::1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0,
                                                        dtype=complex)
        self._rhs_mat_sum(t, &vec[0], &out[0])
        return out

    def rhs(self, double t, complex[::1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0,
                                                        dtype=complex)
        self._rhs_mat(t, &vec[0], &out[0])
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef complex _expect_psi(self, complex* data, int* idx, int* ptr,
                             complex* vec, int isherm):
        cdef complex [::1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy(data, idx, ptr, vec, 1., &y[0], self.shape0)
        cdef int row, num_rows = self.shape0
        cdef complex dot = 0
        for row from 0 <= row < num_rows:
            dot += conj(vec[row]) * y[row]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_sum1(self, double t, complex* vec, int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        self._call_core(t, &out_mat, &coeff[0])
        cdef complex [::1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy(out_mat.data, out_mat.indices, out_mat.indptr, vec, 1.,
               &y[0], self.shape0)
        cdef int row
        cdef complex dot = 0
        free_CSR(&out_mat)
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row]) * y[row]
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
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        self._call_core(t, &out_mat, &coeff[0])
        expect = self._expect_psi(out_mat.data, out_mat.indices,
                                  out_mat.indptr, vec, isherm)
        free_CSR(&out_mat)
        return expect

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm):
        cdef complex [::1] y = np.zeros(self.shape0, dtype=complex)
        cdef int row
        cdef complex dot = 0
        self._rhs_mat(t, &vec[0], &y[0])
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row]) * y[row]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_last(self, double t, complex* vec, int isherm):
        cdef complex [::1] coeff = np.empty(self.N_ops, dtype=complex)
        self.factor(t, &coeff[0])
        cdef int i
        expect = self._expect_psi(self.cte.data, self.cte.indices,
                                  self.cte.indptr, vec, 0)
        for i in range(self.N_ops):
            expect += self._expect_psi(self.ops[i].data, self.ops[i].indices,
                                       self.ops[i].indptr, vec, 0) * coeff[i]
        if isherm:
            return real(expect)
        else:
            return expect

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef complex _expect_rho(self, complex* data, int* idx, int* ptr,
                             complex* rho_vec, int isherm):
        cdef size_t row
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>np.sqrt(num_rows)
        cdef complex dot = 0.0
        for row from 0 <= row < num_rows by n+1:
            row_start = ptr[row]
            row_end = ptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += data[jj] * rho_vec[idx[jj]]
        if isherm == 0:
            return dot
        else:
            return real(dot)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_super_sum(self, double t, complex* vec,
                                       int isherm):
        cdef CSR_Matrix out_mat
        init_CSR(&out_mat, self.total_elem, self.shape0, self.shape1)
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        self._call_core(t, &out_mat, &coeff[0])
        expect = self._expect_rho(out_mat.data, out_mat.indices,
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_mat_super_last(self, double t, complex* vec,
                                        int isherm):
        cdef np.ndarray[complex, ndim=1] coeff = np.empty(self.N_ops,
                                                          dtype=complex)
        self.factor(t, &coeff[0])
        cdef int i
        expect = self._expect_rho(self.cte.data, self.cte.indices,
                                  self.cte.indptr, vec, 0)
        for i in range(self.N_ops):
            expect += self._expect_rho(self.ops[i].data, self.ops[i].indices,
                                       self.ops[i].indptr, vec, 0) * coeff[i]
        if isherm:
            return real(expect)
        else:
            return expect

    def expect(self, double t, complex[::1] vec, int isherm):
        if self.super:
            return self._expect_mat_super(t, &vec[0], isherm)
        else:
            return self._expect_mat(t, &vec[0], isherm)

    def expect_all(self, double t, complex[::1] vec, int isherm, type=1):
        if self.super:
            if type == 1:
                return self._expect_mat_super(t, &vec[0], isherm)
            elif type == 2:
                return self._expect_mat_super_last(t, &vec[0], isherm)
            else:
                return self._expect_mat_super_sum(t, &vec[0], isherm)
        else:
            if type == 1:
                return self._expect_mat(t, &vec[0], isherm)
            elif type == 2:
                return self._expect_mat_sum1(t, &vec[0], isherm)
            elif type == 3:
                return self._expect_mat_sum2(t, &vec[0], isherm)
            else:
                return self._expect_mat_last(t, &vec[0], isherm)
