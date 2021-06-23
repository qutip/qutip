#cython: language_level=3

from qutip.cy.sparse_structs cimport CSR_Matrix, COO_Matrix
from qutip.cy.cqobjevo_factor cimport CoeffFunc

cdef class CQobjEvo:
    cdef int shape0, shape1
    cdef object dims
    cdef int super
    cdef int num_ops
    cdef int dyn_args

    #cdef void (*factor_ptr)(double, complex*)
    cdef object factor_func
    cdef CoeffFunc factor_cobj
    cdef int factor_use_cobj
    # prepared buffer
    cdef complex[::1] coeff
    cdef complex* coeff_ptr

    cdef int _factor(self, double t) except -1
    cdef int _factor_dyn(self, double t, complex* state, int[::1] state) except -1
    cdef int _mul_vec(self, double t, complex* vec, complex* out) except -1
    cdef int _mul_matf(self, double t, complex* mat, complex* out,
                    int nrow, int ncols) except -1
    cdef int _mul_matc(self, double t, complex* mat, complex* out,
                    int nrow, int ncols) except -1

    cpdef complex expect(self, double t, complex[::1] vec)
    cdef complex _expect(self, double t, complex* vec) except *
    cdef complex _expect_super(self, double t, complex* rho) except *
    cdef complex _overlapse(self, double t, complex* oper) except *


cdef class CQobjCte(CQobjEvo):
    cdef int total_elem
    # pointer to data
    cdef CSR_Matrix cte


cdef class CQobjCteDense(CQobjEvo):
    # pointer to data
    cdef complex[:, ::1] cte


cdef class CQobjEvoTd(CQobjEvo):
    cdef long total_elem
    # pointer to data
    cdef CSR_Matrix cte
    cdef CSR_Matrix ** ops
    cdef long[::1] sum_elem
    cdef void _call_core(self, CSR_Matrix * out, complex* coeff)


cdef class CQobjEvoTdDense(CQobjEvo):
    # data as array
    cdef complex[:, ::1] cte
    cdef complex[:, :, ::1] ops

    # prepared buffer
    cdef complex[:, ::1] data_t
    cdef complex* data_ptr

    cdef int _factor(self, double t) except -1
    cdef void _call_core(self, complex[:,::1] out, complex* coeff)


cdef class CQobjEvoTdMatched(CQobjEvo):
    cdef int nnz
    # data as array
    cdef int[::1] indptr
    cdef int[::1] indices
    cdef complex[::1] cte
    cdef complex[:, ::1] ops

    # prepared buffer
    cdef complex[::1] data_t
    cdef complex* data_ptr

    cdef int _factor(self, double t) except -1
    cdef void _call_core(self, complex[::1] out, complex* coeff)
