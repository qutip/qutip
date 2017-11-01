from qutip.cy.sparse_structs cimport CSR_Matrix, COO_Matrix


cdef class cy_qobj:
    cdef void _rhs_mat(self, double t, complex* vec, complex* out)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm)


cdef class cy_cte_qobj(cy_qobj):
    cdef int total_elem
    cdef int shape0, shape1
    cdef object dims
    cdef int super

    # pointer to data
    cdef CSR_Matrix cte

    cdef void _rhs_mat(self, double t, complex* vec, complex* out)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm)


cdef class cy_td_qobj(cy_qobj):
    cdef long total_elem
    cdef int shape0, shape1
    cdef object dims
    cdef int super
    cdef void (*factor_ptr)(double, complex*)
    cdef object factor_func
    cdef int factor_use_ptr

    # pointer to data
    cdef CSR_Matrix cte
    cdef CSR_Matrix ** ops
    cdef long[::1] sum_elem
    cdef int N_ops

    cdef void factor(self, double t, complex* out)
    cdef void _call_core(self, double t, CSR_Matrix * out, complex* coeff)
    cdef void _rhs_mat_sum(self, double t, complex* vec, complex* out)
    cdef void _rhs_mat(self, double t, complex* vec, complex* out)
    cdef complex _expect_psi(self, complex* data, int* idx, int* ptr,
                             complex* vec, int isherm)
    cdef complex _expect_mat_sum1(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_sum2(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_last(self, double t, complex* vec, int isherm)
    cdef complex _expect_rho(self, complex* data, int* idx, int* ptr,
                             complex* rho_vec, int isherm)
    cdef complex _expect_mat_super_sum(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_super(self, double t, complex* vec, int isherm)
    cdef complex _expect_mat_super_last(self, double t,
                                        complex* vec, int isherm)
