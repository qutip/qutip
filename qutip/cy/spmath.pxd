#cython: language_level=3

from qutip.cy.sparse_structs cimport CSR_Matrix

cdef int _safe_multiply(int a, int b) except? -1

cdef void _zcsr_add(CSR_Matrix * A, CSR_Matrix * B,
                    CSR_Matrix * C, double complex alpha)

cdef int _zcsr_add_core(double complex * Adata, int * Aind, int * Aptr,
                        double complex * Bdata, int * Bind, int * Bptr,
                        double complex alpha,
                        CSR_Matrix * C,
                        int nrows, int ncols) nogil

cdef void _zcsr_mult(CSR_Matrix * A, CSR_Matrix * B, CSR_Matrix * C)


cdef void _zcsr_kron(CSR_Matrix * A, CSR_Matrix * B, CSR_Matrix * C) except *

cdef void _zcsr_kron_core(double complex * dataA, int * indsA, int * indptrA,
                          double complex * dataB, int * indsB, int * indptrB,
                          CSR_Matrix * out,
                          int rowsA, int rowsB, int colsB) nogil

cdef void _zcsr_transpose(CSR_Matrix * A, CSR_Matrix * B)

cdef void _zcsr_adjoint(CSR_Matrix * A, CSR_Matrix * B)
