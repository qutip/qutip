#cython: language_level=3

from qutip.cy.sparse_structs cimport CSR_Matrix

cdef void fdense2D_to_CSR(complex[::1, :] mat, CSR_Matrix * out,
                                unsigned int nrows, unsigned int ncols)
