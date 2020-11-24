#cython: language_level=3

from qutip.core.data.base cimport idxint
from qutip.core.data cimport CSR, Dense, CSC

cpdef CSR reshape_csr(CSR matrix, idxint n_rows_out, idxint n_cols_out)
cpdef CSR column_stack_csr(CSR matrix)
cpdef CSR column_unstack_csr(CSR matrix, idxint rows)

cpdef CSC reshape_csc(CSC matrix, idxint n_rows_out, idxint n_cols_out)
cpdef CSC column_stack_csc(CSC matrix)
cpdef CSC column_unstack_csc(CSC matrix, idxint rows)

cpdef Dense reshape_dense(Dense matrix, idxint n_rows_out, idxint n_cols_out)
cpdef Dense column_stack_dense(Dense matrix, bint inplace=*)
cpdef Dense column_unstack_dense(Dense matrix, idxint rows, bint inplace=*)
