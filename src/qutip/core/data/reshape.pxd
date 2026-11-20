#cython: language_level=3

from qutip.core.data.base cimport idxint
from qutip.core.data cimport CSR, Dense, Dia

cpdef CSR reshape_csr(CSR matrix, idxint n_rows_out, idxint n_cols_out)
cpdef CSR column_stack_csr(CSR matrix)
cpdef CSR column_unstack_csr(CSR matrix, idxint rows)

cpdef Dia reshape_dia(Dia matrix, idxint n_rows_out, idxint n_cols_out)
cpdef Dia column_stack_dia(Dia matrix)
cpdef Dia column_unstack_dia(Dia matrix, idxint rows)

cpdef Dense reshape_dense(Dense matrix, idxint n_rows_out, idxint n_cols_out)
cpdef Dense column_stack_dense(Dense matrix, bint inplace=*)
cpdef Dense column_unstack_dense(Dense matrix, idxint rows, bint inplace=*)
