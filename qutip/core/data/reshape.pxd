#cython: language_level=3

from qutip.core.data.base cimport idxint
from qutip.core.data.csr cimport CSR

cpdef CSR reshape_csr(CSR matrix, idxint n_rows_out, idxint n_cols_out)
cpdef CSR column_stack_csr(CSR matrix)
cpdef CSR column_unstack_csr(CSR matrix, idxint rows)
