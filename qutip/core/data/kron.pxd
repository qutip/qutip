#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.dia cimport Dia
from qutip.core.data.base cimport Data

cpdef Dense kron_dense(Dense left, Dense right)
cpdef CSR kron_csr(CSR left, CSR right)
cpdef Dia kron_dia(Dia left, Dia right)
cpdef CSR kron_dense_csr_csr(Dense left, CSR right)
cpdef CSR kron_csr_dense_csr(CSR left, Dense right)
cpdef Dia kron_dia_dense_dia(Dia left, Dense right)
cpdef Dia kron_dense_dia_dia(Dense left, Dia right)

cpdef Data kron_transpose_data(Data left, Data right)
cpdef Dense kron_transpose_dense(Dense left, Dense right)
