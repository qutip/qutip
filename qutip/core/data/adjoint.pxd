#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.dia cimport Dia
from qutip.core.data.coo cimport COO

cpdef COO adjoint_coo(COO matrix)
cpdef COO transpose_coo(COO matrix)
cpdef COO conj_coo(COO matrix)

cpdef CSR adjoint_csr(CSR matrix)
cpdef CSR transpose_csr(CSR matrix)
cpdef CSR conj_csr(CSR matrix)

cpdef Dense adjoint_dense(Dense matrix)
cpdef Dense transpose_dense(Dense matrix)
cpdef Dense conj_dense(Dense matrix)

cpdef Dia adjoint_dia(Dia matrix)
cpdef Dia transpose_dia(Dia matrix)
cpdef Dia conj_dia(Dia matrix)
