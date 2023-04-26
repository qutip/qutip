#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.dia cimport Diag

cpdef CSR adjoint_csr(CSR matrix)
cpdef CSR transpose_csr(CSR matrix)
cpdef CSR conj_csr(CSR matrix)

cpdef Dense adjoint_dense(Dense matrix)
cpdef Dense transpose_dense(Dense matrix)
cpdef Dense conj_dense(Dense matrix)

cpdef Diag adjoint_diag(Diag matrix)
cpdef Diag transpose_diag(Diag matrix)
cpdef Diag conj_diag(Diag matrix)
