#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense
from qutip.core.data.csc cimport CSC

cpdef CSR adjoint_csr(CSR matrix)
cpdef CSR transpose_csr(CSR matrix)
cpdef CSR conj_csr(CSR matrix)

cpdef Dense adjoint_dense(Dense matrix)
cpdef Dense transpose_dense(Dense matrix)
cpdef Dense conj_dense(Dense matrix)

cpdef CSC adjoint_csc(CSC matrix)
cpdef CSC transpose_csc(CSC matrix)
cpdef CSC conj_csc(CSC matrix)
