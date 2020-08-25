#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense

cpdef CSR matmul_csr(CSR left, CSR right, double complex scale=*, CSR out=*)
cpdef Dense matmul_dense(Dense left, Dense right, double complex scale=*, Dense out=*)
cpdef Dense matmul_csr_dense_dense(CSR left, Dense right, double complex scale=*, Dense out=*)
