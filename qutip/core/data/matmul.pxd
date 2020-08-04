#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dense cimport Dense

cpdef CSR matmul_csr(CSR left, CSR right, CSR out=*, double complex scale=*)
cpdef Dense matmul_dense(Dense left, Dense right, Dense out=*, double complex scale=*)
cpdef Dense matmul_csr_dense_dense(CSR left, Dense right, Dense out=*,
                                   double complex scale=*)
