#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.csc cimport CSC
from qutip.core.data.dense cimport Dense
from qutip.core.data.base cimport Data

cpdef CSR matmul_csr(CSR left, CSR right, double complex scale=*, CSR out=*)
cpdef CSC matmul_csc(CSC left, CSC right, double complex scale=*, CSC out=*)
cpdef Dense matmul_dense(Dense left, Dense right, double complex scale=*, Dense out=*)

cpdef Dense matmul_csc_dense_dense(CSC left, Dense right, double complex scale=*, Dense out=*)
cdef Dense matmul_data_dense(Data left, Dense right)
cdef void imatmul_data_dense(Data left, Dense right, double complex scale, Dense out)
