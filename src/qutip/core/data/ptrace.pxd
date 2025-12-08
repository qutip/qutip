#cython: language_level=3

from qutip.core.data cimport CSR, Dense

cpdef CSR ptrace_csr(CSR matrix, object dims, object sel)
cpdef Dense ptrace_dense(Dense matrix, object dims, object sel)
cpdef Dense ptrace_csr_dense(CSR matrix, object dims, object sel)
