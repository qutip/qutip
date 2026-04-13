#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Dia

cpdef double complex mean_csr(CSR matrix, double atol=*) noexcept
cpdef double complex mean_dia(Dia matrix, double atol=*) noexcept
cpdef double complex mean_dense(Dense matrix, double atol=*) noexcept

cpdef double mean_abs_csr(CSR matrix, double atol=*) noexcept
cpdef double mean_abs_dia(Dia matrix, double atol=*) noexcept
cpdef double mean_abs_dense(Dense matrix, double atol=*) noexcept
