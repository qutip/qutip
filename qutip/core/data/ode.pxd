#cython: language_level=3

from qutip.core.data cimport Data, Dense, CSR, Dia


cpdef double wrmn_error_dense(Dense diff, Dense state, double atol, double rtol) except -1

cpdef double wrmn_error_csr(CSR diff, CSR state, double atol, double rtol) except -1

cpdef double wrmn_error_dia(Dia diff, Dia state, double atol, double rtol) except -1

cdef double cy_wrmn_error(Data diff, Data state, double atol, double rtol) except -1
