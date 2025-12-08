#cython: language_level=3

from qutip.core.data.csr cimport CSR

cpdef CSR project_csr(CSR state)
