#cython: language_level=3
from qutip.core.cy.coefficient cimport Coefficient


cdef class RateSet:
    cdef:
        Coefficient [:] rate_coeffs

    cpdef double rate_shift(RateSet self, double t) except *
    cpdef double rate(RateSet self, double t, int i) except *


cdef class SqrtShiftedRateCoefficient(Coefficient):
    cdef:
        int rate_idx
        RateSet rates
