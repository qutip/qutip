#cython: language_level=3
from qutip.core.cy.coefficient cimport Coefficient


cdef class RateShiftCoefficient(Coefficient):
    cdef:
        Coefficient [:] rate_coeffs

    cpdef double rate_shift(self, double t) except *
    cpdef Coefficient sqrt_shifted_rate(self, int i)
    cpdef double discrete_martingale(self, double[:] collapse_times, int[:] collapse_idx) except *


cdef class SqrtShiftedRateCoefficient(Coefficient):
    cdef:
        int rate_idx
        RateShiftCoefficient rate_shift
