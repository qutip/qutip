#cython: language_level=3
from qutip.core.cy.coefficient cimport Coefficient


cdef class RateShiftCoefficient(Coefficient):
    cdef:
        Coefficient [:] coeffs

    cpdef double as_double(self, double t) except *


cdef class SqrtRealCoefficient(Coefficient):
    cdef:
        Coefficient base
