#cython: language_level=3
from qutip.core.cy.coefficient cimport Coefficient


cdef class RateShiftCoefficient(Coefficient):
    cdef:
        Coefficient [:] coeffs


cdef class SqrtRealCoefficient(Coefficient):
    cdef:
        Coefficient base
