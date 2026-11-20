#cython: language_level=3
from qutip.core.data cimport Data
from qutip.core.cy.qobjevo cimport QobjEvo


cdef class RHS:
    cdef object derivative
    cdef QobjEvo qevo
    cdef bint inplace, qevo_derr
    cdef Data apply(self, double t, Data state, Data out=*)
