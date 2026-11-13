from qutip.core.data cimport Data, Dense
from qutip.core.cy.qobjevo cimport QobjEvo

cdef class BaseStochasticSystem:
    cdef public int num_diffusion

    cpdef Data drift(self, t, Data state)
    cpdef list diffusion(self, t, Data state)
    cpdef list _shift(self, t, Data state)

cdef class TaylorStochasticSystem(BaseStochasticSystem):
    cdef public Data state
    cdef public double t
    cpdef void set_state(self, double t, Data state) except *

    cpdef Data a(self)
    cpdef Data bi(self, int i)
    cpdef Data Libj(self, int i, int j)
    cpdef Data Lia(self, int i)
    cpdef Data L0bi(self, int i)
    cpdef Data LiLjbk(self, int i, int j, int k)
    cpdef Data L0a(self)

    cpdef complex _shift_i(self, int i)
