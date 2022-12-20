"""
Class to represent a stochastic differential equation system.
"""

from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data

cdef class StochasticSystem:
    """

    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def drift(self, t, state):
        return self.b(t, state)

    def diffusion(self, t, state):
        return self.a(t, state)



cdef class StochasticClosedSystem:
    """

    """
    cdef QobjEvo H
    cdef list c_ops
    cdef list cpcd_ops
    cdef object imp
    cdef Data state
    cdef double t

    def __init__(self, H, c_ops, dt, options=None, implicit=False):
        self.H = H
        self.c_ops = c_ops
        self.cpcd_ops = [op + op.dag() for op in c_ops]
        self.dt = dt

    def drift(self, double t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef Data temp, out

        out = self.L.matmul_data(t, state)
        for i in range(len(self.c_ops)):
            c_op = self.cpcd_ops[i]
            e = c_op.expect_data(t, state)
            c_op = self.c_ops[i]
            temp = c_op.matmul_data(t, state)
            out = _data.add(out, state,  -0.125 * e * e)
            out = _data.add(out, temp, 0.5 * e)

    def diffusion(self, double t, Data state):
        cdef int i
        cdef QobjEvo c_op
        out = []
        for i in range(len(self.c_ops)):
            c_op = self.c_ops[i]
            _out = c_op.matmul_data(t, state)
            c_op = self.cpcd_ops[i]
            expect = c_op.expect_data(t, state)
            out.append(_data.add(out, state, -0.5 * expect))
        return out


cdef class StochasticOpenSystem:
    """

    """
    cdef QobjEvo L
    cdef list c_ops
    cdef object imp
    cdef Data state
    cdef double t

    def __init__(self, H, c_ops):
        self.L = H
        self.c_ops = c_ops

    def drift(self, double t, Data state):
        return self.L.matmul_data(t, state)

    def diffusion(self, double t, Data state):
        cdef int i, k
        cdef QobjEvo c_op
        cdef complex expect
        cdef out = []
        for i in range(self.num_ops):
            c_op = self.c_ops[i]
            vec = c_op.matmul_data(t, state)
            expect = _data.trace_oper_ket(vec)
            out.append(_data.add(vec, state, -expect))
        return out
