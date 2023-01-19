#cython: language_level=3
"""
Class to represent a stochastic differential equation system.
"""

from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data
cimport cython
import numpy as np
from qutip.core import spre, spost

__all__ = [
    "StochasticSystem", "StochasticOpenSystem", "StochasticClosedSystem"
]


cdef class _StochasticSystem:
    cdef readonly int num_collapse
    cdef readonly bint issuper
    cdef readonly object dims
    cdef Data state
    cdef double t
    cdef object imp

    def __init__(self, a, b):
        self.d1 = a
        self.d2 = b
        self.num_collapse = 1

    def drift(self, t, state):
        return self.d1(t, state)

    def diffusion(self, t, state):
        return self.d2(t, state)

    cdef void set_state(self, double t, Data state):
        self.t = t
        self.state = state


cdef class GeneralStochasticSystem(_StochasticSystem):
    cdef object d1, d2

    def __init__(self, a, b):
        self.d1 = a
        self.d2 = b
        self.num_collapse = 1

    def drift(self, t, state):
        return self.d1(t, state)

    def diffusion(self, t, state):
        return self.d2(t, state)


cdef class StochasticClosedSystem(StochasticSystem):
    cdef QobjEvo H
    cdef list c_ops
    cdef list cpcd_ops

    def __init__(self, H, c_ops, heterodyne, implicit=False):
        self.H = H
        self.c_ops = c_ops
        self.cpcd_ops = [op + op.dag() for op in c_ops]
        self.num_collapse = len(c_ops)
        self.issuper = False
        self.dims = self.H.dims

    def drift(self, double t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef Data temp, out

        out = self.L.matmul_data(t, state)
        for i in range(self.num_collapse):
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
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            _out = c_op.matmul_data(t, state)
            c_op = self.cpcd_ops[i]
            expect = c_op.expect_data(t, state)
            out.append(_data.add(out, state, -0.5 * expect))
        return out


cdef class StochasticOpenSystem(StochasticSystem):
    cdef QobjEvo L
    cdef list c_ops

    def __init__(self, H, c_ops, heterodyne):
        self.L = H
        if heterodyne:
            self.c_ops = []
            for c in c_ops:
                self.c_ops += [
                    (spre(c) + spost(c.dag())) / np.sqrt(2),
                    (spre(c) - spost(c.dag())) * -1j / np.sqrt(2)
                ]
        else:
            self.c_ops = [spre(op) + spost(op.dag()) for op in c_ops]
        self.num_collapse = len(self.c_ops)
        self.issuper = True
        self.dims = self.L.dims

    def drift(self, double t, Data state):
        return self.L.matmul_data(t, state)

    def diffusion(self, double t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef complex expect
        cdef out = []
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            vec = c_op.matmul_data(t, state)
            expect = _data.trace_oper_ket(vec)
            out.append(_data.add(vec, state, -expect))
        return out
