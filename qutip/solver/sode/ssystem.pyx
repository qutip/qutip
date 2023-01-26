#cython: language_level=3
"""
Class to represent a stochastic differential equation system.
"""

from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data, dense, Dense
cimport cython
import numpy as np
from qutip.core import spre, spost, liouvillian

__all__ = [
    "GeneralStochasticSystem", "StochasticOpenSystem", "StochasticClosedSystem"
]

@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef Dense _dense_wrap(double complex [::1] x):
    return dense.wrap(&x[0], x.shape[0], 1)


cdef class _StochasticSystem:
    def __init__(self, a, b):
        self.d1 = a
        self.d2 = b
        self.num_collapse = 1
        self.is_set = False

    cpdef Data drift(self, t, Data state):
        raise NotImplementedError

    cpdef list diffusion(self, t, Data state):
        raise NotImplementedError

    cpdef void set_state(self, double t, Data state):
        self.t = t
        self.state = state
        self.is_set = True

    cpdef Data a(self):
        """
          Drift term
        """
        raise NotImplementedError

    cpdef Data bi(self, int i):
        """
          Diffusion term for the ``i``th operator.
        """
        raise NotImplementedError

    cpdef Data Libj(self, int i, int j):
        """
            bi_n * d bj / dx_n
        """
        raise NotImplementedError

    cpdef Data Lia(self, int i):
        """
            bi_n * d a / dx_n
        """
        raise NotImplementedError

    cpdef Data L0bi(self, int i):
        """
            d/dt + a_n * d bi / dx_n + sum_k bk_n bk_m *0.5 d**2 (bi) / (dx_n dx_m)
        """
        raise NotImplementedError

    cpdef Data LiLjbk(self, int i, int j, int k):
        """
            bi_n * d/dx_n ( bj_m * d bk / dx_m)
        """
        raise NotImplementedError

    cpdef Data L0a(self):
        """
            d/dt + a_n * d a / dx_n + sum_k bk_n bk_m *0.5 d**2 (a) / (dx_n dx_m)
        """
        raise NotImplementedError


cdef class GeneralStochasticSystem(_StochasticSystem):
    cdef object d1, d2

    def __init__(self, a, b):
        self.d1 = a
        self.d2 = b
        self.num_collapse = 1

    cpdef Data drift(self, t, Data state):
        return self.d1(t, state)

    cpdef list diffusion(self, t, Data state):
        return self.d2(t, state)


cdef class StochasticClosedSystem(_StochasticSystem):
    cdef QobjEvo H
    cdef list c_ops
    cdef list cpcd_ops

    def __init__(self, H, c_ops, heterodyne):
        self.H = -1j * H
        if heterodyne:
            self.c_ops = []
            for c_op in c_ops:
                self.c_ops.append(c_op / np.sqrt(2))
                self.c_ops.append(c_op * (-1j / np.sqrt(2)))
                self.cpcd_ops.append((c_op + c_op.dag()) / np.sqrt(2))
                self.cpcd_ops.append((-c_op + c_op.dag()) * 1j / np.sqrt(2))
        else:
            self.c_ops = c_ops
            self.cpcd_ops = [op + op.dag() for op in c_ops]

        self.num_collapse = len(self.c_ops)
        for c_op in self.c_ops:
            self.H += -0.5 * c_op.dag() * c_op
        self.issuper = False
        self.dims = self.H.dims

    cpdef Data drift(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef Data temp, out

        out = self.H.matmul_data(t, state)
        for i in range(self.num_collapse):
            c_op = self.cpcd_ops[i]
            e = c_op.expect_data(t, state)
            c_op = self.c_ops[i]
            temp = c_op.matmul_data(t, state)
            out = _data.add(out, state,  -0.125 * e * e)
            out = _data.add(out, temp, 0.5 * e)
        return out

    cpdef list diffusion(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        out = []
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            _out = c_op.matmul_data(t, state)
            c_op = self.cpcd_ops[i]
            expect = c_op.expect_data(t, state)
            out.append(_data.add(_out, state, -0.5 * expect))
        return out


cdef class StochasticOpenSystem(_StochasticSystem):
    cdef QobjEvo L
    cdef list c_ops
    cdef int state_size
    cdef double dt

    cdef readonly Dense _a, temp, _L0a
    cdef readonly complex[::1] expect_Cv
    cdef readonly complex[:, ::1] expect_Cb, _b, _La, _L0b
    cdef readonly complex[:, :, ::1] _Lb
    cdef readonly complex[:, :, :, ::1] _LLb

    def __init__(self, H, c_ops, heterodyne):
        self.L = H + liouvillian(None, c_ops)
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
        self.state_size = self.L.shape[1]
        self.is_set = 0

    cpdef Data drift(self, t, Data state):
        return self.L.matmul_data(t, state)

    cpdef list diffusion(self, t, Data state):
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

    cpdef void set_state(self, double t, Data state):
        cdef n, l
        self.t = t
        self.state = _data.to(_data.Dense, state)

        if not self.is_set:
            n = self.num_collapse
            l = self.state_size
            self.is_set = 1
            self._a = dense.zeros(self.state_size, 1)
            self.temp = dense.zeros(self.state_size, 1)
            self._L0a = dense.zeros(self.state_size, 1)
            self.expect_Cv = np.zeros(n, dtype=complex)
            self.expect_Cb = np.zeros((n, n), dtype=complex)
            self._b = np.zeros((n, l), dtype=complex)
            self._L0b = np.zeros((n, l), dtype=complex)
            self._Lb = np.zeros((n, n, l), dtype=complex)
            self._LLb = np.zeros((n, n, n, l), dtype=complex)
            self._La = np.zeros((n, l), dtype=complex)
            self.dt = 1e-6  #  Make an options

    cpdef Data a(self):

        self.is_set = 2
        _data.imul_dense(self._a, 0)
        self.L.matmul_data(self.t, self.state, self._a)
        return self._a

    cpdef Data bi(self, int n):
        cdef int i
        cdef QobjEvo c_op
        cdef Dense b_vec

        for i in range(self.num_collapse):
            c_op = <QobjEvo> self.c_ops[i]
            b_vec = <Dense> _dense_wrap(self._b[i, :])
            _data.imul_dense(b_vec, 0)
            c_op.matmul_data(self.t, self.state, b_vec)
            self.expect_Cv[i] = _data.trace_oper_ket(b_vec)
            _data.iadd_dense(b_vec, self.state, -self.expect_Cv[i])

        return _dense_wrap(self._b[n, :])

    cpdef Data Libj(self, int n, int m):
        cdef int i, j
        cdef QobjEvo c_op
        cdef Dense b_vec, Lb_vec
        cdef complex expect

        for i in range(self.num_collapse):
            c_op = <QobjEvo> self.c_ops[i]
            for j in range(i, self.num_collapse):
                b_vec = <Dense> _dense_wrap(self._b[j, :])
                Lb_vec = <Dense> _dense_wrap(self._Lb[i, j, :])
                _data.imul_dense(Lb_vec, 0)
                c_op.matmul_data(self.t, b_vec, Lb_vec)
                self.expect_Cb[i,j] = _data.trace_oper_ket(Lb_vec)
                _data.iadd_dense(Lb_vec, b_vec, -self.expect_Cv[i])
                _data.iadd_dense(Lb_vec, self.state, -self.expect_Cb[i,j])

        if m >= n:
            # We only support commutative diffusion
            return _dense_wrap(self._Lb[n, m, :])
        return _dense_wrap(self._Lb[m, n, :])

    cpdef Data Lia(self, int n):
        cdef int i
        cdef QobjEvo c_op
        cdef Dense b_vec, La_vec

        for i in range(self.num_collapse):
            b_vec = <Dense> _dense_wrap(self._b[i, :])
            La_vec = <Dense> _dense_wrap(self._La[i, :])
            _data.imul_dense(La_vec, 0.)
            self.L.matmul_data(self.t, b_vec, La_vec)

        return _dense_wrap(self._La[n, :])

    cpdef Data L0bi(self, int n):
        # L0bi = abi' + dbi/dt + Sum_j bjbjbi"/2
        cdef int i, j
        cdef QobjEvo c_op
        cdef Dense b_vec, L0b_vec, a

        for i in range(self.num_collapse):
            c_op = <QobjEvo> self.c_ops[i]
            L0b_vec = <Dense> _dense_wrap(self._L0b[i, :])
            b_vec = <Dense> _dense_wrap(self._b[i, :])
            _data.imul_dense(L0b_vec, 0.)

            # db/dt
            c_op.matmul_data(self.t + self.dt, self.state, L0b_vec)
            expect = _data.trace_oper_ket(L0b_vec)
            _data.iadd_dense(L0b_vec, self.state, -expect)
            _data.iadd_dense(L0b_vec, b_vec, -1)
            _data.imul_dense(L0b_vec, 1/self.dt)

            # ab'
            _data.imul_dense(self.temp, 0)
            c_op.matmul_data(self.t, self._a, self.temp)
            expect = _data.trace_oper_ket(self.temp)
            _data.iadd_dense(L0b_vec, self.temp, 1)
            _data.iadd_dense(L0b_vec, self._a, -self.expect_Cv[i])
            _data.iadd_dense(L0b_vec, self.state, -expect)

            # bbb" : expect_Cb[i,j] only defined for j>=i
            for j in range(i):
                b_vec = <Dense> _dense_wrap(self._b[j, :])
                _data.iadd_dense(L0b_vec, b_vec, -self.expect_Cb[j,i])
            for j in range(i, self.num_collapse):
                b_vec = <Dense> _dense_wrap(self._b[j, :])
                _data.iadd_dense(L0b_vec, b_vec, -self.expect_Cb[i,j])

        return _dense_wrap(self._L0b[n, :])

    cpdef Data LiLjbk(self, int n, int m, int o):
        # LiLjbk = bi(bj'bk'+bjbk"), i<=j<=k
        # sc_ops must commute (LiLjbk = LjLibk = LkLjbi)
        cdef int i, j, k
        cdef QobjEvo c_op
        cdef Dense bj_vec, bk_vec, LLb_vec, Lb_vec

        for i in range(self.num_collapse):
          for j in range(i,self.num_collapse):
            for k in range(j,self.num_collapse):

                c_op = <QobjEvo> self.c_ops[i]
                LLb_vec = <Dense> _dense_wrap(self._LLb[i, j, k, :])
                Lb_vec = <Dense> _dense_wrap(self._Lb[j, k, :])
                bj_vec = <Dense> _dense_wrap(self._b[j, :])
                bk_vec = <Dense> _dense_wrap(self._b[k, :])
                _data.imul_dense(LLb_vec, 0.)

                c_op.matmul_data(self.t, Lb_vec, LLb_vec)
                expect = _data.trace_oper_ket(LLb_vec)

                _data.iadd_dense(LLb_vec, Lb_vec, -self.expect_Cv[i])
                _data.iadd_dense(LLb_vec, self.state, -expect)
                _data.iadd_dense(LLb_vec, bj_vec, -self.expect_Cb[i,k])
                _data.iadd_dense(LLb_vec, bk_vec, -self.expect_Cb[i,j])

        return _dense_wrap(self._LLb[n, m, o, :])

    cpdef Data L0a(self):
        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        _data.imul_dense(self._L0a, 0.)
        self.L.matmul_data(self.t + self.dt, self.state, self._L0a)
        _data.iadd_dense(self._L0a, self._a, -1)
        _data.imul_dense(self._L0a, 1/self.dt)
        self.L.matmul_data(self.t, self._a, self._L0a)
        return self._L0a
