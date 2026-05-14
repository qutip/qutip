#cython: language_level=3
"""
Class to represent a stochastic differential equation system.
"""

from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data, dense, Dense, imul_dense, iadd_dense
from qutip.core.data.trace cimport trace_oper_ket_dense
cimport cython
import numpy as np
from qutip.core import spre, spost, liouvillian

__all__ = [
    "StochasticOpenSystem", "StochasticClosedSystem"
]

@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef Dense _dense_wrap(double complex [::1] x):
    return dense.wrap(&x[0], x.shape[0], 1)


cdef class _StochasticSystem:
    """
        RHS for stochastic differential equations.

        Contain the deterministic drift term and the diffusion term[s] through
        the ``drift`` and ``diffusion`` methods.

        Derrivatives corresponding to the terms in the ito-tyalor expansion are
        available through a different interface:
        ``set_state``, ``a``, ``bi``, ``Libj`` etc.

        A different interface is used since each term is not independant, but
        which terms is needed change according to the integration method.
        Whereas the raw drift and diffusion are independant.
    """
    def __init__(self):
        raise NotImplementedError

    cpdef Data drift(self, t, Data state):
        """
            Compute the drift term for the ``state`` at time ``t``.
        """
        raise NotImplementedError

    cpdef list diffusion(self, t, Data state):
        """
            Compute the diffusion terms for the ``state`` at time ``t``.
        """
        raise NotImplementedError

    cpdef list expect(self, t, Data state):
        """
            Compute the expectation terms for the ``state`` at time ``t``.
        """
        raise NotImplementedError

    cpdef void set_state(self, double t, Data state) except *:
        """
            Initialize the set of derrivatives.
        """
        raise NotImplementedError

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

    cpdef complex expect_i(self, int i):
        """
          Expectation value of the ``i``th operator.
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
            dbi/dt
            + a_n * d bi / dx_n
            + sum_k bk_n bk_m *0.5 d**2 (bi) / (dx_n dx_m)
        """
        raise NotImplementedError

    cpdef Data LiLjbk(self, int i, int j, int k):
        """
            bi_n * d/dx_n ( bj_m * d bk / dx_m)
        """
        raise NotImplementedError

    cpdef Data L0a(self):
        """
            da/dt
            + a_n * d a / dx_n
            + sum_k bk_n bk_m *0.5 d**2 (a) / (dx_n dx_m)
        """
        raise NotImplementedError


cdef class StochasticClosedSystem(_StochasticSystem):
    """
        RHS for closed quantum stochastic system (ssesolve)

        drift = -1H * psi
              + sum_i (-c_i.dag * c_i / 2 + c_i * e_i / 2 - e_i**2 / 8) * psi

        e_i = <psi| c_i + c_i.dag |psi>

        diffusion = (c_i - e_i / 2) * psi
    """
    cdef readonly list cpcd_ops

    def __init__(self, H, sc_ops):
        self.L = -1j * H
        self.c_ops = sc_ops
        self.cpcd_ops = [op + op.dag() for op in sc_ops]

        self.num_collapse = len(self.c_ops)
        for c_op in self.c_ops:
            self.L += -0.5 * c_op.dag() * c_op

    cpdef Data drift(self, t, Data state):
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
        return out

    cpdef list diffusion(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef list out = []
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            _out = c_op.matmul_data(t, state)
            c_op = self.cpcd_ops[i]
            expect = c_op.expect_data(t, state)
            out.append(_data.add(_out, state, -0.5 * expect))
        return out

    cpdef list expect(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef list expect = []
        for i in range(self.num_collapse):
            c_op = self.cpcd_ops[i]
            expect.append(c_op.expect_data(t, state))
        return expect

    def __reduce__(self):
        return (
            StochasticClosedSystem.restore,
            (self.L, self.c_ops, self.cpcd_ops)
        )

    @classmethod
    def restore(cls, L, c_ops, cpcd_ops):
        cdef StochasticClosedSystem out = cls.__new__(cls)
        out.L = L
        out.c_ops = c_ops
        out.cpcd_ops = cpcd_ops
        out.num_collapse = len(c_ops)
        return out


cdef class StochasticOpenSystem(_StochasticSystem):
    """
        RHS for open quantum stochastic system (smesolve)

        drift = liouvillian(H, sc_ops + c_ops)(rho)

        diffusion = c_i @ rho + rho @ c_i/dag - tr(c_i @ rho) rho
    """
    cdef int state_size, N_root
    cdef double dt
    cdef int _is_set
    cdef bint _a_set, _b_set, _Lb_set, _L0b_set, _La_set, _LLb_set, _L0a_set

    cdef Dense _a, temp, _L0a
    cdef complex[::1] expect_Cv
    cdef complex[:, ::1] expect_Cb, _b, _La, _L0b
    cdef complex[:, :, ::1] _Lb
    cdef complex[:, :, :, ::1] _LLb

    def __init__(self, H, sc_ops, c_ops=(), derr_dt=1e-6):
        if H.issuper:
            self.L = H + liouvillian(None, sc_ops)
        else:
            self.L = liouvillian(H, sc_ops)
        if c_ops:
            self.L = self.L + liouvillian(None, c_ops)

        self.c_ops = [spre(op) + spost(op.dag()) for op in sc_ops]
        self.num_collapse = len(self.c_ops)
        self.state_size = self.L.shape[1]
        self._is_set = 0
        self.N_root = int(self.state_size**0.5)
        self.dt = derr_dt

    cpdef Data drift(self, t, Data state):
        return self.L.matmul_data(t, state)

    cpdef list diffusion(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef complex expect
        cdef list out = []
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            vec = c_op.matmul_data(t, state)
            expect = _data.trace_oper_ket(vec)
            out.append(_data.add(vec, state, -expect))
        return out

    cpdef list expect(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef list expect = []
        for i in range(self.num_collapse):
            c_op = self.c_ops[i]
            vec = c_op.matmul_data(t, state)
            expect.append(_data.trace_oper_ket(vec))
        return expect

    cpdef void set_state(self, double t, Data state_raw) except *:
        cdef n, l
        cdef Dense state = _data.to(Dense, state_raw)
        self.t = t
        if not state.fortran:
            state = state.reorder(fortran=1)
        self.state = state
        self._a_set = False
        self._b_set = False
        self._Lb_set = False
        self._L0b_set = False
        self._La_set = False
        self._LLb_set = False
        self._L0a_set = False

        if not self._is_set:
            n = self.num_collapse
            l = self.state_size
            self._is_set = 1
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

    cpdef Data a(self):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._a_set:
            self._compute_a()
        return self._a

    cdef void _compute_a(StochasticOpenSystem self) except *:
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        imul_dense(self._a, 0)
        self.L.matmul_data(self.t, self.state, self._a)
        self._a_set = True

    cpdef Data bi(self, int i):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._b_set:
            self._compute_b()
        return _dense_wrap(self._b[i, :])

    cpdef complex expect_i(self, int i):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._b_set:
            self._compute_b()
        return self.expect_Cv[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_b(self) except *:
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        cdef int i
        cdef QobjEvo c_op
        cdef Dense b_vec, state=self.state
        for i in range(self.num_collapse):
            c_op = <QobjEvo> self.c_ops[i]
            b_vec = <Dense> _dense_wrap(self._b[i, :])
            imul_dense(b_vec, 0)
            c_op.matmul_data(self.t, state, b_vec)
            self.expect_Cv[i] = trace_oper_ket_dense(b_vec)
            iadd_dense(b_vec, state, -self.expect_Cv[i])
        self._b_set = True

    cpdef Data Libj(self, int i, int j):
        if not self._is_set:
            raise RuntimeError(
                "Derrivatives set for ito taylor expansion need "
                "to receive the state with `set_state`."
            )
        if not self._Lb_set:
            self._compute_Lb()
        # We only support commutative diffusion
        if i > j:
            j, i = i, j
        return _dense_wrap(self._Lb[i, j, :])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_Lb(self) except *:
        cdef int i, j
        cdef QobjEvo c_op
        cdef Dense b_vec, Lb_vec, state=self.state
        cdef complex expect
        if not self._b_set:
            self._compute_b()

        for i in range(self.num_collapse):
            c_op = <QobjEvo> self.c_ops[i]
            for j in range(i, self.num_collapse):
                b_vec = <Dense> _dense_wrap(self._b[j, :])
                Lb_vec = <Dense> _dense_wrap(self._Lb[i, j, :])
                imul_dense(Lb_vec, 0)
                c_op.matmul_data(self.t, b_vec, Lb_vec)
                self.expect_Cb[i,j] = trace_oper_ket_dense(Lb_vec)
                iadd_dense(Lb_vec, b_vec, -self.expect_Cv[i])
                iadd_dense(Lb_vec, state, -self.expect_Cb[i,j])
        self._Lb_set = True

    cpdef Data Lia(self, int i):
        if not self._La_set:
            self._compute_La()
        return _dense_wrap(self._La[i, :])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_La(self) except *:
        cdef int i
        cdef QobjEvo c_op
        cdef Dense b_vec, La_vec
        if not self._b_set:
            self._compute_b()

        for i in range(self.num_collapse):
            b_vec = <Dense> _dense_wrap(self._b[i, :])
            La_vec = <Dense> _dense_wrap(self._La[i, :])
            imul_dense(La_vec, 0.)
            self.L.matmul_data(self.t, b_vec, La_vec)
        self._La_set = True

    cpdef Data L0bi(self, int i):
        # L0bi = abi' + dbi/dt + Sum_j bjbjbi"/2
        if not self._L0b_set:
            self._compute_L0b()
        return _dense_wrap(self._L0b[i, :])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_L0b(self) except *:
        cdef int i, j
        cdef QobjEvo c_op
        cdef Dense b_vec, L0b_vec
        if not self._Lb_set:
            self._compute_Lb()
        if not self._a_set:
            self._compute_a()

        for i in range(self.num_collapse):
            c_op = <QobjEvo> self.c_ops[i]
            L0b_vec = <Dense> _dense_wrap(self._L0b[i, :])
            b_vec = <Dense> _dense_wrap(self._b[i, :])
            imul_dense(L0b_vec, 0.)

            # db/dt
            if not c_op.isconstant:
                c_op.matmul_data(self.t + self.dt, self.state, L0b_vec)
                expect = trace_oper_ket_dense(L0b_vec)
                iadd_dense(L0b_vec, self.state, -expect)
                iadd_dense(L0b_vec, b_vec, -1)
                imul_dense(L0b_vec, 1/self.dt)

            # ab'
            imul_dense(self.temp, 0)
            c_op.matmul_data(self.t, self._a, self.temp)
            expect = trace_oper_ket_dense(self.temp)
            iadd_dense(L0b_vec, self.temp, 1)
            iadd_dense(L0b_vec, self._a, -self.expect_Cv[i])
            iadd_dense(L0b_vec, self.state, -expect)

            # bbb" : expect_Cb[i,j] only defined for j>=i
            for j in range(i):
                b_vec = <Dense> _dense_wrap(self._b[j, :])
                iadd_dense(L0b_vec, b_vec, -self.expect_Cb[j,i])
            for j in range(i, self.num_collapse):
                b_vec = <Dense> _dense_wrap(self._b[j, :])
                iadd_dense(L0b_vec, b_vec, -self.expect_Cb[i,j])
        self._L0b_set = True

    cpdef Data LiLjbk(self, int i, int j, int k):
        # LiLjbk = bi(bj'bk'+bjbk"), i<=j<=k
        if not self._LLb_set:
            self._compute_LLb()
        # Only commutative noise supported
        # Definied for i <= j <= k
        # Simple bubble sort to order the terms
        if i>j: i, j = j, i
        if j>k:
          j, k = k, j
          if i>j: i, j = j, i

        return _dense_wrap(self._LLb[i, j, k, :])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_LLb(self) except *:
        # LiLjbk = bi(bj'bk'+bjbk"), i<=j<=k
        # sc_ops must commute (LiLjbk = LjLibk = LkLjbi)
        cdef int i, j, k
        cdef QobjEvo c_op
        cdef Dense bj_vec, bk_vec, LLb_vec, Lb_vec
        if not self._Lb_set:
            self._compute_Lb()

        for i in range(self.num_collapse):
          for j in range(i, self.num_collapse):
            for k in range(j, self.num_collapse):
                c_op = <QobjEvo> self.c_ops[i]
                LLb_vec = <Dense> _dense_wrap(self._LLb[i, j, k, :])
                Lb_vec = <Dense> _dense_wrap(self._Lb[j, k, :])
                bj_vec = <Dense> _dense_wrap(self._b[j, :])
                bk_vec = <Dense> _dense_wrap(self._b[k, :])
                imul_dense(LLb_vec, 0.)

                c_op.matmul_data(self.t, Lb_vec, LLb_vec)
                expect = trace_oper_ket_dense(LLb_vec)

                iadd_dense(LLb_vec, Lb_vec, -self.expect_Cv[i])
                iadd_dense(LLb_vec, self.state, -expect)
                iadd_dense(LLb_vec, bj_vec, -self.expect_Cb[i,k])
                iadd_dense(LLb_vec, bk_vec, -self.expect_Cb[i,j])

        self._LLb_set = True

    cpdef Data L0a(self):
        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        if not self._L0a_set:
            self._compute_L0a()
        return self._L0a

    cdef void _compute_L0a(self) except *:
        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        imul_dense(self._L0a, 0.)
        if not self.L.isconstant:
            self.L.matmul_data(self.t + self.dt, self.state, self._L0a)
            iadd_dense(self._L0a, self._a, -1)
            imul_dense(self._L0a, 1/self.dt)
        self.L.matmul_data(self.t, self._a, self._L0a)
        self._L0a_set = True

    def __reduce__(self):
        return (
            StochasticOpenSystem.restore,
            (self.L, self.c_ops, self.dt)
        )

    @classmethod
    def restore(cls, L, c_ops, derr_dt):
        cdef StochasticOpenSystem out = cls.__new__(cls)
        out.L = L
        out.c_ops = c_ops
        out.num_collapse = len(c_ops)
        out.state_size = out.L.shape[1]
        out._is_set = 0
        out.N_root = int(out.state_size**0.5)
        out.dt = derr_dt
        return out


cdef class SimpleStochasticSystem(_StochasticSystem):
    """
    Simple system that can be solver analytically.
    Used in tests.

        drift = -iH @ vec

        diffusion = c_i @ vec

    """
    cdef double dt

    def __init__(self, H, c_ops):
        self.L = -1j * H
        self.c_ops = c_ops

        self.num_collapse = len(self.c_ops)
        self.dt = 1e-6

    cpdef Data drift(self, t, Data state):
        return self.L.matmul_data(t, state)

    cpdef list diffusion(self, t, Data state):
        cdef int i
        cdef out = []
        for i in range(self.num_collapse):
            out.append(self.c_ops[i].matmul_data(t, state))
        return out

    cpdef list expect(self, t, Data state):
        cdef int i
        cdef list expect = []
        for i in range(self.num_collapse):
            expect.append(0j)
        return expect

    cpdef void set_state(self, double t, Data state) except *:
        self.t = t
        self.state = state

    cpdef Data a(self):
        return self.L.matmul_data(self.t, self.state)

    cpdef Data bi(self, int i):
        return self.c_ops[i].matmul_data(self.t, self.state)

    cpdef complex expect_i(self, int i):
        return 0j

    cpdef Data Libj(self, int i, int j):
        bj = self.c_ops[i].matmul_data(self.t, self.state)
        return self.c_ops[j].matmul_data(self.t, bj)

    cpdef Data Lia(self, int i):
        bi = self.c_ops[i].matmul_data(self.t, self.state)
        return self.L.matmul_data(self.t, bi)

    cpdef Data L0bi(self, int i):
        # L0bi = abi' + dbi/dt + Sum_j bjbjbi"/2
        a = self.L.matmul_data(self.t, self.state)
        abi = self.c_ops[i].matmul_data(self.t, a)
        b = self.c_ops[i].matmul_data(self.t, self.state)
        bdt = self.c_ops[i].matmul_data(self.t + self.dt, self.state)
        return abi + (bdt - b) / self.dt

    cpdef Data LiLjbk(self, int i, int j, int k):
        bk = self.c_ops[k].matmul_data(self.t, self.state)
        Ljbk = self.c_ops[j].matmul_data(self.t, bk)
        return self.c_ops[i].matmul_data(self.t, Ljbk)

    cpdef Data L0a(self):
        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        a = self.L.matmul_data(self.t, self.state)
        aa = self.L.matmul_data(self.t, a)
        adt = self.L.matmul_data(self.t + self.dt, self.state)
        return aa + (adt - a) / self.dt

    def analytic(self, t, W):
        """
        Analytic solution, H and all c_ops must commute.
        Support time dependance of order 2 (a + b*t + c*t**2)
        """
        def _intergal(f, T):
            return (f(0) + 4 * f(T/2) + f(T)) / 6

        out = _intergal(self.L, t) * t
        for i in range(self.num_collapse):
            out += _intergal(self.c_ops[i], t) * W[i]
            out -= 0.5 * _intergal(
                lambda t: self.c_ops[i](t) @ self.c_ops[i](t), t
            ) * t
        return out.expm().data
