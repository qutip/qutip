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
    "StochasticSystem",
    "TaylorStochasticSystem",
    "StochasticOpenSystem",
    "StochasticClosedSystem",
]

@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef Dense _dense_wrap(double complex [::1] x):
    return dense.wrap(&x[0], x.shape[0], 1)


cdef class BaseStochasticSystem:
    """
    RHS for stochastic differential equations.
    """
    def __init__(self):
        pass

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

    cpdef list _shift(self, t, Data state):
        """
        Shift between the noise ``dW`` and measurement.
        Used in custom evolution when the measurement is known and the noise
        is desired.
        """
        raise NotImplementedError


cdef class StochasticSystem(BaseStochasticSystem):
    """
    Right-hand side (RHS) for Stochastic Differential Equations (SDE).

    Encapsulates the deterministic drift term and the stochastic diffusion
    term[s].

    Parameters
    ----------
    drift : callable ``(t: float, state: Data) -> Data``
        Deterministic drift function.

    diffusion : list of callable or callable
        Stochastic diffusion function(s). Either a list of independant
        diffusion functions or a single function returning a list of diffusion
        term(s). The signature should be
        ``(t: float, state: Data) -> Data | list[Data]``.

    num_diffusion : int
        The total number of Wiener processes (diffusion terms).

    _shift : callable ``(t: float, state: Data) -> list[float]``, optional
        A function with signature defining the shift between noise $dW$ and
        measurement. Not used in normal evolution.
    """
    cdef:
      public object drift_func, diffusion_func, _shift_func

    def __init__(self, drift, diffusion, num_diffusion, _shift=None):
        self.drift_func = drift
        self.diffusion_func = diffusion
        self.num_diffusion = num_diffusion
        self._shift_func = _shift

    cpdef Data drift(self, t, Data state):
        return self.drift_func(t, state)

    cpdef list diffusion(self, t, Data state):
        if isinstance(self.diffusion_func, list):
            return [func(t, state) for func in self.diffusion_func]
        return self.diffusion_func(t, state)

    cpdef list _shift(self, t, Data state):
        if self._shift_func is None:
            raise NotImplementedError
        return self._shift_func(t, state)


cdef class TaylorStochasticSystem(BaseStochasticSystem):
    """
    Base class for SDE systems with analytical Ito-Taylor derivatives.

    To implement a specific SDE, subclass this and override the required
    methods. Solvers update the internal state using `set_state`
    before querying individual operators.

    To use this object, create a child class and overwrite the needed method.
    In order of importance ``a`` and ``bi`` are always needed.
    ``Libj`` is required for order 1 method such as Milstein.
    Other derivative are only needed for higher order methods such are
    taylor order 1.5.

    Overwrite ``__init__`` and ``set_state`` as needed.

    Notes
    -----
    Current SDE integration methods assume that the diffusion terms commute.
    ``Libj(a, b) == Libj(b, a)``, for all $i, j$, at every order (``LiLjbk``).
    """
    def __init__(self, int num_diffusion):
        """
        Parameters
        ----------
        num_diffusion : int
          Number of diffusion terms.
        """
        self.num_diffusion = num_diffusion

    cpdef void set_state(self, double t, Data state) except *:
        """
        Update the internal time and state cache.

        This method is guaranteed to be called by the solver before evaluating
        any drift, diffusion, or derivative operators.
        """
        self.t = t
        self.state = state

    cpdef Data a(self):
        """
        Deterministic drift vector $a(t, x)$.
        """
        raise NotImplementedError

    cpdef Data bi(self, int i):
        """
        Diffusion vector $b^i(t, x)$ for the $i$-th Wiener process.
        """
        raise NotImplementedError

    cpdef Data Libj(self, int i, int j):
        """
        First-order diffusion derivative operator acting on a diffusion term.

        $$(L_i b^j)_\mu = \sum_n b^i_n \frac{\partial b^j_\mu}{\partial x_n}$$
        """
        raise NotImplementedError

    cpdef Data Lia(self, int i):
        """
        First-order diffusion derivative operator acting on the drift term.

        $$(L_i a)_\mu = \sum_n b^i_n \frac{\partial a_\mu}{\partial x_n}$$
        """
        raise NotImplementedError

    cpdef Data L0bi(self, int i):
        """
        Drift derivative operator acting on a diffusion term.

        $$(L_0 b^i)_\mu =
            \frac{\partial b^i_\mu}{\partial t}
            + \sum_n a_n \frac{\partial b^i_\mu}{\partial x_n}
            + \frac{1}{2} \sum_{k,n,m} b^k_n b^k_m
            \frac{\partial^2 b^i_\mu}{\partial x_n \partial x_m}
        $$
        """
        raise NotImplementedError

    cpdef Data LiLjbk(self, int i, int j, int k):
        """
        Second-order diffusion derivative operator acting on a diffusion term.

        $$(L_i L_j b^k)_\mu =
            \sum_n b^i_n \frac{\partial}{\partial x_n}
            \left( \sum_m b^j_m \frac{\partial b^k_\mu}{\partial x_m} \right)
        $$
        """
        raise NotImplementedError

    cpdef Data L0a(self):
        """
        Drift derivative operator acting on the drift term.

        $$(L_0 a)_\mu =
            \frac{\partial a_\mu}{\partial t}
            + \sum_n a_n \frac{\partial a_\mu}{\partial x_n}
            + \frac{1}{2} \sum_{k,n,m} b^k_n b^k_m
              \frac{\partial^2 a_\mu}{\partial x_n \partial x_m}
        $$
        """
        raise NotImplementedError

    cpdef complex _shift_i(self, int i):
        """
        Shift between the noise ``dW`` and measurement.
        Used in custom evolution when the measurement is known and the noise
        is desired.
        """
        raise NotImplementedError

    cpdef Data drift(self, t, Data state):
        if self.t != t or self.state is not state:
            self.set_state(t, state)
        return self.a()

    cpdef list diffusion(self, t, Data state):
        if self.t != t or self.state is not state:
            self.set_state(t, state)
        return [self.bi(i) for i in range(self.num_diffusion)]

    cpdef list _shift(self, t, Data state):
        if self.t != t or self.state is not state:
            self.set_state(t, state)
        return [self._shift_i(i) for i in range(self.num_diffusion)]


cdef class StochasticClosedSystem(BaseStochasticSystem):
    """
        RHS for closed quantum stochastic system (ssesolve)

        drift = -1H * psi
              + sum_i (-c_i.dag * c_i / 2 + c_i * e_i / 2 - e_i**2 / 8) * psi

        e_i = <psi| c_i + c_i.dag |psi>

        diffusion = (c_i - e_i / 2) * psi
    """
    cdef readonly list cpcd_ops
    cdef readonly list c_ops
    cdef readonly QobjEvo L

    def __init__(self, H, sc_ops):
        self.L = -1j * H
        self.c_ops = sc_ops
        self.cpcd_ops = [op + op.dag() for op in sc_ops]

        self.num_diffusion = len(self.c_ops)
        for c_op in self.c_ops:
            self.L += -0.5 * c_op.dag() * c_op

    def _register_feedback(self, val):
        self.L._register_feedback({"WienerFeedback": val}, "stochastic solver")
        for op in self.c_ops:
            op._register_feedback({"WienerFeedback": val}, "stochastic solver")
        for op in self.cpcd_ops:
            op._register_feedback({"WienerFeedback": val}, "stochastic solver")

    cpdef Data drift(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef Data temp, out

        out = self.L.matmul_data(t, state)
        for i in range(self.num_diffusion):
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
        for i in range(self.num_diffusion):
            c_op = self.c_ops[i]
            _out = c_op.matmul_data(t, state)
            c_op = self.cpcd_ops[i]
            expect = c_op.expect_data(t, state)
            out.append(_data.add(_out, state, -0.5 * expect))
        return out

    cpdef list _shift(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef list expect = []
        for i in range(self.num_diffusion):
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
        out.num_diffusion = len(c_ops)
        return out


cdef class StochasticOpenSystem(TaylorStochasticSystem):
    """
        RHS for open quantum stochastic system (smesolve)

        drift = liouvillian(H, sc_ops + c_ops)(rho)

        diffusion = c_i @ rho + rho @ c_i/dag - tr(c_i @ rho) rho
    """
    cdef int state_size, N_root
    cdef double dt
    cdef int _is_set
    cdef bint _a_set, _b_set, _Lb_set, _L0b_set, _La_set, _LLb_set, _L0a_set
    cdef readonly list c_ops
    cdef readonly QobjEvo L

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
        self.num_diffusion = len(self.c_ops)
        self.state_size = self.L.shape[1]
        self._is_set = 0
        self.N_root = int(self.state_size**0.5)
        self.dt = derr_dt

    def _register_feedback(self, val):
        self.L._register_feedback({"WienerFeedback": val}, "stochastic solver")
        for op in self.c_ops:
            op._register_feedback({"WienerFeedback": val}, "stochastic solver")

    cpdef Data drift(self, t, Data state):
        return self.L.matmul_data(t, state)

    cpdef list diffusion(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef complex expect
        cdef list out = []
        for i in range(self.num_diffusion):
            c_op = self.c_ops[i]
            vec = c_op.matmul_data(t, state)
            expect = _data.trace_oper_ket(vec)
            out.append(_data.add(vec, state, -expect))
        return out

    cpdef list _shift(self, t, Data state):
        cdef int i
        cdef QobjEvo c_op
        cdef list expect = []
        for i in range(self.num_diffusion):
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
            n = self.num_diffusion
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

    cpdef complex _shift_i(self, int i):
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
        for i in range(self.num_diffusion):
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

        for i in range(self.num_diffusion):
            c_op = <QobjEvo> self.c_ops[i]
            for j in range(i, self.num_diffusion):
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

        for i in range(self.num_diffusion):
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

        for i in range(self.num_diffusion):
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
            for j in range(i, self.num_diffusion):
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

        for i in range(self.num_diffusion):
          for j in range(i, self.num_diffusion):
            for k in range(j, self.num_diffusion):
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
        out.num_diffusion = len(c_ops)
        out.state_size = out.L.shape[1]
        out._is_set = 0
        out.N_root = int(out.state_size**0.5)
        out.dt = derr_dt
        return out
