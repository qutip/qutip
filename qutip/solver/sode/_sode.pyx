from libc.math cimport sqrt
from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data, Dense, imul_dense, iadd_dense
cimport cython
from qutip.solver.sode.ssystem cimport BaseStochasticSystem, TaylorStochasticSystem
import numpy as np


cdef double INV_SQRT3 = 1/sqrt(3.)


cdef inline void _assign(Dense dst, Dense src):
    """dst[:] = src, in place."""
    imul_dense(dst, 0.)
    iadd_dense(dst, src, 1.)

cdef class Euler:
    cdef BaseStochasticSystem system
    cdef bint measurement_noise

    def __init__(self, BaseStochasticSystem system, measurement_noise=False):
        self.system = system
        self.measurement_noise = measurement_noise

    @cython.wraparound(False)
    def run(
        self, double t, Data state, double dt,
        double[:, :, ::1] dW, int num_step
    ):
        cdef int i
        cdef Data new_state
        if type(state) is not Dense:
            state = _data.to(Dense, state)

        # Scratch buffer handed to every step.
        # A step may accumulate into it and return it, the previous state then becomes the next scratch.
        cdef Dense out = _data.zeros_like(state)
        state = state.copy()
        self._allocate(state)

        for i in range(num_step):
            new_state = self.step(t + i * dt, state, dt, dW[i, :, :], out)
            if new_state is out:
                out = state
            state = new_state
        return state

    cdef void _allocate(self, Dense state):
        """Allocate the scratch states a step needs, once per run."""
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense out):
        """
        Integration scheme:
        Basic Euler order 0.5
        dV = d1 dt + d2_i dW_i
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef int i
        cdef BaseStochasticSystem system = self.system
        cdef list expect

        cdef Dense a = system.drift(t, state)
        cdef list b = system.diffusion(t, state)

        if self.measurement_noise:
            expect = system._shift(t, state)
            for i in range(system.num_diffusion):
                dW[0, i] -= expect[i].real * dt

        imul_dense(out, 0.)
        iadd_dense(out, state, 1)
        iadd_dense(out, a, dt)
        for i in range(system.num_diffusion):
            iadd_dense(out, b[i], dW[0, i])
        return out


cdef class Platen(Euler):
    cdef Dense _d1, _Vt
    cdef list _Vp, _Vm

    cdef void _allocate(self, Dense state):
        cdef int n = self.system.num_diffusion
        self._d1 = _data.zeros_like(state)
        self._Vt = _data.zeros_like(state)
        self._Vp = [_data.zeros_like(state) for _ in range(n)]
        self._Vm = [_data.zeros_like(state) for _ in range(n)]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense out):
        """
        Platen rhs function for both master eq and schrodinger eq.
        dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))/2 * dt
             + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
             + (d2_i(V+)-d2_i(V-))/4 * (dW_i**2 -dt) * dt**(-.5)

        Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
        V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5
        The Theory of Open Quantum Systems
        Chapter 7 Eq. (7.47), H.-P Breuer, F. Petruccione
        """
        cdef BaseStochasticSystem system = self.system
        cdef int i, j, num_ops = system.num_diffusion
        cdef double sqrt_dt = sqrt(dt)
        cdef double sqrt_dt_inv = 0.25 / sqrt_dt
        cdef double dw, dw2, dw2p, dw2m
        cdef Dense d1 = self._d1, Vt = self._Vt, Vp, Vm
        cdef list d2, d2p, d2m, expect

        # d1 = state + a(state) dt
        _assign(d1, state)
        iadd_dense(d1, system.drift(t, state), dt)
        d2 = system.diffusion(t, state)

        if self.measurement_noise:
            expect = system._shift(t, state)
            for i in range(num_ops):
                dW[0, i] -= expect[i].real * dt

        # Vt = d1 + sum_i b_i dW_i ;  Vp_i, Vm_i = d1 +/- b_i sqrt(dt)
        imul_dense(out, 0.)
        iadd_dense(out, d1, 0.5)
        _assign(Vt, d1)
        for i in range(num_ops):
            Vp = self._Vp[i]
            Vm = self._Vm[i]
            _assign(Vp, d1)
            iadd_dense(Vp, d2[i], sqrt_dt)
            _assign(Vm, d1)
            iadd_dense(Vm, d2[i], -sqrt_dt)
            iadd_dense(Vt, d2[i], dW[0, i])

        iadd_dense(out, system.drift(t, Vt), 0.5 * dt)
        iadd_dense(out, state, 0.5)
        for i in range(num_ops):
            d2p = system.diffusion(t, self._Vp[i])
            d2m = system.diffusion(t, self._Vm[i])
            dw = dW[0, i] * 0.25
            iadd_dense(out, d2[i], 2 * dw)

            for j in range(num_ops):
                if i == j:
                    dw2 = sqrt_dt_inv * (dW[0, i] * dW[0, j] - dt)
                    dw2p = dw2 + dw
                    dw2m = -dw2 + dw
                else:
                    dw2p = sqrt_dt_inv * dW[0, i] * dW[0, j]
                    dw2m = -dw2p
                iadd_dense(out, d2p[j], dw2p)
                iadd_dense(out, d2m[j], dw2m)

        return out


cdef class Explicit15(Euler):
    cdef Dense _V
    cdef list _v2p, _v2m, _p2p, _p2m
    cdef double[::1] _dw, _dz, _dwp, _dwm

    def __init__(self, BaseStochasticSystem system):
        self.system = system

    cdef void _allocate(self, Dense state):
        cdef int n = self.system.num_diffusion
        self._V = _data.zeros_like(state)
        self._v2p = [_data.zeros_like(state) for _ in range(n)]
        self._v2m = [_data.zeros_like(state) for _ in range(n)]
        self._p2p = [[_data.zeros_like(state) for _ in range(n)] for _ in range(n)]
        self._p2m = [[_data.zeros_like(state) for _ in range(n)] for _ in range(n)]
        self._dw = np.empty(n)
        self._dz = np.empty(n)
        self._dwp = np.empty(n)
        self._dwm = np.empty(n)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense out):
        """
        Chapter 11.2 Eq. (2.13)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef BaseStochasticSystem system = self.system
        cdef int i, j, k, num_ops = system.num_diffusion
        cdef double sqrt_dt = sqrt(dt)
        cdef double sqrt_dt_inv = 1./sqrt_dt
        cdef double ddz, ddw, ddd
        cdef double[::1] dw = self._dw, dz = self._dz, dwp = self._dwp, dwm = self._dwm
        cdef Dense V = self._V, v2p, v2m, p2p, p2m, d1, d1p, d1m
        cdef list d2, dd2, d2p, d2m, d2pp, d2mm, d2p_all, d2m_all
        cdef object t_obj = t, t_dt = t + dt, t_n = t + dt / num_ops

        for i in range(num_ops):
            dw[i] = dW[0, i]
            dz[i] = 0.5 *(dW[0, i] + INV_SQRT3 * dW[1, i])

        d1 = system.drift(t_obj, state)
        d2 = system.diffusion(t_obj, state)
        dd2 = system.diffusion(t_dt, state)
        # Euler part
        _assign(out, state)
        iadd_dense(out, d1, dt)
        for i in range(num_ops):
            iadd_dense(out, d2[i], dw[i])

        _assign(V, state)
        iadd_dense(V, d1, dt/num_ops)
        for i in range(num_ops):
            v2p = self._v2p[i]
            v2m = self._v2m[i]
            _assign(v2p, V)
            iadd_dense(v2p, d2[i], sqrt_dt)
            _assign(v2m, V)
            iadd_dense(v2m, d2[i], -sqrt_dt)

        d2p_all = []
        d2m_all = []
        for i in range(num_ops):
            v2p = self._v2p[i]
            d2p = system.diffusion(t_obj, v2p)
            d2m = system.diffusion(t_obj, self._v2m[i])
            d2p_all.append(d2p)
            d2m_all.append(d2m)
            ddw = (dw[i] * dw[i] - dt) * 0.25 * sqrt_dt_inv  # 1.0
            iadd_dense(out, d2p[i], ddw)
            iadd_dense(out, d2m[i], -ddw)
            for j in range(num_ops):
                p2p = self._p2p[i][j]
                p2m = self._p2m[i][j]
                _assign(p2p, v2p)
                iadd_dense(p2p, d2p[j], sqrt_dt)
                _assign(p2m, v2p)
                iadd_dense(p2m, d2p[j], -sqrt_dt)

        iadd_dense(out, d1, -0.5*(num_ops) * dt)
        for i in range(num_ops):
            ddz = dz[i] * 0.5 / sqrt_dt # 1.5
            ddd = 0.25 * (dw[i] * dw[i] / 3 - dt) * dw[i] / dt # 1.5
            for j in range(num_ops):
                dwp[j] = 0
                dwm[j] = 0

            d1p = system.drift(t_n, self._v2p[i])
            d1m = system.drift(t_n, self._v2m[i])
            d2p = d2p_all[i]
            d2m = d2m_all[i]
            d2pp = system.diffusion(t_obj, self._p2p[i][i])
            d2mm = system.diffusion(t_obj, self._p2m[i][i])

            iadd_dense(out, d1p, (0.25 + ddz) * dt)
            iadd_dense(out, d1m, (0.25 - ddz) * dt)
            iadd_dense(out, dd2[i], dw[i] - dz[i])
            iadd_dense(out, d2[i], dz[i] - dw[i])
            iadd_dense(out, d2pp[i], ddd)
            iadd_dense(out, d2mm[i], -ddd)
            dwp[i] += -ddd
            dwm[i] += ddd

            for j in range(num_ops):
                ddw = 0.5 * (dw[j] - dz[j])  # O(1.5)
                dwp[j] += ddw
                dwm[j] += ddw
                iadd_dense(out, d2[j], -2*ddw)

                if j > i:
                    ddw = 0.5 * (dw[i] * dw[j]) / sqrt_dt  # O(1.0)
                    dwp[j] += ddw
                    dwm[j] += -ddw

                    ddw = 0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt  # O(1.5)
                    d2pp = system.diffusion(t_obj, self._p2p[j][i])
                    d2mm = system.diffusion(t_obj, self._p2m[j][i])
                    iadd_dense(out, d2pp[j], ddw)
                    iadd_dense(out, d2mm[j], -ddw)
                    dwp[j] += -ddw
                    dwm[j] += ddw

                    for k in range(j+1, num_ops):
                        ddw = 0.5 * dw[i] * dw[j] * dw[k] / dt  # O(1.5)
                        iadd_dense(out, d2pp[k], ddw)
                        iadd_dense(out, d2mm[k], -ddw)
                        dwp[k] += -ddw
                        dwm[k] += ddw

                if j < i:
                    ddw = 0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt  # O(1.5)
                    d2pp = system.diffusion(t_obj, self._p2p[j][i])
                    d2mm = system.diffusion(t_obj, self._p2m[j][i])
                    iadd_dense(out, d2pp[j], ddw)
                    iadd_dense(out, d2mm[j], -ddw)
                    dwp[j] += -ddw
                    dwm[j] += ddw

            for j in range(num_ops):
                iadd_dense(out, d2p[j], dwp[j])
                iadd_dense(out, d2m[j], dwm[j])

        return out


cdef class Milstein:
    cdef TaylorStochasticSystem system
    cdef bint measurement_noise
    cdef double[::1] _dz

    def __init__(self, TaylorStochasticSystem system, measurement_noise=False):
        self.system = system
        self.measurement_noise = measurement_noise

    @cython.wraparound(False)
    def run(self, double t, Data state, double dt, double[:, :, ::1] dW, int ntraj):
        cdef int i
        if type(state) != _data.Dense:
            state = _data.to(_data.Dense, state)
        cdef Dense out = _data.zeros_like(state)
        state = state.copy()

        for i in range(ntraj):
            self.step(t + i * dt, state, dt, dW[i, :, :], out)
            state, out = out, state
        return state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense out):
        """
        Chapter 10.3 Eq. (3.12)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen

        dV = -iH*V*dt + d1*dt + d2_i*dW_i
        + 0.5*d2_i' d2_j*(dW_i*dw_j -dt*delta_ij)
        """
        cdef BaseStochasticSystem system = self.system
        cdef int i, j, num_ops = system.num_diffusion
        cdef double dw

        system.set_state(t, state)

        imul_dense(out, 0.)
        iadd_dense(out, state, 1)
        iadd_dense(out, system.a(), dt)

        if self.measurement_noise:
            for i in range(system.num_diffusion):
                dW[0, i] -= system._shift_i(i).real * dt

        for i in range(num_ops):
            iadd_dense(out, system.bi(i), dW[0, i])

        for i in range(num_ops):
            for j in range(i, num_ops):
                if i == j:
                    dw = (dW[0, i] * dW[0, j] - dt) * 0.5
                else:
                    dw = dW[0, i] * dW[0, j]
                iadd_dense(out, system.Libj(i, j), dw)


cdef class PredCorr:
    cdef Dense euler
    cdef double alpha, eta
    cdef TaylorStochasticSystem system
    cdef bint measurement_noise

    def __init__(
        self, TaylorStochasticSystem system,
        double alpha=0., double eta=0.5,
        measurement_noise=False
    ):
        self.system = system
        self.alpha = alpha
        self.eta = eta
        self.measurement_noise = measurement_noise

    @cython.wraparound(False)
    def run(self, double t, Data state, double dt, double[:, :, ::1] dW, int ntraj):
        cdef int i
        if type(state) != _data.Dense:
            state = _data.to(_data.Dense, state)
        cdef Dense out = _data.zeros_like(state)
        self.euler = _data.zeros_like(state)
        state = state.copy()

        for i in range(ntraj):
            self.step(t + i * dt, state, dt, dW[i, :, :], out)
            state, out = out, state
        return state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense out):
        """
        Chapter 15.5 Eq. (5.4)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef TaylorStochasticSystem system = self.system
        cdef int i, j, k, num_ops = system.num_diffusion
        cdef double eta=self.eta, alpha=self.alpha
        cdef Dense euler = self.euler

        system.set_state(t, state)

        if self.measurement_noise:
            for i in range(system.num_diffusion):
                dW[0, i] -= system._shift_i(i).real * dt

        imul_dense(out, 0.)
        iadd_dense(out, state, 1)
        iadd_dense(out, system.a(), dt * (1-alpha))

        imul_dense(euler, 0.)
        iadd_dense(euler, state, 1)
        iadd_dense(euler, system.a(), dt)

        for i in range(num_ops):
            iadd_dense(euler, system.bi(i), dW[0, i])
            iadd_dense(out, system.bi(i), dW[0, i] * eta)
            iadd_dense(out, system.Libj(i, i), dt * (alpha-1) * 0.5)

        system.set_state(t+dt, euler)
        for i in range(num_ops):
            iadd_dense(out, system.bi(i), dW[0, i] * (1-eta))

        if alpha:
            iadd_dense(out, system.a(), dt*alpha)
            for i in range(num_ops):
                iadd_dense(out, system.Libj(i, i), -dt * alpha * 0.5)

        return out


cdef class Taylor15(Milstein):
    def __init__(self, TaylorStochasticSystem system):
        self.system = system
        self.measurement_noise = False
        self._dz = np.empty(max(system.num_diffusion, 1))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense out):
        """
        Chapter 10.4 Eq. (4.6),
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef TaylorStochasticSystem system = self.system
        system.set_state(t, state)
        cdef int i, j, k, num_ops = system.num_diffusion
        cdef double[:] dz, dw

        num_ops = system.num_diffusion
        dw = dW[0, :]
        dz = self._dz
        for i in range(num_ops):
            dz[i] = 0.5 * (dW[0, i] + dW[1, i] * INV_SQRT3) * dt

        imul_dense(out, 0.)
        iadd_dense(out, state, 1)
        iadd_dense(out, system.a(), dt)
        iadd_dense(out, system.L0a(), 0.5 * dt * dt)

        for i in range(num_ops):
            iadd_dense(out, system.bi(i), dw[i])
            iadd_dense(out, system.Libj(i, i), 0.5 * (dw[i] * dw[i] - dt))
            iadd_dense(out, system.Lia(i), dz[i])
            iadd_dense(out, system.L0bi(i), dw[i] * dt - dz[i])
            iadd_dense(out, system.LiLjbk(i, i, i),
                             0.5 * ((1/3.) * dw[i] * dw[i] - dt) * dw[i])

            for j in range(i+1, num_ops):
                iadd_dense(out, system.Libj(i, j), dw[i] * dw[j])
                iadd_dense(out, system.LiLjbk(i, j, j), 0.5 * (dw[j] * dw[j] -dt) * dw[i])
                iadd_dense(out, system.LiLjbk(i, i, j), 0.5 * (dw[i] * dw[i] -dt) * dw[j])
                for k in range(j+1, num_ops):
                    iadd_dense(out, system.LiLjbk(i, j, k), dw[i]*dw[j]*dw[k])

        return out


cdef class Milstein_imp:
    cdef TaylorStochasticSystem system
    cdef bint use_inv
    cdef double[::1] _dz
    cdef QobjEvo implicit
    cdef Data inv
    cdef double prev_dt
    cdef dict imp_opt

    def __init__(self, TaylorStochasticSystem system, solve_method=None, solve_options={}):
        self.system = system
        self.prev_dt = 0
        self._dz = np.empty(max(system.num_diffusion, 1))
        if solve_method == "inv":
            if not self.system.L.isconstant:
                raise TypeError("The 'inv' integration method requires that the system Hamiltonian or Liouvillian be constant.")
            self.use_inv = True
            self.imp_opt = {}
        else:
            self.use_inv = False
            self.imp_opt = {"method": solve_method, "options": solve_options}


    @cython.wraparound(False)
    def run(self, double t, Data state, double dt, double[:, :, ::1] dW, int ntraj):
        cdef int i
        if type(state) != _data.Dense:
            state = _data.to(_data.Dense, state)
        cdef Dense tmp = _data.zeros_like(state)

        if dt != self.prev_dt:
            self.implicit = 1 - self.system.L * (dt / 2)
            if self.use_inv:
                self.inv = _data.inv(self.implicit._call(0))

        for i in range(ntraj):
            state = self.step(t + i * dt, state, dt, dW[i, :, :], tmp)
        return state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense target):
        """
        Chapter 12.2 Eq. (2.11)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef TaylorStochasticSystem system = self.system
        cdef int i, j, num_ops = system.num_diffusion
        cdef double dw

        system.set_state(t, state)

        imul_dense(target, 0.)
        iadd_dense(target, state, 1)
        iadd_dense(target, system.a(), dt * 0.5)

        for i in range(num_ops):
            iadd_dense(target, system.bi(i), dW[0, i])

        for i in range(num_ops):
            for j in range(i, num_ops):
                if i == j:
                    dw = (dW[0, i] * dW[0, j] - dt) * 0.5
                else:
                    dw = dW[0, i] * dW[0, j]
                iadd_dense(target, system.Libj(i, j), dw)

        if self.use_inv:
            out = _data.matmul(self.inv, target, dtype=Dense)
        else:
            out = _data.solve(self.implicit._call(t+dt), target, **self.imp_opt, dtype=Dense)

        return out


cdef class Taylor15_imp(Milstein_imp):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense target):
        """
        Chapter 12.2 Eq. (2.18),
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef TaylorStochasticSystem system = self.system
        system.set_state(t, state)
        cdef int i, j, k, num_ops = system.num_diffusion
        cdef double[:] dz, dw

        num_ops = system.num_diffusion
        dw = dW[0, :]
        dz = self._dz
        for i in range(num_ops):
            dz[i] = 0.5 * (dW[0, i] + dW[1, i] * INV_SQRT3) * dt

        imul_dense(target, 0.)
        iadd_dense(target, state, 1)
        iadd_dense(target, system.a(), dt * 0.5)

        for i in range(num_ops):
            iadd_dense(target, system.bi(i), dw[i])
            iadd_dense(target, system.Libj(i, i), 0.5 * (dw[i] * dw[i] - dt))
            iadd_dense(target, system.Lia(i), dz[i] - dw[i] * dt * 0.5)
            iadd_dense(target, system.L0bi(i), dw[i] * dt - dz[i])
            iadd_dense(target, system.LiLjbk(i, i, i),
                             0.5 * ((1/3.) * dw[i] * dw[i] - dt) * dw[i])

            for j in range(i+1, num_ops):
                iadd_dense(target, system.Libj(i, j), dw[i] * dw[j])
                iadd_dense(target, system.LiLjbk(i, j, j), 0.5 * (dw[j] * dw[j] -dt) * dw[i])
                iadd_dense(target, system.LiLjbk(i, i, j), 0.5 * (dw[i] * dw[i] -dt) * dw[j])
                for k in range(j+1, num_ops):
                    iadd_dense(target, system.LiLjbk(i, j, k), dw[i]*dw[j]*dw[k])

        if self.use_inv:
            out = _data.matmul(self.inv, target, dtype=Dense)
        else:
            out = _data.solve(self.implicit._call(t+dt), target, **self.imp_opt, dtype=Dense)

        return out
