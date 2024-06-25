#cython: language_level=3

from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data, Dense, imul_dense, iadd_dense
from collections import defaultdict
cimport cython
from qutip.solver.sode.ssystem cimport _StochasticSystem
import numpy as np


cdef class Euler:
    cdef _StochasticSystem system
    cdef bint measurement_noise

    def __init__(self, _StochasticSystem system, measurement_noise=False):
        self.system = system
        self.measurement_noise = measurement_noise

    @cython.wraparound(False)
    def run(
        self, double t, Data state, double dt,
        double[:, :, ::1] dW, int num_step
    ):
        cdef int i
        for i in range(num_step):
            state = self.step(t + i * dt, state, dt, dW[i, :, :])
        return state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Data state, double dt, double[:, :] dW):
        """
        Integration scheme:
        Basic Euler order 0.5
        dV = d1 dt + d2_i dW_i
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef int i
        cdef _StochasticSystem system = self.system
        cdef list expect

        cdef Data a = system.drift(t, state)
        b = system.diffusion(t, state)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= expect[i].real * dt

        cdef Data new_state = _data.add(state, a, dt)
        for i in range(system.num_collapse):
            new_state = _data.add(new_state, b[i], dW[0, i])
        return new_state


cdef class Platen(Euler):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Data step(self, double t, Data state, double dt, double[:, :] dW):
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
        cdef _StochasticSystem system = self.system
        cdef int i, j, num_ops = system.num_collapse
        cdef double sqrt_dt = np.sqrt(dt)
        cdef double sqrt_dt_inv = 0.25 / sqrt_dt
        cdef double dw, dw2, dw2p, dw2m

        cdef Data d1 = _data.add(state, system.drift(t, state), dt)
        cdef list d2 = system.diffusion(t, state)
        cdef Data Vt, out
        cdef list Vp, Vm
        cdef list expect

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= expect[i].real * dt

        out = _data.mul(d1, 0.5)
        Vt = d1.copy()
        Vp = []
        Vm = []
        for i in range(num_ops):
            Vp.append(_data.add(d1, d2[i], sqrt_dt))
            Vm.append(_data.add(d1, d2[i], -sqrt_dt))
            Vt = _data.add(Vt, d2[i], dW[0, i])

        d1 = system.drift(t, Vt)
        out = _data.add(out, d1, 0.5 * dt)
        out = _data.add(out, state, 0.5)
        for i in range(num_ops):
            d2p = system.diffusion(t, Vp[i])
            d2m = system.diffusion(t, Vm[i])
            dw = dW[0, i] * 0.25
            out = _data.add(out, d2[i], 2 * dw)

            for j in range(num_ops):
                if i == j:
                    dw2 = sqrt_dt_inv * (dW[0, i] * dW[0, j] - dt)
                    dw2p = dw2 + dw
                    dw2m = -dw2 + dw
                else:
                    dw2p = sqrt_dt_inv * dW[0, i] * dW[0, j]
                    dw2m = -dw2p
                out = _data.add(out, d2p[j], dw2p)
                out = _data.add(out, d2m[j], dw2m)

        return out


cdef class Explicit15(Euler):
    def __init__(self, _StochasticSystem system):
        self.system = system

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Data step(self, double t, Data state, double dt, double[:, :] dW):
        """
        Chapter 11.2 Eq. (2.13)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef _StochasticSystem system = self.system
        cdef int i, j, k, num_ops = system.num_collapse
        cdef double sqrt_dt = np.sqrt(dt)
        cdef double sqrt_dt_inv = 1./sqrt_dt
        cdef double ddz, ddw, ddd
        cdef double[::1] dz, dw, dwp, dwm

        dw = np.empty(num_ops)
        dz = np.empty(num_ops)
        dwp = np.zeros(num_ops)
        dwm = np.zeros(num_ops)
        for i in range(num_ops):
            dw[i] = dW[0, i]
            dz[i] = 0.5 *(dW[0, i] + 1./np.sqrt(3) * dW[1, i])

        d1 = system.drift(t, state)
        d2 = system.diffusion(t, state)
        dd2 = system.diffusion(t + dt, state)
        # Euler part
        out = _data.add(state, d1, dt)
        for i in range(num_ops):
            out = _data.add(out, d2[i], dw[i])

        V = _data.add(state, d1, dt/num_ops)

        v2p = []
        v2m = []
        for i in range(num_ops):
            v2p.append(_data.add(V, d2[i], sqrt_dt))
            v2m.append(_data.add(V, d2[i], -sqrt_dt))

        p2p = []
        p2m = []
        for i in range(num_ops):
            d2p = system.diffusion(t, v2p[i])
            d2m = system.diffusion(t, v2m[i])
            ddw = (dw[i] * dw[i] - dt) * 0.25 * sqrt_dt_inv  # 1.0
            out = _data.add(out, d2p[i], ddw)
            out = _data.add(out, d2m[i], -ddw)
            temp_p2p = []
            temp_p2m = []
            for j in range(num_ops):
                temp_p2p.append(_data.add(v2p[i], d2p[j], sqrt_dt))
                temp_p2m.append(_data.add(v2p[i], d2p[j], -sqrt_dt))
            p2p.append(temp_p2p)
            p2m.append(temp_p2m)

        out = _data.add(out, d1, -0.5*(num_ops) * dt)

        for i in range(num_ops):
            ddz = dz[i] * 0.5 / sqrt_dt # 1.5
            ddd = 0.25 * (dw[i] * dw[i] / 3 - dt) * dw[i] / dt # 1.5
            for j in range(num_ops):
                dwp[j] = 0
                dwm[j] = 0

            d1p = system.drift(t + dt/num_ops, v2p[i])
            d1m = system.drift(t + dt/num_ops, v2m[i])

            d2p = system.diffusion(t, v2p[i])
            d2m = system.diffusion(t, v2m[i])
            d2pp = system.diffusion(t, p2p[i][i])
            d2mm = system.diffusion(t, p2m[i][i])

            out = _data.add(out, d1p, (0.25 + ddz) * dt)
            out = _data.add(out, d1m, (0.25 - ddz) * dt)

            out = _data.add(out, dd2[i], dw[i] - dz[i])
            out = _data.add(out, d2[i], dz[i] - dw[i])

            out = _data.add(out, d2pp[i], ddd)
            out = _data.add(out, d2mm[i], -ddd)
            dwp[i] += -ddd
            dwm[i] += ddd

            for j in range(num_ops):
                ddw = 0.5 * (dw[j] - dz[j])  # O(1.5)
                dwp[j] += ddw
                dwm[j] += ddw
                out = _data.add(out, d2[j], -2*ddw)

                if j > i:
                    ddw = 0.5 * (dw[i] * dw[j]) / sqrt_dt  # O(1.0)
                    dwp[j] += ddw
                    dwm[j] += -ddw

                    ddw = 0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt  # O(1.5)
                    d2pp = system.diffusion(t, p2p[j][i])
                    d2mm = system.diffusion(t, p2m[j][i])
                    out = _data.add(out, d2pp[j], ddw)
                    out = _data.add(out, d2mm[j], -ddw)
                    dwp[j] += -ddw
                    dwm[j] += ddw

                    for k in range(j+1, num_ops):
                        ddw = 0.5 * dw[i] * dw[j] * dw[k] / dt  # O(1.5)
                        out = _data.add(out, d2pp[k], ddw)
                        out = _data.add(out, d2mm[k], -ddw)
                        dwp[k] += -ddw
                        dwm[k] += ddw

                if j < i:
                    ddw = 0.25 * (dw[j] * dw[j] - dt) * dw[i] / dt  # O(1.5)
                    d2pp = system.diffusion(t, p2p[j][i])
                    d2mm = system.diffusion(t, p2m[j][i])

                    out = _data.add(out, d2pp[j], ddw)
                    out = _data.add(out, d2mm[j], -ddw)
                    dwp[j] += -ddw
                    dwm[j] += ddw

            for j in range(num_ops):
                out = _data.add(out, d2p[j], dwp[j])
                out = _data.add(out, d2m[j], dwm[j])

        return out


cdef class Milstein:
    cdef _StochasticSystem system
    cdef bint measurement_noise

    def __init__(self, _StochasticSystem system, measurement_noise=False):
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
        cdef _StochasticSystem system = self.system
        cdef int i, j, num_ops = system.num_collapse
        cdef double dw

        system.set_state(t, state)

        imul_dense(out, 0.)
        iadd_dense(out, state, 1)
        iadd_dense(out, system.a(), dt)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= system.expect_i(i).real * dt

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
    cdef _StochasticSystem system
    cdef bint measurement_noise

    def __init__(
        self, _StochasticSystem system,
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
        cdef _StochasticSystem system = self.system
        cdef int i, j, k, num_ops = system.num_collapse
        cdef double eta=self.eta, alpha=self.alpha
        cdef Dense euler = self.euler

        system.set_state(t, state)

        if self.measurement_noise:
            expect = system.expect(t, state)
            for i in range(system.num_collapse):
                dW[0, i] -= system.expect_i(i).real * dt

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
    def __init__(self, _StochasticSystem system):
        self.system = system
        self.measurement_noise = False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Data step(self, double t, Dense state, double dt, double[:, :] dW, Dense out):
        """
        Chapter 10.4 Eq. (4.6),
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef _StochasticSystem system = self.system
        system.set_state(t, state)
        cdef int i, j, k, num_ops = system.num_collapse
        cdef double[:] dz, dw

        num_ops = system.num_collapse
        dw = dW[0, :]
        dz = 0.5 * (dW[0, :] + dW[1, :] / np.sqrt(3)) * dt

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
    cdef _StochasticSystem system
    cdef bint use_inv
    cdef QobjEvo implicit
    cdef Data inv
    cdef double prev_dt
    cdef dict imp_opt

    def __init__(self, _StochasticSystem system, solve_method=None, solve_options={}):
        self.system = system
        self.prev_dt = 0
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
        cdef _StochasticSystem system = self.system
        cdef int i, j, num_ops = system.num_collapse
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
            out = _data.matmul(self.inv, target)
        else:
            out = _data.solve(self.implicit._call(t+dt), target, **self.imp_opt)

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
        cdef _StochasticSystem system = self.system
        system.set_state(t, state)
        cdef int i, j, k, num_ops = system.num_collapse
        cdef double[:] dz, dw

        num_ops = system.num_collapse
        dw = dW[0, :]
        dz = 0.5 * (dW[0, :] + dW[1, :] / np.sqrt(3)) * dt

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
            out = _data.matmul(self.inv, target)
        else:
            out = _data.solve(self.implicit._call(t+dt), target, **self.imp_opt)

        return out
