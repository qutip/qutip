#!python
#cython: language_level=3
## cython: profile=True
## cython: linetrace=True
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sp
from scipy.linalg.cython_blas cimport dznrm2 as raw_dznrm2
from qutip.qobj import Qobj
from qutip.cy.cqobjevo cimport CQobjEvo
from qutip.cy.spmatfuncs cimport cy_expect_psi
# from qutip.cy.dopri5 import ode_td_dopri
#from qutip.cy.complex_math cimport conj
include "complex_math.pxi"

cdef int ONE = 1

cdef double dznrm2(complex[::1] psi):
    cdef int l = psi.shape[0]
    return raw_dznrm2(&l, <complex*>&psi[0], &ONE)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[complex, ndim=1] normalize(np.ndarray[complex, ndim=1] psi):
    cdef int i, l = psi.shape[0]
    cdef double norm = dznrm2(psi)
    cdef np.ndarray[ndim=1, dtype=complex] out = np.empty(l, dtype=complex)
    for i in range(l):
        out[i] = psi[i] / norm
    return out

cdef class CyMcOde:
    cdef:
        int steady_state, store_states, col_args
        int norm_steps, l_vec, num_ops
        double norm_t_tol, norm_tol
        list collapses
        list collapses_args
        list c_ops
        list n_ops
        complex[:,::1] states_out
        complex[:,::1] ss_out
        double[::1] n_dp

    def __init__(self, ss, opt):
        self.c_ops = ss.td_c_ops
        self.n_ops = ss.td_n_ops
        self.norm_steps = opt.norm_steps
        self.norm_t_tol = opt.norm_t_tol
        self.norm_tol = opt.norm_tol
        self.steady_state = opt.steady_state_average
        self.store_states = opt.store_states or opt.average_states
        self.collapse = []
        self.l_vec = self.c_ops[0].cte.shape[0]
        self.num_ops = len(ss.td_n_ops)
        self.n_dp = np.zeros(self.num_ops)

        if ss.col_args:
            self.col_args = 1
            self.collapses_args = ss.args[ss.col_args]
            if ss.type == "QobjEvo":
                ss.H_td.coeff_get.get_args()[ss.col_args] = self.collapses_args
                for c in ss.td_c_ops:
                    c.coeff_get.get_args()[ss.col_args] = self.collapses_args
                for c in ss.td_n_ops:
                    c.coeff_get.get_args()[ss.col_args] = self.collapses_args
        else:
            self.col_args = 0

        if self.steady_state:
            self.ss_out = np.zeros((self.l_vec, self.l_vec), dtype=complex)
        else:
            self.ss_out = np.zeros((0, 0), dtype=complex)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void sumsteadystate(self, complex[::1] state):
        cdef int ii, jj, l_vec
        l_vec = state.shape[0]
        for ii in range(l_vec):
          for jj in range(l_vec):
            self.ss_out[ii,jj] += state[ii]*conj(state[jj])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cy_mc_run_ode(self, ODE, tlist_, e_call, prng):
        cdef np.ndarray[double, ndim=1] rand_vals
        cdef np.ndarray[double, ndim=1] tlist = np.array(tlist_)
        cdef np.ndarray[complex, ndim=1] y_prev
        cdef np.ndarray[complex, ndim=1] out_psi = ODE._y
        cdef int num_times = tlist.shape[0]
        cdef int ii, which, k
        cdef double norm2_prev, norm2_psi
        cdef double t_prev

        self.sumsteadystate(out_psi)

        if self.store_states:
            self.states_out = np.zeros((num_times, self.l_vec), dtype=complex)
            for ii in range(self.l_vec):
                self.states_out[0, ii] = out_psi[ii]
        else:
            self.states_out = np.zeros((1, self.l_vec), dtype=complex)

        e_call.step(0, out_psi)
        rand_vals = prng.rand(2)

        # RUN ODE UNTIL EACH TIME IN TLIST
        norm2_prev = dznrm2(ODE._y) ** 2
        for k in range(1, num_times):
            # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
            t_prev = ODE.t
            y_prev = ODE.y
            while t_prev < tlist[k]:
                # integrate up to tlist[k], one step at a time.
                ODE.integrate(tlist[k], step=1)
                if not ODE.successful():
                    print(ODE.t, t_prev, tlist[k])
                    print(ODE._integrator.call_args)
                    raise Exception("ZVODE failed!")
                norm2_psi = dznrm2(ODE._y) ** 2
                if norm2_psi <= rand_vals[0]:
                    # collapse has occured:
                    self._find_collapse(ODE, norm2_psi, t_prev, y_prev,
                                        norm2_prev, rand_vals[0])
                    t_prev = ODE.t
                    y_prev = ODE.y

                    which = self._which_collapse(t_prev, y_prev, rand_vals[1])
                    y_prev = self._collapse(t_prev, which, y_prev)
                    ODE.set_initial_value(y_prev, t_prev)

                    self.collapses.append((t_prev, which))
                    if self.col_args:
                        self.collapses_args.append((t_prev, which))
                    rand_vals = prng.rand(2)
                    norm2_prev = 1. #dznrm2(ODE._y)**2
                else:
                    norm2_prev = norm2_psi
                    t_prev = ODE.t
                    y_prev = ODE.y

            # after while loop
            # ----------------
            out_psi = normalize(ODE._y)
            e_call.step(k, out_psi)
            if self.steady_state:
                self.sumsteadystate(out_psi)
            if self.store_states:
                for ii in range(self.l_vec):
                    self.states_out[k, ii] = out_psi[ii]
        if not self.store_states:
            self.states_out[0, ii] = out_psi[ii]
        return np.array(self.states_out), np.array(self.ss_out), self.collapses


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _find_collapse(self, ODE, double norm2_psi,
                             double t_prev, np.ndarray[complex, ndim=1] y_prev,
                             double norm2_prev, double target_norm):
        # find collapse time to within specified tolerance
        cdef int ii = 0
        cdef double t_final = ODE.t
        cdef double t_guess, norm2_guess

        while ii < self.norm_steps:
            ii += 1
            if (t_final - t_prev) < self.norm_t_tol:
                t_prev = t_final
                y_prev = ODE.y
                break

            t_guess = (t_prev +
                (log(norm2_prev / target_norm)).real  /
                (log(norm2_prev / norm2_psi)).real    *
                (t_final - t_prev))
            if (t_guess - t_prev) < self.norm_t_tol:
                t_guess = t_prev + self.norm_t_tol

            ODE.t = t_prev
            ODE._y = y_prev
            ODE._integrator.call_args[3] = 1
            ODE.integrate(t_guess, step=0)
            if not ODE.successful():
                raise Exception("ZVODE failed after adjusting step size!")

            norm2_guess = dznrm2(ODE._y)**2
            if (np.abs(target_norm - norm2_guess) < self.norm_tol * target_norm):
                    break
            elif (norm2_guess < target_norm):
                # t_guess is still > t_jump
                t_final = t_guess
                norm2_psi = norm2_guess
            else:
                # t_guess < t_jump
                t_prev = t_guess
                y_prev = ODE.y
                norm2_prev = norm2_guess

        if ii > self.norm_steps:
            raise Exception("Norm tolerance not reached. " +
                            "Increase accuracy of ODE solver or " +
                            "Options.norm_steps.")

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _which_collapse(self, double t, np.ndarray[complex, ndim=1] y, double rand):
        # determine which operator does collapse
        cdef int ii, j = self.num_ops
        cdef double e, sum_ = 0
        cdef CQobjEvo cobj
        for ii in range(self.num_ops):
            cobj = <CQobjEvo> self.n_ops[ii].compiled_qobjevo
            e = real(cobj._expect(t, &y[0]))
            self.n_dp[ii] = e
            sum_ += e
        rand *= sum_
        for ii in range(self.num_ops):
            if rand <= self.n_dp[ii]:
                j = ii
                break
            else:
                rand -= self.n_dp[ii]
        return j

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[complex, ndim=1] _collapse(self, double t, int j,
                                               np.ndarray[complex, ndim=1] y):
        cdef CQobjEvo cobj
        cdef np.ndarray[complex, ndim=1] state
        cobj = <CQobjEvo> self.c_ops[j].compiled_qobjevo
        state = cobj.mul_vec(t, y)
        state = normalize(state)
        return state

"""
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_mc_run_diag(psi0, ss, tlist_, e_call, opt, prng):
    cdef np.ndarray[double, ndim=1] rand_vals
    cdef np.ndarray[double, ndim=1] tlist = np.array(tlist_)
    cdef np.ndarray[complex, ndim=1] y_prev
    cdef np.ndarray[complex, ndim=1] out_psi = psi0
    cdef np.ndarray[complex, ndim=2] states_out
    cdef np.ndarray[complex, ndim=2] ss_out
    cdef list collapses = []
    cdef list collapses_args
    cdef list c_ops = ss.td_c_ops
    cdef list n_ops = ss.td_n_ops
    cdef int num_times = tlist.shape[0]
    cdef int l_vec = out_psi.shape[0]
    cdef int num_ops = len(ss.td_n_ops)
    cdef int ii, which, k
    cdef int steady_state, store_states, col_args
    cdef int norm_steps
    cdef double norm_t_tol, norm_tol
    cdef double norm2_prev, norm2_psi, norm2_guess
    cdef double t_prev, t_final, t_guess

    norm_steps = opt.norm_steps
    norm_t_tol = opt.norm_t_tol
    norm_tol = opt.norm_tol
    steady_state = opt.steady_state_average
    store_states = opt.store_states or opt.average_states
    if ss.col_args:
        col_args = 1
        collapses_args = ss.args[ss.col_args]
        if ss.type == "QobjEvo":
            ss.H_td.coeff_get.get_args[ss.col_args] = collapses_args
            for c in ss.td_c_ops:
                c.coeff_get.get_args[ss.col_args] = collapses_args
            for c in ss.td_n_ops:
                c.coeff_get.get_args[ss.col_args] = collapses_args
    else:
        col_args = 0

    if steady_state:
        ss_out = np.zeros((l_vec, l_vec), dtype=complex)
        sumsteadystate(ss_out, out_psi)
    else:
        ss_out = np.zeros((0, 0), dtype=complex)

    if store_states:
        states_out = np.zeros((num_times, l_vec), dtype=complex)
        for ii in range(l_vec):
            states_out[0, ii] = out_psi[ii]
    else:
        states_out = np.zeros((1, l_vec), dtype=complex)

    e_call.step(0, out_psi)
    rand_vals = prng.rand(2)

    # RUN ODE UNTIL EACH TIME IN TLIST
    norm2_prev = dznrm2(psi0) ** 2
    for k in range(1, num_times):
        # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        t_prev = ODE.t
        y_prev = ODE.y
        while t_prev < tlist[k]:
            # integrate up to tlist[k], one step at a time.
            ODE.integrate(tlist[k], step=1)
            if not ODE.successful():
                print(ODE.t, t_prev, tlist[k])
                print(ODE._integrator.call_args)
                raise Exception("ZVODE failed!")
            norm2_psi = dznrm2(ODE._y) ** 2
            if norm2_psi <= rand_vals[0]:
                # collapse has occured:
                _find_collapse(ODE, norm2_psi, t_prev, y_prev,
                               norm2_prev, target_norm,
                               norm_steps, norm_t_tol, norm_tol)
                t_prev = ODE.t
                y_prev = ODE.y

                which = _which_collapse(t_prev, y_prev, n_ops, num_ops, rand_vals[1])
                y_prev = _collapse(t_prev, which, y_prev, c_ops)
                ODE.set_initial_value(y_prev, t_prev)

                collapses.append((t_prev, which))
                if col_args:
                    collapses_args.append((t_prev, which))
                rand_vals = prng.rand(2)
                norm2_prev = 1. #dznrm2(ODE._y)**2
            else:
                norm2_prev = norm2_psi
                t_prev = ODE.t
                y_prev = ODE.y

        # after while loop
        # ----------------
        out_psi = normalize(ODE._y)
        e_call.step(k, out_psi)
        if steady_state:
            sumsteadystate(ss_out, out_psi)
        if store_states:
            for ii in range(l_vec):
                states_out[k, ii] = out_psi[ii]
    if not store_states:
        states_out[0, ii] = out_psi[ii]
    return states_out, ss_out, collapses
"""
