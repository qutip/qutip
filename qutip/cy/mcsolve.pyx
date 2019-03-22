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
import scipy.sparse as sp
from qutip.qobj import Qobj
# from qutip.cy.dopri5 import ode_td_dopri
cimport numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dznrm2 as raw_dznrm2
from qutip.cy.cqobjevo cimport CQobjEvo
#from qutip.cy.complex_math cimport conj
include "complex_math.pxi"
from qutip.cy.spmatfuncs cimport cy_expect_psi

cdef int ONE = 1

cdef double dznrm2(complex[::1] psi):
    cdef int l = psi.shape[0]
    return raw_dznrm2(&l, <complex*>&psi[0], &ONE)

cdef complex[::1] normalize(complex[::1] psi):
    cdef int i, l = psi.shape[0]
    cdef double norm = dznrm2(psi)
    cdef complex[::1] out = np.empty(l, dtype=complex)
    for i in range(l):
        out[i] = psi[i] / norm
    return out

cdef void sumsteadystate(complex[:, ::1] mean, complex[::1] state):
    cdef int ii, jj, l_vec = state.shape[0]
    for ii in range(l_vec):
      for jj in range(l_vec):
        mean[ii,jj] += state[ii]*conj(state[jj])


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_mc_run_ode(ODE, ss, tlist, e_call, opt, prng):
    cdef int ii, j, jj, k, steady_state, store_states
    cdef int num_times = tlist.shape[0], l_vec = ODE._y.shape[0]
    cdef double norm2_psi, norm2_prev, norm2_guess, t_prev, t_final, t_guess
    cdef np.ndarray[double, ndim=1] rand_vals
    cdef np.ndarray[double, ndim=1] tlist = np.array(tlist)
    cdef np.ndarray[complex, ndim=1] y_prev
    cdef np.ndarray[complex, ndim=1] out_psi = ODE._y

    cdef np.ndarray[complex, ndim=2] states_out
    cdef np.ndarray[complex, ndim=2] ss_out
    cdef list collapses = []

    cdef CQobjEvo[::1] c_ops = ss.td_c_ops
    cdef CQobjEvo[::1] n_ops = ss.td_n_ops
    # make array for collapse operator inds
    cdef np.ndarray[long, ndim=1] cinds = np.arange(len(ss.td_c_ops))

    steady_state = opt.steady_state_average
    store_states = opt.store_states or opt.average_states

    if steady_state:
        ss_out = np.zeros((l_vec, l_vec), dtype=complex)
        sumsteadystate(ss_out, out_psi)
    if store_states:
        states_out = np.zeros((num_times, l_vec), dtype=complex)
        states_out[0, :] = out_psi

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
                # find collapse time to within specified tolerance
                # ------------------------------------------------
                ii = 0
                t_final = ODE.t
                while ii < config.norm_steps:
                    ii += 1
                    if (t_final - t_prev) < config.norm_t_tol:
                        t_prev = t_final
                        y_prev = ODE.y
                        break
                    t_guess = t_prev + \
                        np.log(norm2_prev / rand_vals[0]) / \
                        np.log(norm2_prev / norm2_psi) * (t_final - t_prev)
                    if (t_guess - t_prev) < config.norm_t_tol:
                        t_guess = t_prev + config.norm_t_tol
                    ODE._y = y_prev
                    ODE.t = t_prev
                    ODE._integrator.call_args[3] = 1
                    ODE.integrate(t_guess, step=0)
                    if not ODE.successful():
                        raise Exception(
                              "ZVODE failed after adjusting step size!")

                    norm2_guess = dznrm2(ODE._y)**2
                    if (np.abs(rand_vals[0] - norm2_guess) <
                                config.norm_tol * rand_vals[0]):
                            norm2_psi = norm2_guess
                            t_prev = t_guess
                            y_prev = ODE.y
                            break
                    elif (norm2_guess < rand_vals[0]):
                        # t_guess is still > t_jump
                        t_final = t_guess
                        norm2_psi = norm2_guess
                    else:
                        # t_guess < t_jump
                        t_prev = t_guess
                        y_prev = ODE.y
                        norm2_prev = norm2_guess
                if ii > config.norm_steps:
                    raise Exception("Norm tolerance not reached. " +
                                    "Increase accuracy of ODE solver or " +
                                    "Options.norm_steps.")

                collapse_times.append(ODE.t)
                # some string based collapse operators
                n_dp = [op._expect(ODE.t, ODE._y).real() for op in n_ops]
                # determine which operator does collapse and store it
                kk = np.cumsum(n_dp / np.sum(n_dp))
                j = cinds[kk >= rand_vals[1]][0]
                which_oper.append(j)
                state = c_ops[j].mat_mul(ODE.t, ODE._y)
                state = normalize(state)
                ODE.set_initial_value(state, t_prev)
                rand_vals = prng.rand(2)
                norm2_prev = dznrm2(ODE._y)**2
            else:
                norm2_prev = norm2_psi
                t_prev = ODE.t
                y_prev = ODE.y

        # after while loop
        # ----------------
        out_psi = normalize(ODE._y)

        e_call.step(ii, out_psi)
        if steady_state_average:
            sumsteadystate(ss_out, out_psi)
        if store_states:
            states_out[k, :] = out_psi

    return states_out, expect_out, collapse_times, which_oper
