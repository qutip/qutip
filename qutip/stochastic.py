# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
This module contains experimental functions for solving stochastic schrodinger
and master equations. The API should not be considered stable, and is subject
to change when we work more on optimizing this module for performance and
features.

Release target: 2.3.0 or 3.0.0

Todo:

1) test and debug

2) store measurement records

3) add more sme solvers

4) cythonize some rhs or d1,d2 functions

5) parallelize

"""

import inspect

import numpy as np
import scipy
from scipy.linalg import norm
from numpy.random import RandomState

from qutip.odedata import Odedata
from qutip.odeoptions import Odeoptions
from qutip.expect import expect, expect_rho_vec, expect_rho_vec1d
from qutip.qobj import Qobj, isket
from qutip.superoperator import (spre, spost, mat2vec, vec2mat,
                                 liouvillian_fast, lindblad_dissipator)
from qutip.states import ket2dm
from qutip.cyQ.spmatfuncs import cy_expect, spmv, cy_expect_rho_vec
from qutip.gui.progressbar import TextProgressBar

debug = True

class _StochasticSolverData:
    """
    Internal class for passing data between stochastic solver functions.
    """
    def __init__(self, H=None, state0=None, tlist=None, 
                 c_ops=[], sc_ops=[], e_ops=[], ntraj=1, nsubsteps=1,
                 d1=None, d2=None, d2_len=1, rhs=None, homogeneous=True,
                 solver=None, method=None, distribution='normal',
                 store_measurement=False, noise=None,
                 options=Odeoptions(), progress_bar=TextProgressBar()):

        self.H = H
        self.d1 = d1
        self.d2 = d2
        self.d2_len = d2_len
        self.state0 = state0
        self.tlist = tlist
        self.c_ops = c_ops
        self.sc_ops = sc_ops
        self.e_ops = e_ops
        self.ntraj = ntraj
        self.nsubsteps = nsubsteps
        self.solver = solver
        self.method = method
        self.distribution = distribution
        self.homogeneous = homogeneous
        self.rhs = rhs
        self.options = options
        self.progress_bar = progress_bar
        self.store_measurement = store_measurement
        self.noise = noise


def ssesolve(H, psi0, tlist, sc_ops, e_ops, **kwargs):
    """
    Solve stochastic Schrodinger equation. Dispatch to specific solvers
    depending on the value of the `solver` argument.

    .. note::

        Experimental. tlist must be uniform.

    """
    if debug:
        print(inspect.stack()[0][3])

    ssdata = _StochasticSolverData(H=H, state0=psi0, tlist=tlist,
                                   sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    if (ssdata.d1 is None) or (ssdata.d2 is None):

        if ssdata.method == 'homodyne':
            ssdata.d1 = d1_psi_homodyne
            ssdata.d2 = d2_psi_homodyne
            ssdata.d2_len = 1
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'

        elif ssdata.method == 'heterodyne':
            ssdata.d1 = d1_psi_heterodyne
            ssdata.d2 = d2_psi_heterodyne
            ssdata.d2_len = 2
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'

        elif ssdata.method == 'photocurrent':
            ssdata.d1 = d1_psi_photocurrent
            ssdata.d2 = d2_psi_photocurrent
            ssdata.d2_len = 1
            ssdata.homogeneous = False
            ssdata.distribution = 'poisson'

        else:
            raise Exception("Unrecognized method '%s'." % ssdata.method)

    if ssdata.solver == 'euler-maruyama' or ssdata.solver == None:
        ssdata.rhs_func = _rhs_psi_euler_maruyama
        return ssesolve_generic(ssdata, ssdata.options, ssdata.progress_bar)

    elif ssdata.solver == 'platen':
        ssdata.rhs_func = _rhs_psi_platen
        return ssesolve_generic(ssdata, ssdata.options, ssdata.progress_bar)

    else:
        raise Exception("Unrecognized solver '%s'." % ssdata.solver)


def smesolve(H, rho0, tlist, c_ops, sc_ops, e_ops, **kwargs):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` argument.

    .. note::

        Experimental. tlist must be uniform.

    """
    if debug:
        print(inspect.stack()[0][3])

    if isket(rho0):
        rho0 = ket2dm(rho0)

    ssdata = _StochasticSolverData(H=H, state0=rho0, tlist=tlist, c_ops=c_ops,
                                   sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    if (ssdata.d1 is None) or (ssdata.d2 is None):

        if ssdata.method == 'homodyne' or ssdata.method is None:
            ssdata.d1 = d1_rho_homodyne
            ssdata.d2 = d2_rho_homodyne
            ssdata.d2_len = 1
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'

        elif ssdata.method == 'heterodyne':
            ssdata.d1 = d1_rho_heterodyne
            ssdata.d2 = d2_rho_heterodyne
            ssdata.d2_len = 2
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'

        elif ssdata.method == 'photocurrent':
            ssdata.d1 = d1_rho_photocurrent
            ssdata.d2 = d2_rho_photocurrent
            ssdata.d2_len = 1
            ssdata.homogeneous = False
            ssdata.distribution = 'poisson'

        else:
            raise Exception("Unrecognized method '%s'." % ssdata.method)

    if ssdata.distribution == 'poisson':
        ssdata.homogeneous = False

    if ssdata.rhs is None:
        if ssdata.solver == 'euler-maruyama' or ssdata.solver == None:
            ssdata.rhs = _rhs_rho_euler_maruyama
        else:
            raise Exception("Unrecognized solver '%s'." % ssdata.solver)

    return smesolve_generic(ssdata, ssdata.options, ssdata.progress_bar)


def sepdpsolve(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=1, nsubsteps=10,
               options=Odeoptions(), progress_bar=TextProgressBar()):
    """
    A stochastic PDP solver for experimental/development and comparison to the 
    stochastic DE solvers. Use mcsolve for real quantum trajectory
    simulations.
    """
    if debug:
        print(inspect.stack()[0][3])

    ssdata = _StochasticSolverData()
    ssdata.H = H
    ssdata.psi0 = psi0
    ssdata.tlist = tlist
    ssdata.c_ops = c_ops
    ssdata.e_ops = e_ops
    ssdata.ntraj = ntraj
    ssdata.nsubsteps = nsubsteps

    return sepdpsolve_generic(ssdata, options, progress_bar)


def smepdpsolve(H, rho0, tlist, c_ops=[], e_ops=[], ntraj=1, nsubsteps=10,
                options=Odeoptions(), progress_bar=TextProgressBar()):
    """
    A stochastic PDP solver for density matrix evolution.
    """
    if debug:
        print(inspect.stack()[0][3])

    ssdata = _StochasticSolverData()
    ssdata.H = H
    ssdata.rho0 = rho0
    ssdata.tlist = tlist
    ssdata.c_ops = c_ops
    ssdata.e_ops = e_ops
    ssdata.ntraj = ntraj
    ssdata.nsubsteps = nsubsteps

    return smepdpsolve_generic(ssdata, options, progress_bar)


#------------------------------------------------------------------------------
# Generic parameterized stochastic Schrodinger equation solver
#
def ssesolve_generic(ssdata, options, progress_bar):
    """
    internal

    .. note::

        Experimental.

    """
    if debug:
        print(inspect.stack()[0][3])

    N_store = len(ssdata.tlist)
    N_substeps = ssdata.nsubsteps
    N = N_store * N_substeps
    dt = (ssdata.tlist[1] - ssdata.tlist[0]) / N_substeps
    NT = ssdata.ntraj

    data = Odedata()
    data.solver = "ssesolve"
    data.times = ssdata.tlist
    data.expect = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.ss = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.noise = []
    data.measurement = []

    # pre-compute collapse operator combinations that are commonly needed
    # when evaluating the RHS of stochastic Schrodinger equations
    A_ops = []
    for c_idx, c in enumerate(ssdata.sc_ops):
        A_ops.append([c.data,
                      (c + c.dag()).data,
                      (c - c.dag()).data,
                      (c.dag() * c).data])

    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)

        psi_t = ssdata.state0.full()

        noise = ssdata.noise[n] if ssdata.noise else None

        states_list, dW, m = _ssesolve_single_trajectory(
            ssdata.H, dt, ssdata.tlist, N_store, N_substeps, psi_t, A_ops,
            ssdata.e_ops, data, ssdata.rhs_func, ssdata.d1, ssdata.d2,
            ssdata.d2_len, ssdata.homogeneous, ssdata.distribution,
            store_measurement=ssdata.store_measurement, noise=noise)

        # if average -> average...
        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)

    progress_bar.finished()

    # average
    data.expect = data.expect / NT

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n,:]) if e.isherm else data.expect[n,:]
                   for n, e in enumerate(ssdata.e_ops)]

    return data


def _ssesolve_single_trajectory(H, dt, tlist, N_store, N_substeps, psi_t,
                                A_ops, e_ops, data, rhs, d1, d2, d2_len,
                                homogeneous, distribution,
                                store_measurement=False, noise=None):
    """
    Internal function. See ssesolve.
    """

    if noise is None:
        if homogeneous:
            if distribution == 'normal':
                dW = np.sqrt(dt) * scipy.randn(len(A_ops), N_store, N_substeps, d2_len)
            else:
                raise TypeError('Unsupported increment distribution for homogeneous process.')
        else:
            if distribution != 'poisson':
                raise TypeError('Unsupported increment distribution for inhomogeneous process.')

            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = noise

    states_list = []
    measurements = np.zeros((len(tlist), len(A_ops)), dtype=complex)

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect(e.data.data, e.data.indices, e.data.indptr, psi_t)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t))

        psi_prev = np.copy(psi_t)

        for j in range(N_substeps):

            dpsi_t = (-1.0j * dt) * (H.data * psi_t)

            for a_idx, A in enumerate(A_ops):

                if noise is None and not homogeneous:
                    dw_expect = norm(spmv(A[0].data, A[0].indices, A[0].indptr, psi_t)) ** 2 * dt
                    dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect, d2_len)

                dpsi_t += rhs(H.data, psi_t, A, dt, dW[a_idx, t_idx, j, :], d1, d2)

            # increment and renormalize the wave function
            psi_t += dpsi_t
            psi_t /= norm(psi_t)

        if store_measurement:
            for a_idx, A in enumerate(A_ops):
                measurements[t_idx, a_idx] = norm(spmv(A[0].data, A[0].indices, A[0].indptr, psi_prev)) ** 2 * dt * N_substeps + dW[a_idx, t_idx, :, 0].sum()


    return states_list, dW, measurements


#------------------------------------------------------------------------------
# Generic parameterized stochastic master equation solver
#
def smesolve_generic(ssdata, options, progress_bar):
    """
    internal

    .. note::

        Experimental.

    """
    if debug:
        print(inspect.stack()[0][3])

    N_store = len(ssdata.tlist)
    N_substeps = ssdata.nsubsteps
    N = N_store * N_substeps
    dt = (ssdata.tlist[1] - ssdata.tlist[0]) / N_substeps
    NT = ssdata.ntraj

    data = Odedata()
    data.solver = "smesolve"
    data.times = ssdata.tlist
    data.expect = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.ss = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.noise = []
    data.measurement = []

    # pre-compute suporoperator operator combinations that are commonly needed
    # when evaluating the RHS of stochastic master equations
    A_ops = []
    for c_idx, c in enumerate(ssdata.sc_ops):

        n = c.dag() * c
        A_ops.append([spre(c).data, spost(c).data,
                      spre(c.dag()).data, spost(c.dag()).data,
                      spre(n).data, spost(n).data,
                      (spre(c) * spost(c.dag())).data,
                      lindblad_dissipator(c, data_only=True)])

    s_e_ops = [spre(e) for e in ssdata.e_ops]

    # Liouvillian for the deterministic part.
    # needs to be modified for TD systems
    L = liouvillian_fast(ssdata.H, ssdata.c_ops)

    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)

        rho_t = mat2vec(ssdata.state0.full())

        noise = ssdata.noise[n] if ssdata.noise else None

        states_list, dW, m = _smesolve_single_trajectory(
            L, dt, ssdata.tlist, N_store, N_substeps,
            rho_t, A_ops, s_e_ops, data, ssdata.rhs,
            ssdata.d1, ssdata.d2, ssdata.d2_len, ssdata.homogeneous,
            ssdata.distribution, store_measurement=ssdata.store_measurement,
            noise=noise)

        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)

    progress_bar.finished()

    # if options.state_average -> average data.states

    # average
    data.expect = data.expect / NT

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n,:]) if e.isherm else data.expect[n,:]
                   for n, e in enumerate(ssdata.e_ops)]

    return data


def _smesolve_single_trajectory(L, dt, tlist, N_store, N_substeps, rho_t,
                                A_ops, e_ops, data, rhs, d1, d2, d2_len,
                                homogeneous, distribution,
                                store_measurement=False, noise=None):
    """
    Internal function. See smesolve.
    """

    if noise is None:
        if homogeneous:
            if distribution == 'normal':
                dW = np.sqrt(dt) * scipy.randn(len(A_ops), N_store, N_substeps, d2_len)    
            else:
                raise TypeError('Unsupported increment distribution for homogeneous process.')
        else:
            if distribution != 'poisson':
                raise TypeError('Unsupported increment distribution for inhomogeneous process.')

            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = noise

    states_list = []
    measurements = np.zeros((len(tlist), len(A_ops)), dtype=complex)

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = expect_rho_vec(e.data, rho_t)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2 
        else:
            # XXX: need to keep hilbert space structure
            states_list.append(Qobj(vec2mat(rho_t)))

        rho_prev = np.copy(rho_t)

        for j in range(N_substeps):

            drho_t = spmv(L.data.data,
                          L.data.indices,
                          L.data.indptr, rho_t) * dt

            for a_idx, A in enumerate(A_ops):
                if noise is None and not homogeneous:
                    dw_expect = np.real(cy_expect_rho_vec(A[4], rho_t)) * dt
                    dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect, d2_len)

                drho_t += rhs(L.data, rho_t, A, dt, dW[a_idx, t_idx, j, :], d1, d2)

            rho_t += drho_t

        if store_measurement:
            for a_idx, A in enumerate(A_ops):
                measurements[t_idx, a_idx] = cy_expect_rho_vec(A[0], rho_prev) * dt * N_substeps + dW[a_idx, t_idx, :, 0].sum()

    return states_list, dW, measurements


#------------------------------------------------------------------------------
# Generic parameterized stochastic SE PDP solver
#
def sepdpsolve_generic(ssdata, options, progress_bar):
    """
    For internal use.

    .. note::

        Experimental.

    """
    if debug:
        print(inspect.stack()[0][3])

    N_store = len(ssdata.tlist)
    N_substeps = ssdata.nsubsteps
    N = N_store * N_substeps
    dt = (ssdata.tlist[1] - ssdata.tlist[0]) / N_substeps
    NT = ssdata.ntraj

    data = Odedata()
    data.solver = "sepdpsolve"
    data.times = ssdata.tlist
    data.expect = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.ss = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.jump_times = []
    data.jump_op_idx = []

    # effective hamiltonian for deterministic part
    Heff = ssdata.H
    for c in ssdata.c_ops:
        Heff += -0.5j * c.dag() * c
        
    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)
        psi_t = ssdata.psi0.full()

        states_list, jump_times, jump_op_idx = \
            _sepdpsolve_single_trajectory(Heff, dt, ssdata.tlist,
                                          N_store, N_substeps,
                                          psi_t, ssdata.c_ops, ssdata.e_ops, 
                                          data)

        data.states.append(states_list)
        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # average
    data.expect = data.expect / NT

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n,:]) if e.isherm else data.expect[n,:]
                   for n, e in enumerate(ssdata.e_ops)]

    return data


def _sepdpsolve_single_trajectory(Heff, dt, tlist, N_store, N_substeps, psi_t,
                                  c_ops, e_ops, data):
    """
    Internal function.
    """
    states_list = []

    phi_t = np.copy(psi_t)

    prng = RandomState() # todo: seed it
    r_jump, r_op = prng.rand(2)

    jump_times = []
    jump_op_idx = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect(e.data.data, e.data.indices, e.data.indptr, psi_t)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t))

        for j in range(N_substeps):

            if norm(phi_t) ** 2 < r_jump:
                # jump occurs
                p = np.array([norm(c.data * psi_t) ** 2 for c in c_ops])
                p = np.cumsum(p / np.sum(p))
                n = np.where(p >= r_op)[0][0]

                # apply jump
                psi_t = c_ops[n].data * psi_t
                psi_t /= norm(psi_t)
                phi_t = np.copy(psi_t)

                # store info about jump
                jump_times.append(tlist[t_idx] + dt * j)
                jump_op_idx.append(n)

                # get new random numbers for next jump
                r_jump, r_op = prng.rand(2)

            # deterministic evolution wihtout correction for norm decay
            dphi_t = (-1.0j * dt) * (Heff.data * phi_t)

            # deterministic evolution with correction for norm decay
            dpsi_t = (-1.0j * dt) * (Heff.data * psi_t)
            A = 0.5 * np.sum([norm(c.data * psi_t) ** 2 for c in c_ops])
            dpsi_t += dt * A * psi_t

            # increment wavefunctions
            phi_t += dphi_t
            psi_t += dpsi_t

            # ensure that normalized wavefunction remains normalized
            # this allows larger time step than otherwise would be possible
            psi_t /= norm(psi_t)

    return states_list, jump_times, jump_op_idx


#------------------------------------------------------------------------------
# Generic parameterized stochastic ME PDP solver
#
def smepdpsolve_generic(ssdata, options, progress_bar):
    """
    For internal use.

    .. note::

        Experimental.

    """
    if debug:
        print(inspect.stack()[0][3])

    N_store = len(ssdata.tlist)
    N_substeps = ssdata.nsubsteps
    N = N_store * N_substeps
    dt = (ssdata.tlist[1] - ssdata.tlist[0]) / N_substeps
    NT = ssdata.ntraj

    data = Odedata()
    data.solver = "smepdpsolve"
    data.times = ssdata.tlist
    data.expect = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.jump_times = []
    data.jump_op_idx = []

    # Liouvillian for the deterministic part.
    # needs to be modified for TD systems
    L = liouvillian_fast(ssdata.H, ssdata.c_ops)
        
    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)
        rho_t = mat2vec(ssdata.rho0.full())

        states_list, jump_times, jump_op_idx = \
            _smepdpsolve_single_trajectory(L, dt, ssdata.tlist,
                                           N_store, N_substeps,
                                           rho_t, ssdata.c_ops, ssdata.e_ops, 
                                           data)

        data.states.append(states_list)
        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # if options.state_average = True -> average data.states

    # average
    data.expect = data.expect / ssdata.ntraj

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    return data


def _smepdpsolve_single_trajectory(L, dt, tlist, N_store, N_substeps, rho_t,
                                   c_ops, e_ops, data):
    """ 
    Internal function.
    """
    states_list = []

    rho_t = np.copy(rho_t)

    prng = RandomState() # todo: seed it
    r_jump, r_op = prng.rand(2)

    jump_times = []
    jump_op_idx = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                data.expect[e_idx, t_idx] += expect_rho_vec(e, rho_t)
        else:
            states_list.append(Qobj(vec2mat(rho_t)))

        for j in range(N_substeps):

            if expect_rho_vec(d_op, sigma_t) < r_jump:
                # jump occurs
                p = np.array([rho_expect(c.dag() * c, rho_t) for c in c_ops])
                p = np.cumsum(p / np.sum(p))
                n = np.where(p >= r_op)[0][0]

                # apply jump
                rho_t = c_ops[n] * psi_t * c_ops[n].dag()
                rho_t /= rho_expect(c.dag() * c, rho_t)
                rho_t = np.copy(rho_t)

                # store info about jump
                jump_times.append(tlist[t_idx] + dt * j)
                jump_op_idx.append(n)

                # get new random numbers for next jump
                r_jump, r_op = prng.rand(2)

            # deterministic evolution wihtout correction for norm decay
            dsigma_t = spmv(L.data.data,
                            L.data.indices,
                            L.data.indptr, sigma_t) * dt

            # deterministic evolution with correction for norm decay
            drho_t = spmv(L.data.data,
                          L.data.indices,
                          L.data.indptr, rho_t) * dt

            rho_t += drho_t

            # increment density matrices
            sigma_t += dsigma_t
            rho_t += drho_t

    return states_list, jump_times, jump_op_idx


#------------------------------------------------------------------------------
# Helper-functions for stochastic DE
#
# d1 = deterministic part of the contribution to the DE RHS function, to be
#      multiplied by the increament dt
#
# d1 = stochastic part of the contribution to the DE RHS function, to be
#      multiplied by the increament dW
#


#
# For SSE
#

# Function sigurature:
#
# def d(A, psi):
#
#     psi = wave function at the current time step
#
#     A[0] = c
#     A[1] = c + c.dag()
#     A[2] = c - c.dag()
#     A[3] = c.dag() * c
#
#     where c is a collapse operator. The combinations of c's stored in A are
#     precomputed before the time-evolution is started to avoid repeated
#     computations.

def d1_psi_homodyne(A, psi):
    """
    OK
    Todo: cythonize

    .. math::

        D_1(\psi, t) = \\frac{1}{2}(\\langle C + C^\\dagger\\rangle\\psi - 
        C^\\dagger C\\psi - \\frac{1}{4}\\langle C + C^\\dagger\\rangle^2\\psi)

    """

    e1 = cy_expect(A[1].data, A[1].indices, A[1].indptr, psi)
    return 0.5 * (e1 * spmv(A[0].data, A[0].indices, A[0].indptr, psi) -
                  spmv(A[3].data, A[3].indices, A[3].indptr, psi) -
                  0.25 * e1 ** 2 * psi)


def d2_psi_homodyne(A, psi):
    """
    OK
    Todo: cythonize

    .. math::

        D_2(\psi, t) = (C - \\frac{1}{2}\\langle C + C^\\dagger\\rangle)\\psi

    """

    e1 = cy_expect(A[1].data, A[1].indices, A[1].indptr, psi)
    return [spmv(A[0].data, A[0].indices, A[0].indptr, psi) - 0.5 * e1 * psi]


def d1_psi_heterodyne(A, psi):
    """
    Todo: cythonize

    .. math::

        D_1(\psi, t) = -\\frac{1}{2}(C^\\dagger C - \\langle C^\\dagger \\rangle C + 
                        \\frac{1}{2}\\langle C \\rangle\\langle C^\\dagger \\rangle))\psi

    """
    e_C = cy_expect(A[0].data, A[0].indices, A[0].indptr, psi) # e_C
    B = A[0].T.conj()
    e_Cd = cy_expect(B.data, B.indices, B.indptr, psi) # e_Cd

    return  (-0.5 * spmv(A[3].data, A[3].indices, A[3].indptr, psi) +
             0.5 * e_Cd * spmv(A[0].data, A[0].indices, A[0].indptr, psi) -
             0.25 * e_C * e_Cd * psi)


def d2_psi_heterodyne(A, psi):
    """
    Todo: cythonize

        X = \\frac{1}{2}(C + C^\\dagger)

        Y = \\frac{1}{2}(C - C^\\dagger)

        D_{2,1}(\psi, t) = \\sqrt(1/2) * (C - \\langle X \\rangle) \\psi

        D_{2,2}(\psi, t) = -i\\sqrt(1/2) * (C - \\langle Y \\rangle) \\psi

    """

    X = 0.5 * cy_expect(A[1].data, A[1].indices, A[1].indptr, psi)
    Y = 0.5 * cy_expect(A[2].data, A[2].indices, A[2].indptr, psi)

    d2_1 = np.sqrt(0.5) * (spmv(A[0].data, A[0].indices, A[0].indptr, psi) - X * psi)
    d2_2 = -1.0j * np.sqrt(0.5) * (spmv(A[0].data, A[0].indices, A[0].indptr, psi) - Y * psi)

    return [d2_1, d2_2]


def d1_psi_photocurrent(A, psi):
    """
    Todo: cythonize.

    Note: requires poisson increments

    .. math::

        D_1(\psi, t) = - \\frac{1}{2}(C^\dagger C \psi - ||C\psi||^2 \psi)

    """
    return (-0.5 * (spmv(A[3].data, A[3].indices, A[3].indptr, psi)
            -norm(spmv(A[0].data, A[0].indices, A[0].indptr, psi)) ** 2 * psi))


def d2_psi_photocurrent(A, psi):
    """
    Todo: cythonize

    Note: requires poisson increments

    .. math::

        D_2(\psi, t) = C\psi / ||C\psi|| - \psi

    """
    psi_1 = spmv(A[0].data, A[0].indices, A[0].indptr, psi)
    n1 = norm(psi_1)
    if n1 != 0:
        return psi_1 / n1 - psi
    else:
        return - psi


#
# For SME
#

# def d(A, rho_vec):
#
#     rho = density operator in vector form at the current time stemp
#
#     A[0] = spre(c)
#     A[1] = spost(c)
#     A[2] = spre(c.dag())
#     A[3] = spost(c.dag())
#     A[4] = spre(n)
#     A[5] = spost(n)
#     A[6] = (spre(c) * spost(c.dag())
#     A[7] = lindblad_dissipator(c)

def d1_rho_homodyne(A, rho_vec):
    """
    
    D1[a] rho = lindblad_dissipator(a) * rho

    Todo: cythonize
    """
    return spmv(A[7].data, A[7].indices, A[7].indptr, rho_vec)


def d2_rho_homodyne(A, rho_vec):
    """

    D2[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger]
              = (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec]

    Todo: cythonize, add A_L + Ad_R to precomputed operators
    """
    M = A[0] + A[3]

    e1 = cy_expect_rho_vec(M, rho_vec)
    return [spmv(M.data, M.indices, M.indptr, rho_vec) - e1 * rho_vec]


def d1_rho_heterodyne(A, rho_vec):
    """
    todo: cythonize, docstrings
    """
    return spmv(A[7].data, A[7].indices, A[7].indptr, rho_vec)


def d2_rho_heterodyne(A, rho_vec):
    """
    todo: cythonize, docstrings
    """
    M = A[0] + A[3]
    e1 = cy_expect_rho_vec(M, rho_vec)
    d1 = spmv(M.data, M.indices, M.indptr, rho_vec) - e1 * rho_vec
    M = A[0] - A[3]
    e1 = cy_expect_rho_vec(M, rho_vec)
    d2 = spmv(M.data, M.indices, M.indptr, rho_vec) - e1 * rho_vec
    return [1.0/np.sqrt(2) * d1, -1.0j/np.sqrt(2) * d2]


def d1_rho_photocurrent(A, rho_vec):
    """
    Todo: cythonize, add (AdA)_L + AdA_R to precomputed operators
    """
    n_sum = A[4] + A[5]
    e1 = cy_expect_rho_vec(n_sum, rho_vec)
    return -spmv(n_sum.data, n_sum.indices, n_sum.indptr, rho_vec) + e1 * rho_vec


def d2_rho_photocurrent(A, rho_vec):
    """
    Todo: cythonize, add (AdA)_L + AdA_R to precomputed operators
    """
    e1 = cy_expect_rho_vec(A[6], rho_vec) + 1e-15
    return [spmv(A[6].data, A[6].indices, A[6].indptr, rho_vec) / e1 - rho_vec]


#------------------------------------------------------------------------------
# Euler-Maruyama rhs functions for the stochastic Schrodinger and master
# equations
#
def _rhs_psi_euler_maruyama(H, psi_t, A, dt, dW, d1, d2):
    """
    .. note::

        Experimental.

    """
    d2_vec = d2(A, psi_t)
    return d1(A, psi_t) * dt + sum([d2_vec[n] * dW[n] for n in range(len(dW)) if dW[n] != 0])


def _rhs_rho_euler_maruyama(L, rho_t, A, dt, dW, d1, d2):
    """
    .. note::

        Experimental.

    """
    d2_vec = d2(A, rho_t)
    return d1(A, rho_t) * dt + sum([d2_vec[n] * dW[n] for n in range(len(dW)) if dW[n] != 0])


#------------------------------------------------------------------------------
# Platen method
#
def _rhs_psi_platen(H, psi_t, A, dt, dW, d1, d2):
    """
    TODO: support multiple stochastic increments

    .. note::

        Experimental.

    """

    sqrt_dt = np.sqrt(dt)

    dpsi_t_H = (-1.0j * dt) * spmv(H.data, H.indices, H.indptr, psi_t)

    psi_t_1 = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t)[0] * dW[0]
    psi_t_p = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t)[0] * sqrt_dt
    psi_t_m = psi_t + dpsi_t_H + d1(A, psi_t) * dt - d2(A, psi_t)[0] * sqrt_dt

    dpsi_t = 0.50 * (d1(A, psi_t_1) + d1(A, psi_t)) * dt + \
        0.25 * (d2(A, psi_t_p)[0] + d2(A, psi_t_m)[0] + 2 * d2(A, psi_t)[0]) * dW[0] + \
        0.25 * (d2(A, psi_t_p)[0] - d2(A, psi_t_m)[0]) * (dW[0] ** 2 - dt) * sqrt_dt

    return dpsi_t
