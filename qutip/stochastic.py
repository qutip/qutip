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
from qutip.expect import expect
from qutip.qobj import Qobj
from qutip.superoperator import spre, spost, mat2vec, vec2mat, liouvillian_fast
from qutip.cyQ.spmatfuncs import cy_expect, spmv
from qutip.gui.progressbar import TextProgressBar

debug = True

class _StochasticSolverData:
    """
    Internal class for passing data between stochastic solver functions.
    """
    def __init__(self, H=None, state0=None, tlist=None, 
                 c_ops=[], sc_ops=[], e_ops=[], ntraj=1, nsubsteps=1,
                 solver=None, method=None):

        self.H = H
        self.state0 = state0
        self.tlist = tlist
        self.c_ops = c_ops
        self.sc_ops = sc_ops
        self.e_ops = e_ops
        self.ntraj = ntraj
        self.nsubsteps = nsubsteps
        self.solver = None
        self.method = None

def ssesolve(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=1,
             solver='euler-maruyama', method='homodyne',
             nsubsteps=10, d1=None, d2=None, d2_len=1, rhs=None,
             options=Odeoptions(), progress_bar=TextProgressBar()):
    """
    Solve stochastic Schrodinger equation. Dispatch to specific solvers
    depending on the value of the `solver` argument.

    .. note::

        Experimental. tlist must be uniform.

    """
    if debug:
        print(inspect.stack()[0][3])

    ssdata = _StochasticSolverData(H=H, state0=psi0, tlist=tlist, c_ops=c_ops,
                                   e_ops=e_ops, ntraj=ntraj,
                                   nsubsteps=nsubsteps, solver=solver, 
                                   method=method)

    if (d1 is None) or (d2 is None):

        if method == 'homodyne':
            ssdata.d1 = d1_psi_homodyne
            ssdata.d2 = d2_psi_homodyne
            ssdata.d2_len = 1
            ssdata.homogeneous = True
            ssdata.incr_distr = 'normal'

        elif method == 'heterodyne':
            ssdata.d1 = d1_psi_heterodyne
            ssdata.d2 = d2_psi_heterodyne
            ssdata.d2_len = 2
            ssdata.homogeneous = True
            ssdata.incr_distr = 'normal'

        elif method == 'photocurrent':
            ssdata.d1 = d1_psi_photocurrent
            ssdata.d2 = d2_psi_photocurrent
            ssdata.d2_len = 1
            ssdata.homogeneous = False
            ssdata.incr_distr = 'poisson'

        else:
            raise Exception("Unrecognized method '%s'." % method)

    if solver == 'euler-maruyama':
        ssdata.rhs_func = _rhs_psi_euler_maruyama
        return ssesolve_generic(ssdata, options, progress_bar)

    elif solver == 'platen':
        ssdata.rhs_func = _rhs_psi_platen
        return ssesolve_generic(ssdata, options, progress_bar)

    elif solver == 'milstein':
        raise NotImplementedError("Solver '%s' not yet implemented." % solver)

    else:
        raise Exception("Unrecongized solver '%s'." % solver)


def smesolve(H, psi0, tlist, c_ops=[], sc_ops=[], e_ops=[], ntraj=1,
             d1=None, d2=None, d2_len=1, rhs=None,
             method='homodyne', solver='euler-maruyama', nsubsteps=10,
             options=Odeoptions(), progress_bar=TextProgressBar()):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` argument.

    .. note::

        Experimental. tlist must be uniform.

    """
    if debug:
        print(inspect.stack()[0][3])

    if (d1 is None) or (d2 is None):

        if method == 'homodyne':
            d1 = d1_rho_homodyne
            d2 = d2_rho_homodyne
        else:
            raise Exception("Unregognized method '%s'." % method)

    if rhs is None:
        if solver == 'euler-maruyama':
            rhs = _rhs_rho_euler_maruyama
        else:
            raise Exception("Unrecongized solver '%s'." % solver)

    return smesolve_generic(H, psi0, tlist, c_ops, sc_ops, e_ops,
                            rhs, d1, d2, d2_len, ntraj, nsubsteps,
                            options, progress_bar)

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

    # pre-compute collapse operator combinations that are commonly needed
    # when evaluating the RHS of stochastic Schrodinger equations
    A_ops = []
    for c_idx, c in enumerate(ssdata.c_ops):
        A_ops.append([c.data,
                      (c + c.dag()).data,
                      (c - c.dag()).data,
                      (c.dag() * c).data])

    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)

        psi_t = ssdata.state0.full()

        states_list = _ssesolve_single_trajectory(ssdata.H, dt, ssdata.tlist, N_store,
                                                  N_substeps, psi_t, A_ops,
                                                  ssdata.e_ops, data, ssdata.rhs_func,
                                                  ssdata.d1, ssdata.d2, ssdata.d2_len,
                                                  ssdata.homogeneous, ssdata)

        # if average -> average...
        data.states.append(states_list)

    progress_bar.finished()

    # average
    data.expect = data.expect / NT

    # standard error
    data.ss = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))

    return data


def _ssesolve_single_trajectory(H, dt, tlist, N_store, N_substeps, psi_t,
                                A_ops, e_ops, data, rhs, d1, d2, d2_len,
                                homogeneous, ssdata):
    """
    Internal function. See ssesolve.
    """

    if homogeneous:
        if ssdata.incr_distr == 'normal':
            dW = np.sqrt(dt) * scipy.randn(len(A_ops), N_store, N_substeps, d2_len)
        else:
            raise TypeError('Unsupported increment distribution for homogeneous process.')
    else:
        if ssdata.incr_distr != 'poisson':
            raise TypeError('Unsupported increment distribution for inhomogeneous process.')


    states_list = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect(e.data.data, e.data.indices, e.data.indptr, 0, psi_t)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t))

        for j in range(N_substeps):

            dpsi_t = (-1.0j * dt) * (H.data * psi_t)

            for a_idx, A in enumerate(A_ops):

                if homogeneous:
                    dw = dW[a_idx, t_idx, j, :]
                else:
                    dw_expect = norm(spmv(A[0].data, A[0].indices, A[0].indptr, psi_t)) ** 2 * dt
                    dw = np.random.poisson(dw_expect, d2_len)

                dpsi_t += rhs(H.data, psi_t, A, dt, dw, d1, d2)

            # increment and renormalize the wave function
            psi_t += dpsi_t
            psi_t /= norm(psi_t, 2)

    return states_list


#------------------------------------------------------------------------------
# Generic parameterized stochastic master equation solver
#
def smesolve_generic(H, rho0, tlist, c_ops, sc_ops, e_ops,
                     rhs, d1, d2, d2_len, ntraj, nsubsteps,
                     options, progress_bar):
    """
    internal

    .. note::

        Experimental.

    """
    if debug:
        print(inspect.stack()[0][3])

    N_store = len(tlist)
    N_substeps = nsubsteps
    N = N_store * N_substeps
    dt = (tlist[1] - tlist[0]) / N_substeps

    data = Odedata()
    data.solver = "smesolve"
    data.times = tlist
    data.expect = np.zeros((len(e_ops), N_store), dtype=complex)

    # pre-compute collapse operator combinations that are commonly needed
    # when evaluating the RHS of stochastic master equations
    A_ops = []
    for c_idx, c in enumerate(sc_ops):

        # xxx: precompute useful operator expressions...
        cdc = c.dag() * c
        Ldt = spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc)
        LdW = spre(c) + spost(c.dag())
        Lm = spre(c) + spost(c.dag())  # currently same as LdW

        A_ops.append([Ldt.data, LdW.data, Lm.data])

    # Liouvillian for the deterministic part
    L = liouvillian_fast(H, c_ops)  # needs to be modified for TD systems

    progress_bar.start(ntraj)

    for n in range(ntraj):
        progress_bar.update(n)

        rho_t = mat2vec(rho0.full())

        states_list = _smesolve_single_trajectory(
            L, dt, tlist, N_store, N_substeps,
            rho_t, A_ops, e_ops, data, rhs, d1, d2, d2_len)

        # if average -> average...
        data.states.append(states_list)

    progress_bar.finished()

    # average
    data.expect = data.expect / ntraj

    return data


def _smesolve_single_trajectory(L, dt, tlist, N_store, N_substeps, rho_t,
                                A_ops, e_ops, data, rhs, d1, d2, d2_len):
    """
    Internal function. See smesolve.
    """

    dW = np.sqrt(dt) * scipy.randn(len(A_ops), N_store, N_substeps, d2_len)

    states_list = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                # XXX: need to keep hilbert space structure
                data.expect[e_idx, t_idx] += expect(e, Qobj(vec2mat(rho_t)))
        else:
            states_list.append(Qobj(rho_t))  # dito

        for j in range(N_substeps):

            drho_t = spmv(L.data.data,
                          L.data.indices,
                          L.data.indptr, rho_t) * dt

            for a_idx, A in enumerate(A_ops):
                drho_t += rhs(L.data, rho_t, A,
                              dt, dW[a_idx, t_idx, j, :], d1, d2)

            rho_t += drho_t

    return states_list


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
    data.solver = "spdpsolve"
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

        # if average -> average...
        data.states.append(states_list)

        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # average
    data.expect = data.expect / NT

    # standard error
    if NT > 1:
        data.ss = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))

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
                s = cy_expect(e.data.data, e.data.indices, e.data.indptr, 0, psi_t)
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

    data = Odedata()
    data.solver = "smepdpsolve"
    data.times = ssdata.tlist
    data.expect = np.zeros((len(ssdata.e_ops), N_store), dtype=complex)
    data.jump_times = []
    data.jump_op_idx = []

    # effective hamiltonian for deterministic part
    Heff = ssdata.H
    for c in ssdata.c_ops:
        Heff += -0.5j * c.dag() * c
        
    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)
        rho_t = ssdata.rho0.full()

        states_list, jump_times, jump_op_idx = \
            _smepdpsolve_single_trajectory(Heff, dt, ssdata.tlist,
                                           N_store, N_substeps,
                                           rho_t, ssdata.c_ops, ssdata.e_ops, 
                                           data)

        # if average -> average...
        data.states.append(states_list)

        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # average
    data.expect = data.expect / ssdata.ntraj

    return data


def _smepdpsolve_single_trajectory(Heff, dt, tlist, N_store, N_substeps, rho_t,
                                   c_ops, e_ops, data):
    """ 
    Internal function.
    """
    states_list = []

    raise NotImplemented("SME PDP solver not yet completed")

    phi_t = np.copy(psi_t)

    prng = RandomState() # todo: seed it
    r_jump, r_op = prng.rand(2)

    jump_times = []
    jump_op_idx = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                data.expect[e_idx, t_idx] += _rho_expect(e, rho_t)
        else:
            states_list.append(Qobj(rho_t))

        for j in range(N_substeps):

            if _rho_expect(d_op, sigma_t) < r_jump:
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

            # deterministic evolution with correction for norm decay

            # increment wavefunctions
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

    e1 = cy_expect(A[1].data, A[1].indices, A[1].indptr, 0, psi)
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

    e1 = cy_expect(A[1].data, A[1].indices, A[1].indptr, 0, psi)
    return [spmv(A[0].data, A[0].indices, A[0].indptr, psi) - 0.5 * e1 * psi]


def d1_psi_heterodyne(A, psi):
    """
    not working/tested
    Todo: cythonize
    """
    e1 = cy_expect(A[0].data, A[0].indices, A[0].indptr, 0, psi)

    B = A[0].T.conj()
    e2 = cy_expect(B.data, B.indices, B.indptr, 0, psi)

    return (e2 * spmv(A[0].data, A[0].indices, A[0].indptr, psi)
            - 0.5 * spmv(A[3].data, A[3].indices, A[3].indptr, psi)
            - 0.25 * e1 * e2 * psi)


def d2_psi_heterodyne(A, psi):
    """
    not working/tested
    Todo: cythonize
    """

    e1 = 1/np.sqrt(2.0) * cy_expect(A[1].data, A[1].indices, A[1].indptr, 0, psi)
    d2_re = spmv(A[0].data, A[0].indices, A[0].indptr, psi) - e1 * psi

    e1 = 1/np.sqrt(2.0) * cy_expect(A[2].data, A[2].indices, A[2].indptr, 0, psi)
    d2_im = spmv(1j * A[0].data, A[0].indices, A[0].indptr, psi) + e1 * psi

    return [d2_re, d2_im]


def d1_psi_photocurrent(A, psi):
    """
    Todo: cythonize.

    Note: requires poisson increments

    .. math::

        D_1(\psi, t) = - \\frac{1}{2}(C^\dagger C \psi - ||C\psi||^2 \psi)

    """

    n1 = norm(spmv(A[0].data, A[0].indices, A[0].indptr, psi))
    return (-0.5 * (spmv(A[3].data, A[3].indices, A[3].indptr, psi)
            - n1 ** 2 * psi))


def d2_psi_photocurrent(A, psi):
    """
    Todo: cythonize

    Note: requires poisson increments

    .. math::

        D_2(\psi, t) = C\psi / ||C\psi|| - \psi

    """
    psi_1 = spmv(A[0].data, A[0].indices, A[0].indptr, psi)
    n1 = norm(psi_1)
    return psi_1 / n1 - psi


#
# For SME
#

# def d(A, rho):
#
#     rho = wave function at the current time stemp
#
#     A[0] = Ldt (liouvillian contribution for a collapse operator)
#     A[1] = LdW (stochastic contribution)
#     A[3] = Lm
#


def _rho_expect(oper, state):
    prod = spmv(oper.data, oper.indices, oper.indptr, state)
    return sum(vec2mat(prod).diagonal())


def d1_rho_homodyne(A, rho):
    """
    not tested
    Todo: cythonize
    """
    return spmv(A[0].data, A[0].indices, A[0].indptr, rho)


def d2_rho_homodyne(A, rho):
    """
    not tested
    Todo: cythonize
    """

    e1 = _rho_expect(A[2], rho)
    return [spmv(A[1].data, A[1].indices, A[1].indptr, rho) - e1 * rho]


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
    return d1(A, psi_t) * dt + sum([d2_vec[n] * dW[n] for n in range(len(dW))])


def _rhs_rho_euler_maruyama(L, rho_t, A, dt, dW, d1, d2):
    """
    .. note::

        Experimental.

    """
    d2_vec = d2(A, rho_t)
    return d1(A, rho_t) * dt + sum([d2_vec[n] * dW[n] for n in range(len(dW))])


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
