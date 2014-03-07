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
#
#    Significant parts of this code were contributed by Denis Vasilyev.
#
###############################################################################

"""
This module contains experimental functions for solving stochastic schrodinger
and master equations. The API should not be considered stable, and is subject
to change when we work more on optimizing this module for performance and
features.

Release target: 2.3.0 or 3.0.0

Todo:

1) test and debug - always more to do here

2) store measurement records - done

3) add more sme solvers - done

4) cythonize some rhs or d1,d2 functions - done

5) parallelize

"""

import numpy as np
import scipy.sparse as sp
import scipy
from scipy.linalg.blas import get_blas_funcs
try:
    norm = get_blas_funcs("znrm2", dtype=np.float64)
except:
    from scipy.linalg import norm

from numpy.random import RandomState

from qutip.qobj import Qobj, isket
from qutip.states import ket2dm
from qutip.operators import commutator
from qutip.odedata import Odedata
from qutip.expect import expect, expect_rho_vec
from qutip.superoperator import (spre, spost, mat2vec, vec2mat,
                                 liouvillian_fast, lindblad_dissipator)
from qutip.cy.spmatfuncs import cy_expect_psi_csr, spmv, cy_expect_rho_vec
from qutip.cy.stochastic import (cy_d1_rho_photocurrent,
                                 cy_d2_rho_photocurrent)
from qutip.gui.progressbar import TextProgressBar
from qutip.odeoptions import Odeoptions
from qutip.settings import debug


if debug:
    import inspect


class _StochasticSolverData:
    """
    Internal class for passing data between stochastic solver functions.
    """
    def __init__(self, H=None, state0=None, tlist=None, c_ops=[], sc_ops=[],
                 e_ops=[], m_ops=None, args=None, ntraj=1, nsubsteps=1,
                 d1=None, d2=None, d2_len=1, dW_factors=None, rhs=None,
                 gen_A_ops=None, gen_noise=None, homogeneous=True, solver=None,
                 method=None, distribution='normal', store_measurement=False,
                 noise=None, normalize=True,
                 options=Odeoptions(), progress_bar=TextProgressBar()):

        self.H = H
        self.d1 = d1
        self.d2 = d2
        self.d2_len = d2_len
        self.dW_factors = dW_factors if dW_factors else np.ones(d2_len)
        self.state0 = state0
        self.tlist = tlist
        self.c_ops = c_ops
        self.sc_ops = sc_ops
        self.e_ops = e_ops

        if m_ops is None:
            self.m_ops = [[c for _ in range(d2_len)] for c in sc_ops]
        else:
            self.m_ops = m_ops

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
        self.store_states = options.store_states
        self.noise = noise
        self.args = args
        self.normalize = normalize

        self.gen_noise = gen_noise
        self.gen_A_ops = gen_A_ops


def ssesolve(H, psi0, tlist, sc_ops, e_ops, **kwargs):
    """
    Solve stochastic Schrodinger equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`. Must be uniform.

    sc_ops : list of :class:`qutip.qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the equation of motion.

    e_ops : list of :class:`qutip.qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See StochasticSolverData.

    Returns
    -------

    output: :class:`qutip.odedata`

        An instance of the class :class:`qutip.odedata`.
    """
    if debug:
        print(inspect.stack()[0][3])

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    ssdata = _StochasticSolverData(H=H, state0=psi0, tlist=tlist,
                                   sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    if ssdata.gen_A_ops is None:
        ssdata.gen_A_ops = _generate_psi_A_ops

    if (ssdata.d1 is None) or (ssdata.d2 is None):

        if ssdata.method == 'homodyne':
            ssdata.d1 = d1_psi_homodyne
            ssdata.d2 = d2_psi_homodyne
            ssdata.d2_len = 1
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'
            if not "dW_factors" in kwargs:
                ssdata.dW_factors = np.array([1])
            if not "m_ops" in kwargs:
                ssdata.m_ops = [[c + c.dag()] for c in ssdata.sc_ops]

        elif ssdata.method == 'heterodyne':
            ssdata.d1 = d1_psi_heterodyne
            ssdata.d2 = d2_psi_heterodyne
            ssdata.d2_len = 2
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'
            if not "dW_factors" in kwargs:
                ssdata.dW_factors = np.array([np.sqrt(2), np.sqrt(2)])
            if not "m_ops" in kwargs:
                ssdata.m_ops = [[(c + c.dag()), (-1j) * (c - c.dag())]
                                for idx, c in enumerate(ssdata.sc_ops)]

        elif ssdata.method == 'photocurrent':
            ssdata.d1 = d1_psi_photocurrent
            ssdata.d2 = d2_psi_photocurrent
            ssdata.d2_len = 1
            ssdata.homogeneous = False
            ssdata.distribution = 'poisson'

            if not "dW_factors" in kwargs:
                ssdata.dW_factors = np.array([1])
            if not "m_ops" in kwargs:
                ssdata.m_ops = [[None] for c in ssdata.sc_ops]
 
        else:
            raise Exception("Unrecognized method '%s'." % ssdata.method)

    if ssdata.distribution == 'poisson':
        ssdata.homogeneous = False

    if ssdata.solver == 'euler-maruyama' or ssdata.solver == None:
        ssdata.rhs_func = _rhs_psi_euler_maruyama

    elif ssdata.solver == 'platen':
        ssdata.rhs_func = _rhs_psi_platen

    else:
        raise Exception("Unrecognized solver '%s'." % ssdata.solver)

    res = ssesolve_generic(ssdata, ssdata.options, ssdata.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def smesolve(H, rho0, tlist, c_ops, sc_ops, e_ops, **kwargs):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    rho0 : :class:`qutip.qobj`
        initial density matrix of state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`. Must be uniform.

    c_ops : list of :class:`qutip.qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of :class:`qutip.qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the D1 and D2 functions
        are defined.

    e_ops : list of :class:`qutip.qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See StochasticSolverData.

    Returns
    -------

    output: :class:`qutip.odedata`

        An instance of the class :class:`qutip.odedata`.


    TODO: add check for commuting jump operators in Milstein.
    """

    if debug:
        print(inspect.stack()[0][3])

    if isket(rho0):
        rho0 = ket2dm(rho0)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    ssdata = _StochasticSolverData(H=H, state0=rho0, tlist=tlist, c_ops=c_ops,
                                   sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    if (ssdata.d1 is None) or (ssdata.d2 is None):

        if ssdata.method == 'homodyne' or ssdata.method is None:
            ssdata.d1 = d1_rho_homodyne
            ssdata.d2 = d2_rho_homodyne
            ssdata.d2_len = 1
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'
            if not "dW_factors" in kwargs:
                ssdata.dW_factors = np.array([np.sqrt(1)])
            if not "m_ops" in kwargs:
                ssdata.m_ops = [[c + c.dag()] for c in ssdata.sc_ops]
    
        elif ssdata.method == 'heterodyne':
            ssdata.d1 = d1_rho_heterodyne
            ssdata.d2 = d2_rho_heterodyne
            ssdata.d2_len = 2
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'
            if not "dW_factors" in kwargs:
                ssdata.dW_factors = np.array([np.sqrt(2), np.sqrt(2)])
            if not "m_ops" in kwargs:
                ssdata.m_ops = [[(c + c.dag()), -1j * (c - c.dag())]
                                for c in ssdata.sc_ops]

        elif ssdata.method == 'photocurrent':
            ssdata.d1 = cy_d1_rho_photocurrent
            ssdata.d2 = cy_d2_rho_photocurrent
            ssdata.d2_len = 1
            ssdata.homogeneous = False
            ssdata.distribution = 'poisson'

            if not "dW_factors" in kwargs:
                ssdata.dW_factors = np.array([1])
            if not "m_ops" in kwargs:
                ssdata.m_ops = [[None] for c in ssdata.sc_ops]
        else:
            raise Exception("Unrecognized method '%s'." % ssdata.method)

    if ssdata.distribution == 'poisson':
        ssdata.homogeneous = False

    if ssdata.gen_A_ops is None:
        ssdata.gen_A_ops = _generate_rho_A_ops

    if ssdata.rhs is None:
        if ssdata.solver == 'euler-maruyama' or ssdata.solver == None:
            ssdata.rhs = _rhs_rho_euler_maruyama

        elif ssdata.solver == 'milstein':
            if ssdata.method == 'homodyne' or ssdata.method is None:
                if len(sc_ops) == 1:
                    ssdata.rhs = _rhs_rho_milstein_homodyne_single
                else:
                    ssdata.rhs = _rhs_rho_milstein_homodyne

            elif ssdata.method == 'heterodyne':
                ssdata.rhs = _rhs_rho_milstein_homodyne
                ssdata.d2_len = 1
                ssdata.sc_ops = []
                for sc in iter(sc_ops):
                    ssdata.sc_ops += [sc / sqrt(2), -1.0j * sc / sqrt(2)]

        elif ssdata.solver == 'euler-maruyama_fast' and ssdata.method == 'homodyne':
            ssdata.rhs = _rhs_rho_euler_homodyne_fast
            ssdata.gen_A_ops = _generate_A_ops_Euler

        elif ssdata.solver == 'milstein_fast':
            ssdata.gen_A_ops = _generate_A_ops_Milstein
            ssdata.gen_noise = _generate_noise_Milstein
            if ssdata.method == 'homodyne' or ssdata.method is None:
                if len(sc_ops) == 1:
                    ssdata.rhs = _rhs_rho_milstein_homodyne_single_fast
                elif len(sc_ops) == 2:
                    ssdata.rhs = _rhs_rho_milstein_homodyne_two_fast
                else:
                    ssdata.rhs = _rhs_rho_milstein_homodyne_fast

            elif ssdata.method == 'heterodyne':
                ssdata.d2_len = 1
                ssdata.sc_ops = []
                for sc in iter(sc_ops):
                    ssdata.sc_ops += [sc / sqrt(2), -1.0j * sc / sqrt(2)]
                if len(sc_ops) == 1:
                    ssdata.rhs = _rhs_rho_milstein_homodyne_two_fast
                else:
                    ssdata.rhs = _rhs_rho_milstein_homodyne_fast

        else:
            raise Exception("Unrecognized solver '%s'." % ssdata.solver)

    res = smesolve_generic(ssdata, ssdata.options, ssdata.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def sepdpsolve(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=1, nsubsteps=10,
               options=Odeoptions(), progress_bar=TextProgressBar()):
    """
    A stochastic PDP solver for experimental/development and comparison to the
    stochastic DE solvers. Use mcsolve for real quantum trajectory
    simulations.
    """
    if debug:
        print(inspect.stack()[0][3])

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    ssdata = _StochasticSolverData()
    ssdata.H = H
    ssdata.psi0 = psi0
    ssdata.tlist = tlist
    ssdata.c_ops = c_ops
    ssdata.e_ops = e_ops
    ssdata.ntraj = ntraj
    ssdata.nsubsteps = nsubsteps

    res = sepdpsolve_generic(ssdata, options, progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    return res


def smepdpsolve(H, rho0, tlist, c_ops=[], e_ops=[], ntraj=1, nsubsteps=10,
                options=Odeoptions(), progress_bar=TextProgressBar()):
    """
    A stochastic PDP solver for density matrix evolution.
    """
    if debug:
        print(inspect.stack()[0][3])

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    ssdata = _StochasticSolverData()
    ssdata.H = H
    ssdata.rho0 = rho0
    ssdata.tlist = tlist
    ssdata.c_ops = c_ops
    ssdata.e_ops = e_ops
    ssdata.ntraj = ntraj
    ssdata.nsubsteps = nsubsteps

    res = smepdpsolve_generic(ssdata, options, progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    return res


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
    A_ops = ssdata.gen_A_ops(ssdata.sc_ops, ssdata.H)

    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)

        psi_t = ssdata.state0.full().ravel()

        noise = ssdata.noise[n] if ssdata.noise else None

        states_list, dW, m = _ssesolve_single_trajectory(data,
             ssdata.H, dt, ssdata.tlist, N_store, N_substeps, psi_t, A_ops,
             ssdata.e_ops, ssdata.m_ops, ssdata.rhs_func, ssdata.d1, ssdata.d2,
             ssdata.d2_len, ssdata.dW_factors, ssdata.homogeneous, ssdata.distribution, ssdata.args,
             store_measurement=ssdata.store_measurement, noise=noise,
             normalize=ssdata.normalize)

        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([ket2dm(data.states[m][n]) for m in range(NT)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / NT

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :]) if e.isherm else data.expect[n, :]
                   for n, e in enumerate(ssdata.e_ops)]

    return data


def _ssesolve_single_trajectory(data, H, dt, tlist, N_store, N_substeps, psi_t,
                                A_ops, e_ops, m_ops, rhs, d1, d2, d2_len,
                                dW_factors, homogeneous, distribution, args,
                                store_measurement=False, noise=None,
                                normalize=True):
    """
    Internal function. See ssesolve.
    """

    if noise is None:
        if homogeneous:
            if distribution == 'normal':
                dW = np.sqrt(dt) * \
                    scipy.randn(len(A_ops), N_store, N_substeps, d2_len)
            else:
                raise TypeError('Unsupported increment distribution for homogeneous process.')
        else:
            if distribution != 'poisson':
                raise TypeError('Unsupported increment distribution for inhomogeneous process.')

            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = noise

    states_list = []
    measurements = np.zeros((len(tlist), len(m_ops), d2_len), dtype=complex)

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_psi_csr(e.data.data,
                                      e.data.indices,
                                      e.data.indptr, psi_t, 0)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t))

        psi_prev = np.copy(psi_t)

        for j in range(N_substeps):

            if noise is None and not homogeneous:
                for a_idx, A in enumerate(A_ops):
                    #dw_expect = norm(spmv(A[0], psi_t)) ** 2 * dt
                    dw_expect = cy_expect_psi_csr(A[3].data,
                                                  A[3].indices,
                                                  A[3].indptr, psi_t, 1) * dt
                    dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect, d2_len)

            psi_t = rhs(H.data, psi_t, t + dt * j,
                        A_ops, dt, dW[:, t_idx, j, :], d1, d2, args)

            # optionally renormalize the wave function
            if normalize:
                psi_t /= norm(psi_t)

        if store_measurement:
            for m_idx, m in enumerate(m_ops):
                for dW_idx, dW_factor in enumerate(dW_factors):
                    if m[dW_idx]:
                        m_data = m[dW_idx].data
                        m_expt = cy_expect_psi_csr(m_data.data,
                                                   m_data.indices,
                                                   m_data.indptr,
                                                   psi_t, 0)
                    else:
                        m_expt = 0
                    measurements[t_idx, m_idx, dW_idx] = (m_expt +
                       dW_factor * dW[m_idx, t_idx, :, dW_idx].sum() /
                       (dt * N_substeps))

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(2))

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

    # Liouvillian for the deterministic part.
    # needs to be modified for TD systems
    L = liouvillian_fast(ssdata.H, ssdata.c_ops)

    # pre-compute suporoperator operator combinations that are commonly needed
    # when evaluating the RHS of stochastic master equations
    A_ops = ssdata.gen_A_ops(ssdata.sc_ops, L.data, dt)

    # use .data instead of Qobj ?
    s_e_ops = [spre(e) for e in ssdata.e_ops]

    if ssdata.m_ops:
        s_m_ops = [[spre(m) if m else None for m in m_op]
                   for m_op in ssdata.m_ops]
    else:
        s_m_ops = [[spre(c) for _ in range(ssdata.d2_len)]
                   for c in ssdata.sc_ops]

    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)

        rho_t = mat2vec(ssdata.state0.full()).ravel()

        # noise = ssdata.noise[n] if ssdata.noise else None
        if ssdata.noise:
            noise = ssdata.noise[n]
        elif ssdata.gen_noise:
            noise = ssdata.gen_noise(
                len(A_ops), N_store, N_substeps, ssdata.d2_len, dt)
        else:
            noise = None

        states_list, dW, m = _smesolve_single_trajectory(data,
                                 L, dt, ssdata.tlist, N_store, N_substeps,
                                 rho_t, A_ops, s_e_ops, s_m_ops, ssdata.rhs,
                                 ssdata.d1, ssdata.d2, ssdata.d2_len,
                                 ssdata.dW_factors, ssdata.homogeneous,
                                 ssdata.distribution, ssdata.args,
                                 store_measurement=ssdata.store_measurement,
                                 store_states=ssdata.store_states, noise=noise)

        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([data.states[m][n] for m in range(NT)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / NT

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :]) if e.isherm else data.expect[n, :]
                   for n, e in enumerate(ssdata.e_ops)]

    return data


def _smesolve_single_trajectory(data, L, dt, tlist, N_store, N_substeps, rho_t,
                                A_ops, e_ops, m_ops, rhs, d1, d2, d2_len, dW_factors,
                                homogeneous, distribution, args,
                                store_measurement=False,
                                store_states=False, noise=None):
    """
    Internal function. See smesolve.
    """

    if noise is None:
        if homogeneous:
            if distribution == 'normal':
                dW = np.sqrt(
                    dt) * scipy.randn(len(A_ops), N_store, N_substeps, d2_len)
            else:
                raise TypeError('Unsupported increment distribution for homogeneous process.')
        else:
            if distribution != 'poisson':
                raise TypeError('Unsupported increment distribution for inhomogeneous process.')

            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = noise

    states_list = []
    measurements = np.zeros((len(tlist), len(m_ops), d2_len), dtype=complex)

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_rho_vec(e.data, rho_t, 0)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2

        if store_states or not e_ops:
            # XXX: need to keep hilbert space structure
            states_list.append(Qobj(vec2mat(rho_t)))

        rho_prev = np.copy(rho_t)

        for j in range(N_substeps):

            if noise is None and not homogeneous:
                for a_idx, A in enumerate(A_ops):
                    dw_expect = cy_expect_rho_vec(A[4], rho_t, 1) * dt
                    if dw_expect > 0:
                        dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect, d2_len)
                    else:
                        dW[a_idx, t_idx, j, :] = np.zeros(d2_len)

            rho_t = rhs(L.data, rho_t, t + dt * j,
                        A_ops, dt, dW[:, t_idx, j, :], d1, d2, args)

        if store_measurement:
            for m_idx, m in enumerate(m_ops):
                for dW_idx, dW_factor in enumerate(dW_factors):
                    if m[dW_idx]:
                        m_expt = cy_expect_rho_vec(m[dW_idx].data, rho_prev, 0)
                    else:
                        m_expt = 0
                    measurements[t_idx, m_idx, dW_idx] = m_expt + dW_factor * \
                        dW[m_idx, t_idx, :, dW_idx].sum() / (dt * N_substeps)

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(2))

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
        psi_t = ssdata.psi0.full().ravel()

        states_list, jump_times, jump_op_idx = \
            _sepdpsolve_single_trajectory(data, Heff, dt, ssdata.tlist,
                                          N_store, N_substeps,
                                          psi_t, ssdata.c_ops, ssdata.e_ops)

        data.states.append(states_list)
        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([data.states[m][n] for m in range(NT)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / NT

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :]) if e.isherm else data.expect[n, :]
                   for n, e in enumerate(ssdata.e_ops)]

    return data


def _sepdpsolve_single_trajectory(data, Heff, dt, tlist, N_store, N_substeps,
                                  psi_t, c_ops, e_ops):
    """
    Internal function.
    """
    states_list = []

    phi_t = np.copy(psi_t)

    prng = RandomState()  # todo: seed it
    r_jump, r_op = prng.rand(2)

    jump_times = []
    jump_op_idx = []

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_psi_csr(
                    e.data.data, e.data.indices, e.data.indptr, psi_t, 0)
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
        rho_t = mat2vec(ssdata.rho0.full()).ravel()

        states_list, jump_times, jump_op_idx = \
            _smepdpsolve_single_trajectory(data, L, dt, ssdata.tlist,
                                           N_store, N_substeps,
                                           rho_t, ssdata.c_ops, ssdata.e_ops)

        data.states.append(states_list)
        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([data.states[m][n] for m in range(NT)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / ssdata.ntraj

    # standard error
    if NT > 1:
        data.se = (data.ss - NT * (data.expect ** 2)) / (NT * (NT - 1))
    else:
        data.se = None

    return data


def _smepdpsolve_single_trajectory(data, L, dt, tlist, N_store, N_substeps,
                                   rho_t, c_ops, e_ops):
    """
    Internal function.
    """
    states_list = []

    rho_t = np.copy(rho_t)

    prng = RandomState()  # todo: seed it
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
            dsigma_t = spmv(L.data, sigma_t) * dt

            # deterministic evolution with correction for norm decay
            drho_t = spmv(L.data, rho_t) * dt

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


def _generate_psi_A_ops(sc_ops, H):
    """
    pre-compute superoperator operator combinations that are commonly needed
    when evaluating the RHS of stochastic schrodinger equations
    """

    A_ops = []
    for c_idx, c in enumerate(sc_ops):
        A_ops.append([c.data,
                      (c + c.dag()).data,
                      (c - c.dag()).data,
                      (c.dag() * c).data])
 
    return A_ops


def d1_psi_homodyne(A, psi):
    """
    OK
    Todo: cythonize

    .. math::

        D_1(C, \psi) = \\frac{1}{2}(\\langle C + C^\\dagger\\rangle\\C psi -
        C^\\dagger C\\psi - \\frac{1}{4}\\langle C + C^\\dagger\\rangle^2\\psi)

    """

    e1 = cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    return 0.5 * (e1 * spmv(A[0], psi) -
                  spmv(A[3], psi) -
                  0.25 * e1 ** 2 * psi)


def d2_psi_homodyne(A, psi):
    """
    OK
    Todo: cythonize

    .. math::

        D_2(\psi, t) = (C - \\frac{1}{2}\\langle C + C^\\dagger\\rangle)\\psi

    """

    e1 = cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    return [spmv(A[0], psi) - 0.5 * e1 * psi]


def d1_psi_heterodyne(A, psi):
    """
    Todo: cythonize

    .. math::

        D_1(\psi, t) = -\\frac{1}{2}(C^\\dagger C - \\langle C^\\dagger \\rangle C +
                        \\frac{1}{2}\\langle C \\rangle\\langle C^\\dagger \\rangle))\psi

    """
    e_C = cy_expect_psi_csr(A[0].data, A[0].indices, A[0].indptr, psi, 0)
    B = A[0].T.conj()
    e_Cd = cy_expect_psi_csr(B.data, B.indices, B.indptr, psi, 0)

    return (-0.5 * spmv(A[3], psi) +
            0.5 * e_Cd * spmv(A[0], psi) -
            0.25 * e_C * e_Cd * psi)


def d2_psi_heterodyne(A, psi):
    """
    Todo: cythonize

        X = \\frac{1}{2}(C + C^\\dagger)

        Y = \\frac{1}{2}(C - C^\\dagger)

        D_{2,1}(\psi, t) = \\sqrt(1/2) (C - \\langle X \\rangle) \\psi

        D_{2,2}(\psi, t) = -i\\sqrt(1/2) (C - \\langle Y \\rangle) \\psi

    """

    X = 0.5 * cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    Y = 0.5 * cy_expect_psi_csr(A[2].data, A[2].indices, A[2].indptr, psi, 0)

    d2_1 = np.sqrt(0.5) * (spmv(A[0], psi) - X * psi)
    d2_2 = -1.0j * np.sqrt(0.5) * (spmv(A[0], psi) - Y * psi)

    return [d2_1, d2_2]


def d1_psi_photocurrent(A, psi):
    """
    Todo: cythonize.

    Note: requires poisson increments

    .. math::

        D_1(\psi, t) = - \\frac{1}{2}(C^\dagger C \psi - ||C\psi||^2 \psi)

    """
    return (-0.5 * (spmv(A[3], psi)
            - norm(spmv(A[0], psi)) ** 2 * psi))


def d2_psi_photocurrent(A, psi):
    """
    Todo: cythonize

    Note: requires poisson increments

    .. math::

        D_2(\psi, t) = C\psi / ||C\psi|| - \psi

    """
    psi_1 = spmv(A[0], psi)
    n1 = norm(psi_1)
    if n1 != 0:
        return [psi_1 / n1 - psi]
    else:
        return [- psi]


#
# For SME
#

# def d(A, rho_vec):
#
#     rho = density operator in vector form at the current time stemp
#
#     A[0] = spre(a) = A_L
#     A[1] = spost(a) = A_R
#     A[2] = spre(a.dag()) = Ad_L
#     A[3] = spost(a.dag()) = Ad_R
#     A[4] = spre(a.dag() * a) = (Ad A)_L
#     A[5] = spost(a.dag() * a) = (Ad A)_R
#     A[6] = (spre(a) * spost(a.dag()) = A_L * Ad_R
#     A[7] = lindblad_dissipator(a)


def _generate_rho_A_ops(sc, L, dt):
    """
    pre-compute superoperator operator combinations that are commonly needed
    when evaluating the RHS of stochastic master equations
    """
    out = []
    for c_idx, c in enumerate(sc):
        n = c.dag() * c
        out.append([spre(c).data, spost(c).data,
                    spre(c.dag()).data, spost(c.dag()).data,
                    spre(n).data, spost(n).data, (spre(c) * spost(c.dag())).data,
                    lindblad_dissipator(c, data_only=True)])

    return out


def _generate_A_ops_Euler(sc, L, dt):
    """
    combine precomputed operators in one long operator for the Euler method
    """
    A_len = len(sc)
    out = []
    out += [spre(c).data + spost(c.dag()).data for c in sc]
    out += [(L + np.sum(
        [lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) * dt]
    out1 = [[sp.vstack(out).tocsr(), sc[0].shape[0]]]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]
    return out1


def _generate_A_ops_Milstein(sc, L, dt):
    """
    combine precomputed operators in one long operator for the Milstein method
    with commuting stochastic jump operators.
    """
    A_len = len(sc)
    temp = [spre(c).data + spost(c.dag()).data for c in sc]
    out = []
    out += temp
    out += [temp[n] * temp[n] for n in range(A_len)]
    out += [temp[n] * temp[m] for (n, m) in np.ndindex(A_len, A_len) if n > m]
    out += [(L + np.sum(
        [lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) * dt]
    out1 = [[sp.vstack(out).tocsr(), sc[0].shape[0]]]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]
    return out1


def _generate_noise_Milstein(sc_len, N_store, N_substeps, d2_len, dt):
    """
    generate noise terms for the fast Milstein scheme
    """
    dW_temp = np.sqrt(dt) * scipy.randn(sc_len, N_store, N_substeps, 1)
    if sc_len == 1:
        noise = np.vstack([dW_temp, 0.5 * (dW_temp * dW_temp - dt * np.ones((sc_len, N_store, N_substeps, 1)))])
    else:
        noise = np.vstack([dW_temp, 0.5 * (dW_temp * dW_temp - dt * np.ones((sc_len, N_store, N_substeps, 1)))] +
                          [[dW_temp[n] * dW_temp[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m]])
    return noise


def sop_H(A, rho_vec):
    """
    Evaluate the superoperator

    H[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger] rho
            -> (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec] rho_vec

    Todo: cythonize, add A_L + Ad_R to precomputed operators
    """
    M = A[0] + A[3]

    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    return spmv(M, rho_vec) - e1 * rho_vec


def sop_G(A, rho_vec):
    """
    Evaluate the superoperator

    G[a] rho = a rho a^\dagger / Tr[a rho a^\dagger] - rho
            -> A_L Ad_R rho_vec / Tr[A_L Ad_R rho_vec] - rho_vec

    Todo: cythonize, add A_L + Ad_R to precomputed operators
    """

    e1 = cy_expect_rho_vec(A[6], rho_vec, 0)

    if e1 > 1e-15:
        return spmv(A[6], rho_vec) / e1 - rho_vec
    else:
        return -rho_vec


def d1_rho_homodyne(A, rho_vec):
    """

    D1[a] rho = lindblad_dissipator(a) * rho

    Todo: cythonize
    """
    return spmv(A[7], rho_vec)


def d2_rho_homodyne(A, rho_vec):
    """

    D2[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger]
              = (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec]

    Todo: cythonize, add A_L + Ad_R to precomputed operators
    """
    M = A[0] + A[3]

    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    return [spmv(M, rho_vec) - e1 * rho_vec]


def d1_rho_heterodyne(A, rho_vec):
    """
    todo: cythonize, docstrings
    """
    return spmv(A[7], rho_vec)


def d2_rho_heterodyne(A, rho_vec):
    """
    todo: cythonize, docstrings
    """
    M = A[0] + A[3]
    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    d1 = spmv(M, rho_vec) - e1 * rho_vec
    M = A[0] - A[3]
    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    d2 = spmv(M, rho_vec) - e1 * rho_vec
    return [1.0 / np.sqrt(2) * d1, -1.0j / np.sqrt(2) * d2]


def d1_rho_photocurrent(A, rho_vec):
    """
    Todo: cythonize, add (AdA)_L + AdA_R to precomputed operators
    """
    n_sum = A[4] + A[5]
    e1 = cy_expect_rho_vec(n_sum, rho_vec, 0)
    return 0.5 * (e1 * rho_vec - spmv(n_sum, rho_vec))


def d2_rho_photocurrent(A, rho_vec):
    """
    Todo: cythonize, add (AdA)_L + AdA_R to precomputed operators
    """
    e1 = cy_expect_rho_vec(A[6], rho_vec, 0)
    return [spmv(A[6], rho_vec) / e1 - rho_vec] if e1.real > 1e-15 else [-rho_vec]


#------------------------------------------------------------------------------
# Deterministic part of the rho/psi update functions. TODO: Make these
# compatible with qutip's time-dependent hamiltonian and collapse operators
#
def _rhs_psi_deterministic(H, psi_t, t, dt, args):
    """
    Deterministic contribution to the density matrix change
    """
    dpsi_t = (-1.0j * dt) * (H * psi_t)

    return dpsi_t


def _rhs_rho_deterministic(L, rho_t, t, dt, args):
    """
    Deterministic contribution to the density matrix change
    """
    drho_t = spmv(L, rho_t) * dt

    return drho_t


#------------------------------------------------------------------------------
# Euler-Maruyama rhs functions for the stochastic Schrodinger and master
# equations
#

def _rhs_psi_euler_maruyama(H, psi_t, t, A_ops, dt, dW, d1, d2, args):
    """
    .. note::

        Experimental.

    """
    dW_len = len(dW[0, :])
    dpsi_t = _rhs_psi_deterministic(H, psi_t, t, dt, args)

    for a_idx, A in enumerate(A_ops):
        d2_vec = d2(A, psi_t)
        dpsi_t += d1(A, psi_t) * dt + np.sum([d2_vec[n] * dW[a_idx, n]
                                              for n in range(dW_len)
                                              if dW[a_idx, n] != 0], axis=0)

    return psi_t + dpsi_t


def _rhs_rho_euler_maruyama(L, rho_t, t, A_ops, dt, dW, d1, d2, args):
    """
    .. note::

        Experimental.

    """
    dW_len = len(dW[0, :])

    drho_t = _rhs_rho_deterministic(L, rho_t, t, dt, args)

    for a_idx, A in enumerate(A_ops):
        d2_vec = d2(A, rho_t)
        drho_t += d1(A, rho_t) * dt
        drho_t += np.sum([d2_vec[n] * dW[a_idx, n] 
                          for n in range(dW_len) if dW[a_idx, n] != 0], axis=0)

    return rho_t + drho_t


def _rhs_rho_euler_homodyne_fast(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    fast Euler-Maruyama for homodyne detection
    """

    dW = ddW[:, 0]

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))

    drho_t = d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    drho_t += (1.0 - np.inner(e, dW)) * rho_t
    return drho_t


#------------------------------------------------------------------------------
# Platen method
#
def _rhs_psi_platen(H, psi_t, t, A_ops, dt, dW, d1, d2, args):
    """
    TODO: support multiple stochastic increments

    .. note::

        Experimental.

    """

    sqrt_dt = np.sqrt(dt)

    dW_len = len(dW[0, :])
    dpsi_t = _rhs_psi_deterministic(H, psi_t, t, dt, args)

    for a_idx, A in enumerate(A_ops):
        # XXX: This needs to be revised now that
        # dpsi_t is the change for all stochastic collapse operators

        # TODO: needs to be updated to support mutiple Weiner increments
        dpsi_t_H = (-1.0j * dt) * spmv(H, psi_t)

        psi_t_1 = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t)[0] * dW[a_idx, 0]
        psi_t_p = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t)[0] * sqrt_dt
        psi_t_m = psi_t + dpsi_t_H + d1(A, psi_t) * dt - d2(A, psi_t)[0] * sqrt_dt

        dpsi_t += 0.50 * (d1(A, psi_t_1) + d1(A, psi_t)) * dt + \
            0.25 * (d2(A, psi_t_p)[0] + d2(A, psi_t_m)[0] + 2 * d2(A, psi_t)[0]) * dW[a_idx, 0] + \
            0.25 * (d2(A, psi_t_p)[0] - d2(A, psi_t_m)[0]) * (
                dW[a_idx, 0] ** 2 - dt) * sqrt_dt

    return dpsi_t


#------------------------------------------------------------------------------
# Milstein rhs functions for the stochastic master equation
#
#
def _rhs_rho_milstein_homodyne_single(L, rho_t, t, A_ops, dt, dW, d1, d2, args):
    """
    .. note::

        Experimental.
        Milstein scheme for homodyne detection with single jump operator.

    """

    A = A_ops[0]
    M = A[0] + A[3]
    e1 = cy_expect_rho_vec(M, rho_t, 0)

    d2_vec = spmv(M, rho_t)
    d2_vec2 = spmv(M, d2_vec)
    e2 = cy_expect_rho_vec(M, d2_vec, 0)

    drho_t = _rhs_rho_deterministic(L, rho_t, t, dt, args)
    drho_t += spmv(A[7], rho_t) * dt
    drho_t += (d2_vec - e1 * rho_t) * dW[0, 0]
    drho_t += 0.5 * (d2_vec2 - 2 * e1 * d2_vec + (-e2 + 2 * e1 * e1) * rho_t) * (dW[0, 0] * dW[0, 0] - dt)
    return rho_t + drho_t


def _rhs_rho_milstein_homodyne(L, rho_t, t, A_ops, dt, dW, d1, d2, args):
    """
    .. note::

        Experimental.
        Milstein scheme for homodyne detection.
        This implementation works for commuting stochastic jump operators.
        TODO: optimizations: do calculation for n>m only

    """
    A_len = len(A_ops)

    M = np.array([A_ops[n][0] + A_ops[n][3] for n in range(A_len)])
    e1 = np.array([cy_expect_rho_vec(M[n], rho_t, 0) for n in range(A_len)])

    d1_vec = np.sum([spmv(A_ops[n][7], rho_t)
                     for n in range(A_len)], axis=0)

    d2_vec = np.array([spmv(M[n], rho_t)
                       for n in range(A_len)])

    # This calculation is suboptimal. We need only values for m>n in case of
    # commuting jump operators.
    d2_vec2 = np.array([[spmv(M[n], d2_vec[m])
                         for m in range(A_len)] for n in range(A_len)])
    e2 = np.array([[cy_expect_rho_vec(M[n], d2_vec[m], 0)
                    for m in range(A_len)] for n in range(A_len)])

    drho_t = _rhs_rho_deterministic(L, rho_t, t, dt, args)
    drho_t += d1_vec * dt
    drho_t += np.sum([(d2_vec[n] - e1[n] * rho_t) * dW[n, 0]
                      for n in range(A_len)], axis=0)
    drho_t += 0.5 * np.sum([(d2_vec2[n, n] - 2.0 * e1[n] * d2_vec[n] +
                            (-e2[n, n] + 2.0 * e1[n] * e1[n]) * rho_t) * (dW[n, 0] * dW[n, 0] - dt)
                            for n in range(A_len)], axis=0)

    # This calculation is suboptimal. We need only values for m>n in case of
    # commuting jump operators.
    drho_t += 0.5 * np.sum([(d2_vec2[n, m] - e1[m] * d2_vec[n] - e1[n] * d2_vec[m] +
                          (-e2[n, m] + 2.0 * e1[n] * e1[m]) * rho_t) * (dW[n, 0] * dW[m, 0])
                            for (n, m) in np.ndindex(A_len, A_len) if n != m], axis=0)

    return rho_t + drho_t


def _rhs_rho_milstein_homodyne_single_fast(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    fast Milstein for homodyne detection with 1 stochastic operator
    """
    dW = ddW[:, 0]

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))

    e[1] -= 2.0 * e[0] * e[0]

    drho_t = (1.0 - np.inner(e, dW)) * rho_t
    dW[0] -= 2.0 * e[0] * dW[1]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return drho_t


def _rhs_rho_milstein_homodyne_two_fast(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    fast Milstein for homodyne detection with 2 stochastic operators
    """
    dW = ddW[:, 0]

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))
    d_vec[-2] -= np.dot(e[:2][::-1], d_vec[:2])

    e[2:4] -= 2.0 * e[:2] * e[:2]
    e[4] -= 2.0 * e[1] * e[0]

    drho_t = (1.0 - np.inner(e, dW)) * rho_t
    dW[:2] -= 2.0 * e[:2] * dW[2:4]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return drho_t


def _rhs_rho_milstein_homodyne_fast(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    fast Milstein for homodyne detection with >2 stochastic operators
    """
    dW = ddW[:, 0]
    sc_len = len(A)
    sc2_len = 2 * sc_len

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = np.real(d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))
    d_vec[sc2_len:-1] -= np.array([e[m] * d_vec[n] + e[n] * d_vec[m] 
                                   for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    e[sc_len:sc2_len] -= 2.0 * e[:sc_len] * e[:sc_len]
    e[sc2_len:] -= 2.0 * np.array([e[n] * e[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    drho_t = (1.0 - np.inner(e, dW)) * rho_t
    dW[:sc_len] -= 2.0 * e[:sc_len] * dW[sc_len:sc2_len]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return drho_t
