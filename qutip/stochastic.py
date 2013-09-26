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
# Copyright (C) 2012-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

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
import scipy
from scipy.linalg import norm
from numpy.random import RandomState

from qutip.odedata import Odedata
from qutip.odeoptions import Odeoptions
from qutip.expect import expect, expect_rho_vec
from qutip.qobj import Qobj, isket
from qutip.superoperator import (spre, spost, mat2vec, vec2mat,
                                 liouvillian_fast, lindblad_dissipator)
from qutip.states import ket2dm
from qutip.cyQ.spmatfuncs import cy_expect, spmv, cy_expect_rho_vec
from qutip.gui.progressbar import TextProgressBar

from qutip.settings import debug
from qutip.operators import commutator

if debug:
    import inspect

class _StochasticSolverData:
    """
    Internal class for passing data between stochastic solver functions.
    """
    def __init__(self, H=None, state0=None, tlist=None, c_ops=[], sc_ops=[],
                 e_ops=[], m_ops=None, args=None, ntraj=1, nsubsteps=1,
                 d1=None, d2=None, d2_len=1, dW_factors=None, rhs=None, homogeneous=True,
                 solver=None, method=None, distribution='normal',
                 store_measurement=False, noise=None, normalize=True,
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
            self.m_ops = [[c for _ in range(d2_len)] for c in c_ops]
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

    if (ssdata.d1 is None) or (ssdata.d2 is None):

        if ssdata.method == 'homodyne':
            ssdata.d1 = d1_psi_homodyne
            ssdata.d2 = d2_psi_homodyne
            ssdata.d2_len = 1
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'
            if not hasattr(kwargs, "m_ops"):
                ssdata.m_ops = [[c + c.dag()] for c in ssdata.sc_ops]

        elif ssdata.method == 'heterodyne':
            ssdata.d1 = d1_psi_heterodyne
            ssdata.d2 = d2_psi_heterodyne
            ssdata.d2_len = 2
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'
            if not hasattr(kwargs, "dW_factors"):
                ssdata.dW_factors = np.array([np.sqrt(2), np.sqrt(2)])
            if not hasattr(kwargs, "m_ops"):
                # XXX: ugly hack to get around that we need the quadrature
                # operators and not the collapse operator (factor sqrt(gamma)
                # difference).
                g_vec = [0.5 * (commutator((c + c.dag()), -1j * (c - c.dag()))[0,0]).imag
                         for c in ssdata.sc_ops]
                ssdata.m_ops = [[(c + c.dag()) / np.sqrt(g_vec[idx]),
                                 (-1j) * (c - c.dag()) / np.sqrt(g_vec[idx])]
                                 for idx, c in enumerate(ssdata.sc_ops)]
                

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
            if not hasattr(kwargs, "m_ops"):
                ssdata.m_ops = [[c + c.dag()] for c in ssdata.sc_ops]
            if not hasattr(kwargs, "dW_factors"):
                ssdata.dW_factors = np.array([np.sqrt(2)])

        elif ssdata.method == 'heterodyne':
            ssdata.d1 = d1_rho_heterodyne
            ssdata.d2 = d2_rho_heterodyne
            ssdata.d2_len = 2
            ssdata.homogeneous = True
            ssdata.distribution = 'normal'
            if not hasattr(kwargs, "dW_factors"):
                ssdata.dW_factors = np.array([np.sqrt(2), np.sqrt(2)])
            if not hasattr(kwargs, "m_ops"):
                # XXX: ugly hack to get around that we need the quadrature
                # operators without the sqrt(gamma) factor from the collapse
                # operators
                g_vec = [0.5 * (commutator((c + c.dag()), -1j * (c - c.dag()))[0,0]).imag
                         for c in ssdata.sc_ops]
                ssdata.m_ops = [[(c + c.dag()) / np.sqrt(g_vec[idx]),
                                 (-1j) * (c - c.dag()) / np.sqrt(g_vec[idx])]
                                 for idx, c in enumerate(ssdata.sc_ops)]


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
        			ssdata.sc_ops.append(sc/sqrt(2))
        			ssdata.sc_ops.append(-1.0j*sc/sqrt(2))

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
    A_ops = []
    for c_idx, c in enumerate(ssdata.sc_ops):
        A_ops.append([c.data,
                      (c + c.dag()).data,
                      (c - c.dag()).data,
                      (c.dag() * c).data])

    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)

        psi_t = ssdata.state0.full().ravel()

        noise = ssdata.noise[n] if ssdata.noise else None

        states_list, dW, m = _ssesolve_single_trajectory(data,
            ssdata.H, dt, ssdata.tlist, N_store, N_substeps, psi_t, A_ops,
            ssdata.e_ops, ssdata.m_ops, ssdata.rhs_func, ssdata.d1, ssdata.d2,
            ssdata.d2_len, ssdata.homogeneous, ssdata.distribution, ssdata.args,
            store_measurement=ssdata.store_measurement, noise=noise,
            normalize=ssdata.normalize)

        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum(state_list).unit() for state_list in data.states]

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


def _ssesolve_single_trajectory(data, H, dt, tlist, N_store, N_substeps, psi_t,
                                A_ops, e_ops, m_ops, rhs, d1, d2, d2_len,
                                homogeneous, distribution, args,
                                store_measurement=False, noise=None,
                                normalize=True):
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
    measurements = np.zeros((len(tlist), len(m_ops), d2_len), dtype=complex)

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

            if noise is None and not homogeneous:
                for a_idx, A in enumerate(A_ops):
                    dw_expect = norm(spmv(A[0].data, A[0].indices, A[0].indptr, psi_t)) ** 2 * dt
                    dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect, d2_len)

            psi_t = rhs(H.data, psi_t, t + dt * j,
                        A_ops, dt, dW[:, t_idx, j, :], d1, d2, args)

            # optionally renormalize the wave function
            if normalize:
                psi_t /= norm(psi_t)

        if store_measurement:
            for m_idx, m in enumerate(m_ops):
                for dW_idx, dW_factor in enumerate(dW_factors):
                    phi = spmv(m[dW_idx].data.data, m[dW_idx].data.indices, m[dW_idx].data.indptr, psi_prev)
                    measurements[t_idx, m_idx, dW_idx] = (norm(phi) ** 2 +
                                                  dW[m_idx, t_idx, :, 0].sum() / (dt * N_substeps))

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

    # use .data instead of Qobj ?
    s_e_ops = [spre(e) for e in ssdata.e_ops]

    if ssdata.m_ops:
        s_m_ops = [[spre(m) for m in m_op] for m_op in ssdata.m_ops]
    else:
        s_m_ops = [[spre(c) for _ in range(ssdata.d2_len)] for c in ssdata.sc_ops]

    
    # Liouvillian for the deterministic part.
    # needs to be modified for TD systems
    L = liouvillian_fast(ssdata.H, ssdata.c_ops)

    progress_bar.start(ssdata.ntraj)

    for n in range(ssdata.ntraj):
        progress_bar.update(n)

        rho_t = mat2vec(ssdata.state0.full()).ravel()

        noise = ssdata.noise[n] if ssdata.noise else None

        states_list, dW, m = _smesolve_single_trajectory(data,
            L, dt, ssdata.tlist, N_store, N_substeps,
            rho_t, A_ops, s_e_ops, s_m_ops, ssdata.rhs,
            ssdata.d1, ssdata.d2, ssdata.d2_len, ssdata.dW_factors, ssdata.homogeneous,
            ssdata.distribution, ssdata.args,
            store_measurement=ssdata.store_measurement,
            store_states=ssdata.store_states, noise=noise)

        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum(state_list).unit() for state_list in data.states]

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
    measurements = np.zeros((len(tlist), len(m_ops), d2_len), dtype=complex)

    for t_idx, t in enumerate(tlist):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_rho_vec(e.data, rho_t)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2 
        
        if store_states or not e_ops:
            # XXX: need to keep hilbert space structure
            states_list.append(Qobj(vec2mat(rho_t)))

        rho_prev = np.copy(rho_t)

        for j in range(N_substeps):

            if noise is None and not homogeneous:
                for a_idx, A in enumerate(A_ops):
                    dw_expect = np.real(cy_expect_rho_vec(A[4], rho_t)) * dt
                    dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect, d2_len)

            rho_t = rhs(L.data, rho_t, t + dt * j,
                        A_ops, dt, dW[:, t_idx, j, :], d1, d2, args)

        if store_measurement:
            for m_idx, m in enumerate(m_ops):
                for dW_idx, dW_factor in enumerate(dW_factors):
                    m_expt = cy_expect_rho_vec(m[dW_idx].data, rho_prev)
                    measurements[t_idx, m_idx, dW_idx] = m_expt + dW_factor * dW[m_idx, t_idx, :, dW_idx].sum() / (dt * N_substeps)

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
        data.states = [sum(state_list).unit() for state_list in data.states]

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


def _sepdpsolve_single_trajectory(data, Heff, dt, tlist, N_store, N_substeps,
                                  psi_t, c_ops, e_ops):
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
        data.states = [sum(state_list).unit() for state_list in data.states]
    
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
#     A[0] = spre(a) = A_L
#     A[1] = spost(a) = A_R
#     A[2] = spre(a.dag()) = Ad_L
#     A[3] = spost(a.dag()) = Ad_R
#     A[4] = spre(a.dag() * a) = (Ad A)_L
#     A[5] = spost(a.dag() * a) = (Ad A)_R
#     A[6] = (spre(a) * spost(a.dag()) = A_L * Ad_R
#     A[7] = lindblad_dissipator(a) 


def sop_H(A, rho_vec):
    """
    Evaluate the superoperator

    H[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger]
            -> (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec]

    Todo: cythonize, add A_L + Ad_R to precomputed operators
    """
    M = A[0] + A[3]

    e1 = cy_expect_rho_vec(M, rho_vec)
    return spmv(M.data, M.indices, M.indptr, rho_vec) - e1 * rho_vec


def sop_G(A, rho_vec):
    """
    Evaluate the superoperator

    G[a] rho = a rho a^\dagger / Tr[a rho a^\dagger] - rho
            -> A_L Ad_R rho_vec / Tr[A_L Ad_R rho_vec] - rho_vec

    Todo: cythonize, add A_L + Ad_R to precomputed operators
    """

    e1 = cy_expect_rho_vec(A[6], rho_vec)

    if e1 > 1e-15:
        return spmv(A[6].data, A[6].indices, A[6].indptr, rho_vec) / e1 - rho_vec
    else:
        return -rho_vec


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
    drho_t = spmv(L.data,
                  L.indices,
                  L.indptr, rho_t) * dt

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
        dpsi_t += d1(A, psi_t) * dt + sum([d2_vec[n] * dW[a_idx, n]
                                           for n in range(dW_len)
                                           if dW[a_idx, n] != 0])

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
        drho_t += sum([d2_vec[n] * dW[a_idx, n] for n in range(dW_len) if dW[a_idx, n] != 0])

    return rho_t + drho_t


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
        dpsi_t_H = (-1.0j * dt) * spmv(H.data, H.indices, H.indptr, psi_t)

        psi_t_1 = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t)[0] * dW[a_idx,0]
        psi_t_p = psi_t + dpsi_t_H + d1(A, psi_t) * dt + d2(A, psi_t)[0] * sqrt_dt
        psi_t_m = psi_t + dpsi_t_H + d1(A, psi_t) * dt - d2(A, psi_t)[0] * sqrt_dt

        dpsi_t += 0.50 * (d1(A, psi_t_1) + d1(A, psi_t)) * dt + \
            0.25 * (d2(A, psi_t_p)[0] + d2(A, psi_t_m)[0] + 2 * d2(A, psi_t)[0]) * dW[a_idx,0] + \
            0.25 * (d2(A, psi_t_p)[0] - d2(A, psi_t_m)[0]) * (dW[a_idx,0] ** 2 - dt) * sqrt_dt

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
    e1 = cy_expect_rho_vec(M, rho_t)
    
    d2_vec = spmv(M.data, M.indices, M.indptr, rho_t)
    d2_vec2 = spmv(M.data, M.indices, M.indptr, d2_vec)
    e2 = cy_expect_rho_vec(M, d2_vec)
    
    drho_t = _rhs_rho_deterministic(L, rho_t, t, dt, args)
    drho_t += spmv(A[7].data, A[7].indices, A[7].indptr, rho_t)*dt
    drho_t += (d2_vec - e1*rho_t)*dW[0,0]
    drho_t += 0.5 * (d2_vec2 - 2*e1*d2_vec + (-e2 + 2*e1*e1)*rho_t)*(dW[0,0]*dW[0,0] - dt)
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
    e1 = np.array([cy_expect_rho_vec(M[n], rho_t) for n in range(A_len)])
    
    d1_vec = np.sum([spmv(A_ops[n][7].data, A_ops[n][7].indices, A_ops[n][7].indptr, rho_t)
                  for n in range(A_len)], axis=0)
    
    d2_vec = np.array([spmv(M[n].data, M[n].indices, M[n].indptr, rho_t) 
    				   for n in range(A_len)])
    
    #This calculation is suboptimal. We need only values for m>n in case of commuting jump operators.
    d2_vec2 = np.array([[spmv(M[n].data, M[n].indices, M[n].indptr, d2_vec[m]) 
    					for m in range(A_len)] for n in range(A_len)])
    e2 = np.array([[cy_expect_rho_vec(M[n], d2_vec[m]) 
    				for m in range(A_len)] for n in range(A_len)])
    
    drho_t = _rhs_rho_deterministic(L, rho_t, t, dt, args)
    drho_t += d1_vec * dt
    drho_t += np.sum([(d2_vec[n] - e1[n]*rho_t)*dW[n,0] 
    				for n in range(A_len)], axis=0)
    drho_t += 0.5*np.sum([(d2_vec2[n,n] - 2.0*e1[n]*d2_vec[n] + \
    					(-e2[n,n] + 2.0*e1[n]*e1[n])*rho_t)*(dW[n,0]*dW[n,0] - dt) 
    					for n in range(A_len)], axis=0)
    
    #This calculation is suboptimal. We need only values for m>n in case of commuting jump operators.
    drho_t += 0.5*np.sum([(d2_vec2[n,m] - e1[m]*d2_vec[n] - e1[n]*d2_vec[m] + \
    					(-e2[n,m] + 2.0*e1[n]*e1[m])*rho_t)*(dW[n,0]*dW[m,0]) 
    					for (n,m) in np.ndindex(A_len,A_len) if n != m], axis=0)
    
    return rho_t + drho_t
