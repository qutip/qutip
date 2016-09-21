# -*- coding: utf-8 -*-
#
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
This module contains functions for solving stochastic schrodinger and master
equations. The API should not be considered stable, and is subject to change
when we work more on optimizing this module for performance and features.
"""

__all__ = ['ssesolve', 'ssepdpsolve', 'smesolve', 'smepdpsolve']

import numpy as np
import scipy.sparse as sp
from scipy.linalg.blas import get_blas_funcs
try:
    norm = get_blas_funcs("znrm2", dtype=np.float64)
except:
    from scipy.linalg import norm

from numpy.random import RandomState

from qutip.qobj import Qobj, isket
from qutip.states import ket2dm
from qutip.solver import Result
from qutip.expect import expect, expect_rho_vec
from qutip.superoperator import (spre, spost, mat2vec, vec2mat,
                                 liouvillian, lindblad_dissipator)
from qutip.cy.spmatfuncs import cy_expect_psi_csr, spmv, cy_expect_rho_vec
from qutip.cy.stochastic import (cy_d1_rho_photocurrent,
                                 cy_d2_rho_photocurrent)
from qutip.parallel import serial_map
from qutip.ui.progressbar import TextProgressBar
from qutip.solver import Options, _solver_safety_check
from qutip.settings import debug


if debug:
    import qutip.logging_utils
    import inspect
    logger = qutip.logging_utils.get_logger()


class StochasticSolverOptions:
    """Class of options for stochastic solvers such as
    :func:`qutip.stochastic.ssesolve`, :func:`qutip.stochastic.smesolve`, etc.
    Options can be specified either as arguments to the constructor::

        sso = StochasticSolverOptions(nsubsteps=100, ...)

    or by changing the class attributes after creation::

        sso = StochasticSolverOptions()
        sso.nsubsteps = 1000

    The stochastic solvers :func:`qutip.stochastic.ssesolve`,
    :func:`qutip.stochastic.smesolve`, :func:`qutip.stochastic.ssepdpsolve` and
    :func:`qutip.stochastic.smepdpsolve` all take the same keyword arguments as
    the constructor of these class, and internally they use these arguments to
    construct an instance of this class, so it is rarely needed to explicitly
    create an instance of this class.

    Attributes
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    state0 : :class:`qutip.Qobj`
        Initial state vector (ket) or density matrix.

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        List of deterministic collapse operators.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the equation of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    m_ops : list of :class:`qutip.Qobj`
        List of operators representing the measurement operators. The expected
        format is a nested list with one measurement operator for each
        stochastic increament, for each stochastic collapse operator.

    args : dict / list
        List of dictionary of additional problem-specific parameters.
        Implicit methods can adjust tolerance via args = {'tol':value}

    ntraj : int
        Number of trajectors.

    nsubsteps : int
        Number of sub steps between each time-spep given in `times`.

    d1 : function
        Function for calculating the operator-valued coefficient to the
        deterministic increment dt.

    d2 : function
        Function for calculating the operator-valued coefficient to the
        stochastic increment(s) dW_n, where n is in [0, d2_len[.

    d2_len : int (default 1)
        The number of stochastic increments in the process.

    dW_factors : array
        Array of length d2_len, containing scaling factors for each
        measurement operator in m_ops.

    rhs : function
        Function for calculating the deterministic and stochastic contributions
        to the right-hand side of the stochastic differential equation. This
        only needs to be specified when implementing a custom SDE solver.

    generate_A_ops : function
        Function that generates a list of pre-computed operators or super-
        operators. These precomputed operators are used in some d1 and d2
        functions.

    generate_noise : function
        Function for generate an array of pre-computed noise signal.

    homogeneous : bool (True)
        Wheter or not the stochastic process is homogenous. Inhomogenous
        processes are only supported for poisson distributions.

    solver : string
        Name of the solver method to use for solving the stochastic
        equations. Valid values are: 
        1/2 order algorithms: 'euler-maruyama', 'fast-euler-maruyama',
        'pc-euler' is a predictor-corrector method which is more 
        stable than explicit methods,
        1 order algorithms: 'milstein', 'fast-milstein', 'platen',
        'milstein-imp' is semi-implicit Milstein method,
        3/2 order algorithms: 'taylor15', 
        'taylor15-imp' is semi-implicit Taylor 1.5 method.
        Implicit methods can adjust tolerance via args = {'tol':value},
        default is {'tol':1e-6}

    method : string ('homodyne', 'heterodyne', 'photocurrent')
        The name of the type of measurement process that give rise to the
        stochastic equation to solve. Specifying a method with this keyword
        argument is a short-hand notation for using pre-defined d1 and d2
        functions for the corresponding stochastic processes.

    distribution : string ('normal', 'poission')
        The name of the distribution used for the stochastic increments.

    store_measurements : bool (default False)
        Whether or not to store the measurement results in the
        :class:`qutip.solver.SolverResult` instance returned by the solver.

    noise : array
        Vector specifying the noise.

    normalize : bool (default True)
        Whether or not to normalize the wave function during the evolution.

    options : :class:`qutip.solver.Options`
        Generic solver options.

    map_func: function
        A map function or managing the calls to single-trajactory solvers.

    map_kwargs: dictionary
        Optional keyword arguments to the map_func function function.

    progress_bar : :class:`qutip.ui.BaseProgressBar`
        Optional progress bar class instance.

    """
    def __init__(self, H=None, state0=None, times=None, c_ops=[], sc_ops=[],
                 e_ops=[], m_ops=None, args=None, ntraj=1, nsubsteps=1,
                 d1=None, d2=None, d2_len=1, dW_factors=None, rhs=None,
                 generate_A_ops=None, generate_noise=None, homogeneous=True,
                 solver=None, method=None, distribution='normal',
                 store_measurement=False, noise=None, normalize=True,
                 options=None, progress_bar=None, map_func=None,
                 map_kwargs=None):

        if options is None:
            options = Options()

        if progress_bar is None:
            progress_bar = TextProgressBar()

        self.H = H
        self.d1 = d1
        self.d2 = d2
        self.d2_len = d2_len
        self.dW_factors = dW_factors if dW_factors else np.ones(d2_len)
        self.state0 = state0
        self.times = times
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

        self.generate_noise = generate_noise
        self.generate_A_ops = generate_A_ops

        if self.ntraj > 1 and map_func:
            self.map_func = map_func
        else:
            self.map_func = serial_map

        self.map_kwargs = map_kwargs if map_kwargs is not None else {}


def ssesolve(H, psi0, times, sc_ops=[], e_ops=[], _safe_mode=True, **kwargs):
    """
    Solve the stochastic Schrödinger equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the equation of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`
        An instance of the class :class:`qutip.solver.SolverResult`.
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    if _safe_mode:
        _solver_safety_check(H, psi0, sc_ops, e_ops)
    
    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=psi0, times=times,
                                  sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    if sso.generate_A_ops is None:
        sso.generate_A_ops = _generate_psi_A_ops

    if (sso.d1 is None) or (sso.d2 is None):

        if sso.method == 'homodyne':
            sso.d1 = d1_psi_homodyne
            sso.d2 = d2_psi_homodyne
            sso.d2_len = 1
            sso.homogeneous = True
            sso.distribution = 'normal'
            if "dW_factors" not in kwargs:
                sso.dW_factors = np.array([1])
            if "m_ops" not in kwargs:
                sso.m_ops = [[c + c.dag()] for c in sso.sc_ops]

        elif sso.method == 'heterodyne':
            sso.d1 = d1_psi_heterodyne
            sso.d2 = d2_psi_heterodyne
            sso.d2_len = 2
            sso.homogeneous = True
            sso.distribution = 'normal'
            if "dW_factors" not in kwargs:
                sso.dW_factors = np.array([np.sqrt(2), np.sqrt(2)])
            if "m_ops" not in kwargs:
                sso.m_ops = [[(c + c.dag()), (-1j) * (c - c.dag())]
                             for idx, c in enumerate(sso.sc_ops)]

        elif sso.method == 'photocurrent':
            sso.d1 = d1_psi_photocurrent
            sso.d2 = d2_psi_photocurrent
            sso.d2_len = 1
            sso.homogeneous = False
            sso.distribution = 'poisson'

            if "dW_factors" not in kwargs:
                sso.dW_factors = np.array([1])
            if "m_ops" not in kwargs:
                sso.m_ops = [[None] for c in sso.sc_ops]

        else:
            raise Exception("Unrecognized method '%s'." % sso.method)

    if sso.distribution == 'poisson':
        sso.homogeneous = False

    if sso.solver == 'euler-maruyama' or sso.solver is None:
        sso.rhs = _rhs_psi_euler_maruyama

    elif sso.solver == 'platen':
        sso.rhs = _rhs_psi_platen

    else:
        raise Exception("Unrecognized solver '%s'." % sso.solver)

    res = _ssesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def smesolve(H, rho0, times, c_ops=[], sc_ops=[], e_ops=[], 
            _safe_mode=True ,**kwargs):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.

    TODO
    ----
        Add checks for commuting jump operators in Milstein method.
    """

    if debug:
        logger.debug(inspect.stack()[0][3])

    if isket(rho0):
        rho0 = ket2dm(rho0)

    if _safe_mode:
        _solver_safety_check(H, rho0, c_ops+sc_ops, e_ops)
    
    
    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times, c_ops=c_ops,
                                  sc_ops=sc_ops, e_ops=e_ops, **kwargs)

    if (sso.d1 is None) or (sso.d2 is None):

        if sso.method == 'homodyne' or sso.method is None:
            sso.d1 = d1_rho_homodyne
            sso.d2 = d2_rho_homodyne
            sso.d2_len = 1
            sso.homogeneous = True
            sso.distribution = 'normal'
            if "dW_factors" not in kwargs:
                sso.dW_factors = np.array([np.sqrt(1)])
            if "m_ops" not in kwargs:
                sso.m_ops = [[c + c.dag()] for c in sso.sc_ops]

        elif sso.method == 'heterodyne':
            sso.d1 = d1_rho_heterodyne
            sso.d2 = d2_rho_heterodyne
            sso.d2_len = 2
            sso.homogeneous = True
            sso.distribution = 'normal'
            if "dW_factors" not in kwargs:
                sso.dW_factors = np.array([np.sqrt(2), np.sqrt(2)])
            if "m_ops" not in kwargs:
                sso.m_ops = [[(c + c.dag()), -1j * (c - c.dag())]
                             for c in sso.sc_ops]

        elif sso.method == 'photocurrent':
            sso.d1 = cy_d1_rho_photocurrent
            sso.d2 = cy_d2_rho_photocurrent
            sso.d2_len = 1
            sso.homogeneous = False
            sso.distribution = 'poisson'

            if "dW_factors" not in kwargs:
                sso.dW_factors = np.array([1])
            if "m_ops" not in kwargs:
                sso.m_ops = [[None] for c in sso.sc_ops]
        else:
            raise Exception("Unrecognized method '%s'." % sso.method)

    if sso.distribution == 'poisson':
        sso.homogeneous = False

    if sso.generate_A_ops is None:
        sso.generate_A_ops = _generate_rho_A_ops

    if sso.rhs is None:
        if sso.solver == 'euler-maruyama' or sso.solver is None:
            sso.rhs = _rhs_rho_euler_maruyama

        elif sso.solver == 'milstein':
            if sso.method == 'homodyne' or sso.method is None:
                if len(sc_ops) == 1:
                    sso.rhs = _rhs_rho_milstein_homodyne_single
                else:
                    sso.rhs = _rhs_rho_milstein_homodyne

            elif sso.method == 'heterodyne':
                sso.rhs = _rhs_rho_milstein_homodyne
                sso.d2_len = 1
                sso.sc_ops = []
                for sc in iter(sc_ops):
                    sso.sc_ops += [sc / np.sqrt(2), -1.0j * sc / np.sqrt(2)]

        elif sso.solver == 'fast-euler-maruyama' and sso.method == 'homodyne':
            sso.rhs = _rhs_rho_euler_homodyne_fast
            sso.generate_A_ops = _generate_A_ops_Euler

        elif sso.solver == 'fast-milstein':
            sso.generate_A_ops = _generate_A_ops_Milstein
            sso.generate_noise = _generate_noise_Milstein
            if sso.method == 'homodyne' or sso.method is None:
                if len(sc_ops) == 1:
                    sso.rhs = _rhs_rho_milstein_homodyne_single_fast
                elif len(sc_ops) == 2:
                    sso.rhs = _rhs_rho_milstein_homodyne_two_fast
                else:
                    sso.rhs = _rhs_rho_milstein_homodyne_fast

            elif sso.method == 'heterodyne':
                sso.d2_len = 1
                sso.sc_ops = []
                for sc in iter(sc_ops):
                    sso.sc_ops += [sc / np.sqrt(2), -1.0j * sc / np.sqrt(2)]
                if len(sc_ops) == 1:
                    sso.rhs = _rhs_rho_milstein_homodyne_two_fast
                else:
                    sso.rhs = _rhs_rho_milstein_homodyne_fast
                 
        elif sso.solver == 'taylor15':
            sso.generate_A_ops = _generate_A_ops_simple
            sso.generate_noise = _generate_noise_Taylor_15
            if sso.method == 'homodyne' or sso.method is None:
                if len(sc_ops) == 1:
                    sso.rhs = _rhs_rho_taylor_15_one
                #elif len(sc_ops) == 2:
                #    sso.rhs = _rhs_rho_taylor_15_two
                else:
                    raise Exception("Only one stochastic operator is supported")
            else:
                raise Exception("Only homodyne is available")

        elif sso.solver == 'milstein-imp':
            sso.generate_A_ops = _generate_A_ops_implicit
            sso.generate_noise = _generate_noise_Milstein
            if sso.args == None:
                sso.args = {'tol':1e-6}
            if sso.method == 'homodyne' or sso.method is None:
                if len(sc_ops) == 1:
                    sso.rhs = _rhs_rho_milstein_implicit
                else:
                    raise Exception("Only one stochastic operator is supported")
            else:
                raise Exception("Only homodyne is available") 

        elif sso.solver == 'taylor15-imp':  
            sso.generate_A_ops = _generate_A_ops_implicit
            sso.generate_noise = _generate_noise_Taylor_15
            if sso.args == None:
                sso.args = {'tol':1e-6}
            if sso.method == 'homodyne' or sso.method is None:
                if len(sc_ops) == 1:
                    sso.rhs = _rhs_rho_taylor_15_implicit
                else:
                    raise Exception("Only one stochastic operator is supported")
            else:
                raise Exception("Only homodyne is available")

        elif sso.solver == 'pc-euler':
            sso.generate_A_ops = _generate_A_ops_Milstein
            sso.generate_noise = _generate_noise_Milstein # could also work without this
            if sso.method == 'homodyne' or sso.method is None:
                if len(sc_ops) == 1:
                    sso.rhs = _rhs_rho_pred_corr_homodyne_single
                else:
                    raise Exception("Only one stochastic operator is supported")
            else:
                raise Exception("Only homodyne is available")

        else:
            raise Exception("Unrecognized solver '%s'." % sso.solver)

    res = _smesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def ssepdpsolve(H, psi0, times, c_ops, e_ops, **kwargs):
    """
    A stochastic (piecewse deterministic process) PDP solver for wavefunction
    evolution. For most purposes, use :func:`qutip.mcsolve` instead for quantum
    trajectory simulations.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.

    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=psi0, times=times, c_ops=c_ops,
                                  e_ops=e_ops, **kwargs)

    res = _ssepdpsolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    return res


def smepdpsolve(H, rho0, times, c_ops, e_ops, **kwargs):
    """
    A stochastic (piecewse deterministic process) PDP solver for density matrix
    evolution.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix.

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.

    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times, c_ops=c_ops,
                                  e_ops=e_ops, **kwargs)

    res = _smepdpsolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    return res


# -----------------------------------------------------------------------------
# Generic parameterized stochastic Schrodinger equation solver
#
def _ssesolve_generic(sso, options, progress_bar):
    """
    Internal function for carrying out a sse integration. Used by ssesolve.
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    sso.N_store = len(sso.times)
    sso.N_substeps = sso.nsubsteps
    sso.dt = (sso.times[1] - sso.times[0]) / sso.N_substeps
    nt = sso.ntraj

    data = Result()
    data.solver = "ssesolve"
    data.times = sso.times
    data.expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    data.ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    data.noise = []
    data.measurement = []

    # pre-compute collapse operator combinations that are commonly needed
    # when evaluating the RHS of stochastic Schrodinger equations
    sso.A_ops = sso.generate_A_ops(sso.sc_ops, sso.H)

    map_kwargs = {'progress_bar': progress_bar}
    map_kwargs.update(sso.map_kwargs)

    task = _ssesolve_single_trajectory
    task_args = (sso,)
    task_kwargs = {}

    results = sso.map_func(task, list(range(sso.ntraj)),
                           task_args, task_kwargs, **map_kwargs)

    for result in results:
        states_list, dW, m, expect, ss = result
        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)
        data.expect += expect
        data.ss += ss

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([ket2dm(data.states[mm][n])
                            for mm in range(nt)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / nt

    # standard error
    if nt > 1:
        data.se = (data.ss - nt * (data.expect ** 2)) / (nt * (nt - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :])
                   if e.isherm else data.expect[n, :]
                   for n, e in enumerate(sso.e_ops)]

    return data


def _ssesolve_single_trajectory(n, sso):
    """
    Internal function. See ssesolve.
    """
    dt = sso.dt
    times = sso.times
    d1, d2 = sso.d1, sso.d2
    d2_len = sso.d2_len
    e_ops = sso.e_ops
    H_data = sso.H.data
    A_ops = sso.A_ops

    expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)

    psi_t = sso.state0.full().ravel()
    dims = sso.state0.dims

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * np.random.randint(0, 4294967295 // (sso.ntraj+1)))

    if sso.noise is None:
        if sso.homogeneous:
            if sso.distribution == 'normal':
                dW = np.sqrt(dt) * \
                    np.random.randn(len(A_ops), sso.N_store, sso.N_substeps,
                                    d2_len)
            else:
                raise TypeError('Unsupported increment distribution for ' +
                                'homogeneous process.')
        else:
            if sso.distribution != 'poisson':
                raise TypeError('Unsupported increment distribution for ' +
                                'inhomogeneous process.')

            dW = np.zeros((len(A_ops), sso.N_store, sso.N_substeps, d2_len))
    else:
        dW = sso.noise[n]

    states_list = []
    measurements = np.zeros((len(times), len(sso.m_ops), d2_len),
                            dtype=complex)

    for t_idx, t in enumerate(times):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_psi_csr(e.data.data,
                                      e.data.indices,
                                      e.data.indptr, psi_t, 0)
                expect[e_idx, t_idx] += s
                ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t, dims=dims))

        for j in range(sso.N_substeps):

            if sso.noise is None and not sso.homogeneous:
                for a_idx, A in enumerate(A_ops):
                    # dw_expect = norm(spmv(A[0], psi_t)) ** 2 * dt
                    dw_expect = cy_expect_psi_csr(A[3].data,
                                                  A[3].indices,
                                                  A[3].indptr, psi_t, 1) * dt
                    dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect,
                                                               d2_len)

            psi_t = sso.rhs(H_data, psi_t, t + dt * j,
                            A_ops, dt, dW[:, t_idx, j, :], d1, d2, sso.args)

            # optionally renormalize the wave function
            if sso.normalize:
                psi_t /= norm(psi_t)

        if sso.store_measurement:
            for m_idx, m in enumerate(sso.m_ops):
                for dW_idx, dW_factor in enumerate(sso.dW_factors):
                    if m[dW_idx]:
                        m_data = m[dW_idx].data
                        m_expt = cy_expect_psi_csr(m_data.data,
                                                   m_data.indices,
                                                   m_data.indptr,
                                                   psi_t, 0)
                    else:
                        m_expt = 0
                    mm = (m_expt + dW_factor *
                          dW[m_idx, t_idx, :, dW_idx].sum() /
                          (dt * sso.N_substeps))
                    measurements[t_idx, m_idx, dW_idx] = mm

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(2))

    return states_list, dW, measurements, expect, ss


# -----------------------------------------------------------------------------
# Generic parameterized stochastic master equation solver
#
def _smesolve_generic(sso, options, progress_bar):
    """
    Internal function. See smesolve.
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    sso.N_store = len(sso.times)
    sso.N_substeps = sso.nsubsteps
    sso.dt = (sso.times[1] - sso.times[0]) / sso.N_substeps
    nt = sso.ntraj

    data = Result()
    data.solver = "smesolve"
    data.times = sso.times
    data.expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    data.ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    data.noise = []
    data.measurement = []

    # Liouvillian for the deterministic part.
    # needs to be modified for TD systems
    sso.L = liouvillian(sso.H, sso.c_ops)

    # pre-compute suporoperator operator combinations that are commonly needed
    # when evaluating the RHS of stochastic master equations
    sso.A_ops = sso.generate_A_ops(sso.sc_ops, sso.L.data, sso.dt)

    # use .data instead of Qobj ?
    sso.s_e_ops = [spre(e) for e in sso.e_ops]

    if sso.m_ops:
        sso.s_m_ops = [[spre(m) if m else None for m in m_op]
                       for m_op in sso.m_ops]
    else:
        sso.s_m_ops = [[spre(c) for _ in range(sso.d2_len)]
                       for c in sso.sc_ops]

    map_kwargs = {'progress_bar': progress_bar}
    map_kwargs.update(sso.map_kwargs)

    task = _smesolve_single_trajectory
    task_args = (sso,)
    task_kwargs = {}

    results = sso.map_func(task, list(range(sso.ntraj)),
                           task_args, task_kwargs, **map_kwargs)

    for result in results:
        states_list, dW, m, expect, ss = result
        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)
        data.expect += expect
        data.ss += ss

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([data.states[mm][n] for mm in range(nt)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / nt

    # standard error
    if nt > 1:
        data.se = (data.ss - nt * (data.expect ** 2)) / (nt * (nt - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :])
                   if e.isherm else data.expect[n, :]
                   for n, e in enumerate(sso.e_ops)]

    return data


def _smesolve_single_trajectory(n, sso):
    """
    Internal function. See smesolve.
    """
    dt = sso.dt
    times = sso.times
    d1, d2 = sso.d1, sso.d2
    d2_len = sso.d2_len
    L_data = sso.L.data
    N_substeps = sso.N_substeps
    N_store = sso.N_store
    A_ops = sso.A_ops

    rho_t = mat2vec(sso.state0.full()).ravel()
    dims = sso.state0.dims

    expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * np.random.randint(0, 4294967295 // (sso.ntraj+1)))

    if sso.noise is None:
        if sso.generate_noise:
            dW = sso.generate_noise(len(A_ops), N_store, N_substeps,
                                    sso.d2_len, dt)
        elif sso.homogeneous:
            if sso.distribution == 'normal':
                dW = np.sqrt(dt) * np.random.randn(len(A_ops), N_store,
                                                   N_substeps, d2_len)
            else:
                raise TypeError('Unsupported increment distribution for ' +
                                'homogeneous process.')
        else:
            if sso.distribution != 'poisson':
                raise TypeError('Unsupported increment distribution for ' +
                                'inhomogeneous process.')

            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = sso.noise[n]

    states_list = []
    measurements = np.zeros((len(times), len(sso.s_m_ops), d2_len),
                            dtype=complex)

    for t_idx, t in enumerate(times):

        if sso.s_e_ops:
            for e_idx, e in enumerate(sso.s_e_ops):
                s = cy_expect_rho_vec(e.data, rho_t, 0)
                expect[e_idx, t_idx] += s
                ss[e_idx, t_idx] += s ** 2

        if sso.store_states or not sso.s_e_ops:
            states_list.append(Qobj(vec2mat(rho_t), dims=dims))

        rho_prev = np.copy(rho_t)

        for j in range(N_substeps):

            if sso.noise is None and not sso.homogeneous:
                for a_idx, A in enumerate(A_ops):
                    dw_expect = cy_expect_rho_vec(A[4], rho_t, 1) * dt
                    if dw_expect > 0:
                        dW[a_idx, t_idx, j, :] = np.random.poisson(dw_expect,
                                                                   d2_len)
                    else:
                        dW[a_idx, t_idx, j, :] = np.zeros(d2_len)

            rho_t = sso.rhs(L_data, rho_t, t + dt * j,
                            A_ops, dt, dW[:, t_idx, j, :], d1, d2, sso.args)

        if sso.store_measurement:
            for m_idx, m in enumerate(sso.s_m_ops):
                for dW_idx, dW_factor in enumerate(sso.dW_factors):
                    if m[dW_idx]:
                        m_expt = cy_expect_rho_vec(m[dW_idx].data, rho_prev, 0)
                    else:
                        m_expt = 0
                    measurements[t_idx, m_idx, dW_idx] = m_expt + dW_factor * \
                        dW[m_idx, t_idx, :, dW_idx].sum() / (dt * N_substeps)

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(2))

    return states_list, dW, measurements, expect, ss


# -----------------------------------------------------------------------------
# Generic parameterized stochastic SE PDP solver
#
def _ssepdpsolve_generic(sso, options, progress_bar):
    """
    For internal use. See ssepdpsolve.
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    N_store = len(sso.times)
    N_substeps = sso.nsubsteps
    dt = (sso.times[1] - sso.times[0]) / N_substeps
    nt = sso.ntraj

    data = Result()
    data.solver = "sepdpsolve"
    data.times = sso.tlist
    data.expect = np.zeros((len(sso.e_ops), N_store), dtype=complex)
    data.ss = np.zeros((len(sso.e_ops), N_store), dtype=complex)
    data.jump_times = []
    data.jump_op_idx = []

    # effective hamiltonian for deterministic part
    Heff = sso.H
    for c in sso.c_ops:
        Heff += -0.5j * c.dag() * c

    progress_bar.start(sso.ntraj)
    for n in range(sso.ntraj):
        progress_bar.update(n)
        psi_t = sso.state0.full().ravel()

        states_list, jump_times, jump_op_idx = \
            _ssepdpsolve_single_trajectory(data, Heff, dt, sso.times,
                                           N_store, N_substeps,
                                           psi_t, sso.state0.dims,
                                           sso.c_ops, sso.e_ops)

        data.states.append(states_list)
        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([data.states[m][n] for m in range(nt)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / nt

    # standard error
    if nt > 1:
        data.se = (data.ss - nt * (data.expect ** 2)) / (nt * (nt - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :])
                   if e.isherm else data.expect[n, :]
                   for n, e in enumerate(sso.e_ops)]

    return data


def _ssepdpsolve_single_trajectory(data, Heff, dt, times, N_store, N_substeps,
                                   psi_t, dims, c_ops, e_ops):
    """
    Internal function. See ssepdpsolve.
    """
    states_list = []

    phi_t = np.copy(psi_t)

    prng = RandomState()  # todo: seed it
    r_jump, r_op = prng.rand(2)

    jump_times = []
    jump_op_idx = []

    for t_idx, t in enumerate(times):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = cy_expect_psi_csr(
                    e.data.data, e.data.indices, e.data.indptr, psi_t, 0)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t, dims=dims))

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
                jump_times.append(times[t_idx] + dt * j)
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


# -----------------------------------------------------------------------------
# Generic parameterized stochastic ME PDP solver
#
def _smepdpsolve_generic(sso, options, progress_bar):
    """
    For internal use. See smepdpsolve.
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    N_store = len(sso.times)
    N_substeps = sso.nsubsteps
    dt = (sso.times[1] - sso.times[0]) / N_substeps
    nt = sso.ntraj

    data = Result()
    data.solver = "smepdpsolve"
    data.times = sso.times
    data.expect = np.zeros((len(sso.e_ops), N_store), dtype=complex)
    data.jump_times = []
    data.jump_op_idx = []

    # Liouvillian for the deterministic part.
    # needs to be modified for TD systems
    L = liouvillian(sso.H, sso.c_ops)

    progress_bar.start(sso.ntraj)

    for n in range(sso.ntraj):
        progress_bar.update(n)
        rho_t = mat2vec(sso.rho0.full()).ravel()

        states_list, jump_times, jump_op_idx = \
            _smepdpsolve_single_trajectory(data, L, dt, sso.times,
                                           N_store, N_substeps,
                                           rho_t, sso.rho0.dims,
                                           sso.c_ops, sso.e_ops)

        data.states.append(states_list)
        data.jump_times.append(jump_times)
        data.jump_op_idx.append(jump_op_idx)

    progress_bar.finished()

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([data.states[m][n] for m in range(nt)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / sso.ntraj

    # standard error
    if nt > 1:
        data.se = (data.ss - nt * (data.expect ** 2)) / (nt * (nt - 1))
    else:
        data.se = None

    return data


def _smepdpsolve_single_trajectory(data, L, dt, times, N_store, N_substeps,
                                   rho_t, dims, c_ops, e_ops):
    """
    Internal function. See smepdpsolve.
    """
    states_list = []

    rho_t = np.copy(rho_t)
    sigma_t = np.copy(rho_t)

    prng = RandomState()  # todo: seed it
    r_jump, r_op = prng.rand(2)

    jump_times = []
    jump_op_idx = []

    for t_idx, t in enumerate(times):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                data.expect[e_idx, t_idx] += expect_rho_vec(e, rho_t)
        else:
            states_list.append(Qobj(vec2mat(rho_t), dims=dims))

        for j in range(N_substeps):

            if sigma_t.norm() < r_jump:
                # jump occurs
                p = np.array([expect(c.dag() * c, rho_t) for c in c_ops])
                p = np.cumsum(p / np.sum(p))
                n = np.where(p >= r_op)[0][0]

                # apply jump
                rho_t = c_ops[n] * rho_t * c_ops[n].dag()
                rho_t /= expect(c_ops[n].dag() * c_ops[n], rho_t)
                sigma_t = np.copy(rho_t)

                # store info about jump
                jump_times.append(times[t_idx] + dt * j)
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


# -----------------------------------------------------------------------------
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


def d1_psi_homodyne(t, psi, A, args):
    """
    OK
    Need to cythonize

    .. math::

        D_1(C, \psi) = \\frac{1}{2}(\\langle C + C^\\dagger\\rangle\\C psi -
        C^\\dagger C\\psi - \\frac{1}{4}\\langle C + C^\\dagger\\rangle^2\\psi)

    """

    e1 = cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    return 0.5 * (e1 * spmv(A[0], psi) -
                  spmv(A[3], psi) -
                  0.25 * e1 ** 2 * psi)


def d2_psi_homodyne(t, psi, A, args):
    """
    OK
    Need to cythonize

    .. math::

        D_2(\psi, t) = (C - \\frac{1}{2}\\langle C + C^\\dagger\\rangle)\\psi

    """

    e1 = cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    return [spmv(A[0], psi) - 0.5 * e1 * psi]


def d1_psi_heterodyne(t, psi, A, args):
    """
    Need to cythonize

    .. math::

        D_1(\psi, t) = -\\frac{1}{2}(C^\\dagger C -
        \\langle C^\\dagger \\rangle C +
        \\frac{1}{2}\\langle C \\rangle\\langle C^\\dagger \\rangle))\psi

    """
    e_C = cy_expect_psi_csr(A[0].data, A[0].indices, A[0].indptr, psi, 0)
    B = A[0].T.conj()
    e_Cd = cy_expect_psi_csr(B.data, B.indices, B.indptr, psi, 0)

    return (-0.5 * spmv(A[3], psi) +
            0.5 * e_Cd * spmv(A[0], psi) -
            0.25 * e_C * e_Cd * psi)


def d2_psi_heterodyne(t, psi, A, args):
    """
    Need to cythonize

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


def d1_psi_photocurrent(t, psi, A, args):
    """
    Need to cythonize.

    Note: requires poisson increments

    .. math::

        D_1(\psi, t) = - \\frac{1}{2}(C^\dagger C \psi - ||C\psi||^2 \psi)

    """
    return (-0.5 * (spmv(A[3], psi)
            - norm(spmv(A[0], psi)) ** 2 * psi))


def d2_psi_photocurrent(t, psi, A, args):
    """
    Need to cythonize

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
#     A[_idx_A_L] = spre(a) = A_L
#     A[_idx_A_R] = spost(a) = A_R
#     A[_idx_Ad_L] = spre(a.dag()) = Ad_L
#     A[_idx_Ad_R] = spost(a.dag()) = Ad_R
#     A[_idx_AdA_L] = spre(a.dag() * a) = (Ad A)_L
#     A[_idx_AdA_R] = spost(a.dag() * a) = (Ad A)_R
#     A[_idx_A_LxAd_R] = (spre(a) * spost(a.dag()) = A_L * Ad_R
#     A[_idx_LD] = lindblad_dissipator(a)

_idx_A_L = 0
_idx_A_R = 1
_idx_Ad_L = 2
_idx_Ad_R = 3
_idx_AdA_L = 4
_idx_AdA_R = 5
_idx_A_LxAd_R = 6
_idx_LD = 7


def _generate_rho_A_ops(sc, L, dt):
    """
    pre-compute superoperator operator combinations that are commonly needed
    when evaluating the RHS of stochastic master equations
    """
    out = []
    for c_idx, c in enumerate(sc):
        n = c.dag() * c
        out.append([spre(c).data,
                    spost(c).data,
                    spre(c.dag()).data,
                    spost(c.dag()).data,
                    spre(n).data,
                    spost(n).data,
                    (spre(c) * spost(c.dag())).data,
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

    # XXX: fix this!
    out1[0][0].indices = np.array(out1[0][0].indices, dtype=np.int32)
    out1[0][0].indptr = np.array(out1[0][0].indptr, dtype=np.int32)

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

    # XXX: fix this!
    out1[0][0].indices = np.array(out1[0][0].indices, dtype=np.int32)
    out1[0][0].indptr = np.array(out1[0][0].indptr, dtype=np.int32)

    return out1


def _generate_A_ops_simple(sc, L, dt):
    """
    pre-compute superoperator operator combinations that are commonly needed
    when evaluating the RHS of stochastic master equations
    """

    A_len = len(sc)
    temp = [spre(c).data + spost(c.dag()).data for c in sc]
    tempL = (L + np.sum([lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) # Lagrangian

    out = []
    out += temp
    out += [tempL]
    
    out1 = [out]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]

    return out1
    
    
def _generate_A_ops_implicit(sc, L, dt):
    """
    pre-compute superoperator operator combinations that are commonly needed
    when evaluating the RHS of stochastic master equations
    """

    A_len = len(sc)
    temp = [spre(c).data + spost(c.dag()).data for c in sc]
    tempL = (L + np.sum([lindblad_dissipator(c, data_only=True) for c in sc], axis=0)) # Lagrangian

    out = []
    out += temp
    out += [sp.eye(L.shape[0], format='csr') - 0.5*dt*tempL]
    out += [tempL]
    
    out1 = [out]
    # the following hack is required for compatibility with old A_ops
    out1 += [[] for n in range(A_len - 1)]

    return out1


def _generate_noise_Milstein(sc_len, N_store, N_substeps, d2_len, dt):
    """
    generate noise terms for the fast Milstein scheme
    """
    dW_temp = np.sqrt(dt) * np.random.randn(sc_len, N_store, N_substeps, 1)
    if sc_len == 1:
        noise = np.vstack([dW_temp, 0.5 * (dW_temp * dW_temp - dt *
                          np.ones((sc_len, N_store, N_substeps, 1)))])
    else:
        noise = np.vstack(
            [dW_temp,
             0.5 * (dW_temp * dW_temp -
                    dt * np.ones((sc_len, N_store, N_substeps, 1)))] +
            [[dW_temp[n] * dW_temp[m]
              for (n, m) in np.ndindex(sc_len, sc_len) if n > m]])

    return noise

def _generate_noise_Taylor_15(sc_len, N_store, N_substeps, d2_len, dt):
    """
    generate noise terms for the strong Taylor 1.5 scheme
    """
    U1 = np.random.randn(sc_len, N_store, N_substeps, 1) 
    U2 = np.random.randn(sc_len, N_store, N_substeps, 1) 
    dW = U1 * np.sqrt(dt) 
    dZ = 0.5 * dt**(3./2) * (U1 + 1./np.sqrt(3) * U2) 

    if sc_len == 1:
        noise = np.vstack([ dW, 0.5 * (dW * dW - dt), dZ, dW * dt - dZ, 0.5 * (1./3. * dW**2 - dt) * dW ])                    
    
    elif sc_len == 2:
        noise = np.vstack([ dW, 0.5 * (dW**2 - dt), dZ, dW * dt - dZ, 0.5 * (1./3. * dW**2 - dt) * dW] 
                    + [[dW[n] * dW[m] for (n, m) in np.ndindex(sc_len, sc_len) if n < m]]  # Milstein
                    + [[0.5 * dW[n] * (dW[m]**2 - dt) for (n, m) in np.ndindex(sc_len, sc_len) if n != m]])

    #else:
        #noise = np.vstack([ dW, 0.5 * (dW**2 - dt), dZ, dW * dt - dZ, 0.5 * (1./3. * dW**2 - dt) * dW] 
                    #+ [[dW[n] * dW[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m]]  # Milstein
                    #+ [[0.5 * dW[n] * (dW[m]**2 - dt) for (n, m) in np.ndindex(sc_len, sc_len) if n != m]]
                    #+ [[dW[n] * dW[m] * dW[k] for (n, m, k) in np.ndindex(sc_len, sc_len, sc_len) if n>m>k]])  
    else:
        raise Exception("too many stochastic operators")

    return noise


def sop_H(A, rho_vec):
    """
    Evaluate the superoperator

    H[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger] rho
            -> (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec] rho_vec

    Need to cythonize, add A_L + Ad_R to precomputed operators
    """
    M = A[0] + A[3]

    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    return spmv(M, rho_vec) - e1 * rho_vec


def sop_G(A, rho_vec):
    """
    Evaluate the superoperator

    G[a] rho = a rho a^\dagger / Tr[a rho a^\dagger] - rho
            -> A_L Ad_R rho_vec / Tr[A_L Ad_R rho_vec] - rho_vec

    Need to cythonize, add A_L + Ad_R to precomputed operators
    """

    e1 = cy_expect_rho_vec(A[6], rho_vec, 0)

    if e1 > 1e-15:
        return spmv(A[6], rho_vec) / e1 - rho_vec
    else:
        return -rho_vec


def d1_rho_homodyne(t, rho_vec, A, args):
    """
    D1[a] rho = lindblad_dissipator(a) * rho

    Need to cythonize
    """
    return spmv(A[7], rho_vec)


def d2_rho_homodyne(t, rho_vec, A, args):
    """
    D2[a] rho = a rho + rho a^\dagger - Tr[a rho + rho a^\dagger]
              = (A_L + Ad_R) rho_vec - E[(A_L + Ad_R) rho_vec]

    Need to cythonize, add A_L + Ad_R to precomputed operators
    """
    M = A[0] + A[3]

    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    return [spmv(M, rho_vec) - e1 * rho_vec]


def d1_rho_heterodyne(t, rho_vec, A, args):
    """
    Need to cythonize, docstrings
    """
    return spmv(A[7], rho_vec)


def d2_rho_heterodyne(t, rho_vec, A, args):
    """
    Need to cythonize, docstrings
    """
    M = A[0] + A[3]
    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    d1 = spmv(M, rho_vec) - e1 * rho_vec
    M = A[0] - A[3]
    e1 = cy_expect_rho_vec(M, rho_vec, 0)
    d2 = spmv(M, rho_vec) - e1 * rho_vec
    return [1.0 / np.sqrt(2) * d1, -1.0j / np.sqrt(2) * d2]


def d1_rho_photocurrent(t, rho_vec, A, args):
    """
    Need to cythonize, add (AdA)_L + AdA_R to precomputed operators
    """
    n_sum = A[4] + A[5]
    e1 = cy_expect_rho_vec(n_sum, rho_vec, 0)
    return 0.5 * (e1 * rho_vec - spmv(n_sum, rho_vec))


def d2_rho_photocurrent(t, rho_vec, A, args):
    """
    Need to cythonize, add (AdA)_L + AdA_R to precomputed operators
    """
    e1 = cy_expect_rho_vec(A[6], rho_vec, 0)
    if e1.real > 1e-15:
        return [spmv(A[6], rho_vec) / e1 - rho_vec]
    else:
        return [-rho_vec]


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Euler-Maruyama rhs functions for the stochastic Schrodinger and master
# equations
#

def _rhs_psi_euler_maruyama(H, psi_t, t, A_ops, dt, dW, d1, d2, args):
    """
    Euler-Maruyama rhs function for wave function solver.
    """
    dW_len = len(dW[0, :])
    dpsi_t = _rhs_psi_deterministic(H, psi_t, t, dt, args)

    for a_idx, A in enumerate(A_ops):
        d2_vec = d2(t, psi_t, A, args)
        dpsi_t += d1(t, psi_t, A, args) * dt + \
            np.sum([d2_vec[n] * dW[a_idx, n]
                    for n in range(dW_len) if dW[a_idx, n] != 0], axis=0)

    return psi_t + dpsi_t


def _rhs_rho_euler_maruyama(L, rho_t, t, A_ops, dt, dW, d1, d2, args):
    """
    Euler-Maruyama rhs function for density matrix solver.
    """
    dW_len = len(dW[0, :])

    drho_t = _rhs_rho_deterministic(L, rho_t, t, dt, args)

    for a_idx, A in enumerate(A_ops):
        d2_vec = d2(t, rho_t, A, args)
        drho_t += d1(t, rho_t, A, args) * dt
        drho_t += np.sum([d2_vec[n] * dW[a_idx, n]
                          for n in range(dW_len) if dW[a_idx, n] != 0], axis=0)

    return rho_t + drho_t


def _rhs_rho_euler_homodyne_fast(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    Fast Euler-Maruyama for homodyne detection.
    """

    dW = ddW[:, 0]

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2)

    drho_t = d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    drho_t += (1.0 - np.inner(np.real(e), dW)) * rho_t
    return drho_t


# -----------------------------------------------------------------------------
# Platen method
#
def _rhs_psi_platen(H, psi_t, t, A_ops, dt, dW, d1, d2, args):
    """
    TODO: support multiple stochastic increments

    .. note::

        Experimental.

    """

    sqrt_dt = np.sqrt(dt)

    dpsi_t = _rhs_psi_deterministic(H, psi_t, t, dt, args)

    for a_idx, A in enumerate(A_ops):
        # XXX: This needs to be revised now that
        # dpsi_t is the change for all stochastic collapse operators

        # TODO: needs to be updated to support mutiple Weiner increments
        dpsi_t_H = (-1.0j * dt) * spmv(H, psi_t)

        psi_t_1 = (psi_t + dpsi_t_H +
                   d1(A, psi_t) * dt +
                   d2(A, psi_t)[0] * dW[a_idx, 0])
        psi_t_p = (psi_t + dpsi_t_H +
                   d1(A, psi_t) * dt +
                   d2(A, psi_t)[0] * sqrt_dt)
        psi_t_m = (psi_t + dpsi_t_H +
                   d1(A, psi_t) * dt -
                   d2(A, psi_t)[0] * sqrt_dt)

        dpsi_t += (
            0.50 * (d1(A, psi_t_1) + d1(A, psi_t)) * dt +
            0.25 * (d2(A, psi_t_p)[0] + d2(A, psi_t_m)[0] +
                    2 * d2(A, psi_t)[0]) * dW[a_idx, 0] +
            0.25 * (d2(A, psi_t_p)[0] - d2(A, psi_t_m)[0]) *
            (dW[a_idx, 0] ** 2 - dt) / sqrt_dt
            )

    return dpsi_t


# -----------------------------------------------------------------------------
# Milstein rhs functions for the stochastic master equation
#
def _rhs_rho_milstein_homodyne_single(L, rho_t, t, A_ops, dt, dW, d1, d2,
                                      args):
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
    drho_t += 0.5 * (d2_vec2 - 2 * e1 * d2_vec + (-e2 + 2 * e1 * e1) *
                     rho_t) * (dW[0, 0] * dW[0, 0] - dt)
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
    drho_t += 0.5 * np.sum(
        [(d2_vec2[n, n] - 2.0 * e1[n] * d2_vec[n] +
         (-e2[n, n] + 2.0 * e1[n] * e1[n]) * rho_t) * (dW[n, 0]*dW[n, 0] - dt)
         for n in range(A_len)], axis=0)

    # This calculation is suboptimal. We need only values for m>n in case of
    # commuting jump operators.
    drho_t += 0.5 * np.sum(
        [(d2_vec2[n, m] - e1[m] * d2_vec[n] - e1[n] * d2_vec[m] +
         (-e2[n, m] + 2.0 * e1[n] * e1[m]) * rho_t) * (dW[n, 0] * dW[m, 0])
         for (n, m) in np.ndindex(A_len, A_len) if n != m], axis=0)

    return rho_t + drho_t


def _rhs_rho_milstein_homodyne_single_fast(L, rho_t, t, A, dt, ddW, d1, d2,
                                           args):
    """
    fast Milstein for homodyne detection with 1 stochastic operator
    """
    dW = np.copy(ddW[:, 0])

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))

    e[1] -= 2.0 * e[0] * e[0]

    drho_t = - np.inner(e, dW) * rho_t
    dW[0] -= 2.0 * e[0] * dW[1]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t


def _rhs_rho_milstein_homodyne_two_fast(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    fast Milstein for homodyne detection with 2 stochastic operators
    """
    dW = np.copy(ddW[:, 0])

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))
    d_vec[-2] -= np.dot(e[:2][::-1], d_vec[:2])

    e[2:4] -= 2.0 * e[:2] * e[:2]
    e[4] -= 2.0 * e[1] * e[0]

    drho_t = - np.inner(e, dW) * rho_t
    dW[:2] -= 2.0 * e[:2] * dW[2:4]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t


def _rhs_rho_milstein_homodyne_fast(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    fast Milstein for homodyne detection with >2 stochastic operators
    """
    dW = np.copy(ddW[:, 0])
    sc_len = len(A)
    sc2_len = 2 * sc_len

    d_vec = spmv(A[0][0], rho_t).reshape(-1, len(rho_t))
    e = np.real(d_vec[:-1].reshape(
        -1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))
    d_vec[sc2_len:-1] -= np.array(
        [e[m] * d_vec[n] + e[n] * d_vec[m]
         for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    e[sc_len:sc2_len] -= 2.0 * e[:sc_len] * e[:sc_len]
    e[sc2_len:] -= 2.0 * np.array(
        [e[n] * e[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    drho_t = - np.inner(e, dW) * rho_t
    dW[:sc_len] -= 2.0 * e[:sc_len] * dW[sc_len:sc2_len]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t


def _rhs_rho_taylor_15_one(L, rho_t, t, A, dt, ddW, d1, d2,
                                           args):
    """
    strong order 1.5 Tylor scheme for homodyne detection with 1 stochastic operator
    """

    dW = ddW[:, 0]
    A = A[0]

    #reusable operators and traces
    a = A[-1] * rho_t
    e0 = cy_expect_rho_vec(A[0], rho_t, 1)
    b = A[0] * rho_t - e0 * rho_t
    TrAb = cy_expect_rho_vec(A[0], b, 1)
    Lb = A[0] * b - TrAb * rho_t - e0 * b
    TrALb = cy_expect_rho_vec(A[0], Lb, 1)
    TrAa = cy_expect_rho_vec(A[0], a, 1)

    drho_t = a * dt
    drho_t += b * dW[0]
    drho_t += Lb * dW[1] # Milstein term

    # new terms: 
    drho_t += A[-1] * b * dW[2]
    drho_t += (A[0] * a - TrAa * rho_t - e0 * a - TrAb * b) * dW[3]
    drho_t += A[-1] * a * (0.5 * dt*dt)
    drho_t += (A[0] * Lb - TrALb * rho_t - (2 * TrAb) * b - e0 * Lb) * dW[4] 
        
    return rho_t + drho_t

#include _rhs_rho_Taylor_15_two#

def _rhs_rho_milstein_implicit(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    Drift implicit Milstein (theta = 1/2, eta = 0)
    Wang, X., Gan, S., & Wang, D. (2012). 
    A family of fully implicit Milstein methods for stiff stochastic differential 
    equations with multiplicative noise. 
    BIT Numerical Mathematics, 52(3), 741–772.
    """
    
    dW = ddW[:, 0]
    A = A[0]
    

    #reusable operators and traces
    a = A[-1] * rho_t * (0.5 * dt)
    e0 = cy_expect_rho_vec(A[0], rho_t, 1)
    b = A[0] * rho_t - e0 * rho_t
    TrAb = cy_expect_rho_vec(A[0], b, 1)

    drho_t = b * dW[0] 
    drho_t += a
    drho_t += (A[0] * b - TrAb * rho_t - e0 * b) * dW[1] # Milstein term
    drho_t += rho_t

    v, check = sp.linalg.bicgstab(A[-2], drho_t, x0 = drho_t + a, tol=args['tol'])

    return v
    
def _rhs_rho_taylor_15_implicit(L, rho_t, t, A, dt, ddW, d1, d2, args):
    """
    Drift implicit Taylor 1.5 (alpha = 1/2, beta = doesn't matter)
    Chaptert 12.2 Eq. (2.18) in Numerical Solution of Stochastic Differential Equations
    By Peter E. Kloeden, Eckhard Platen
    """
    
    dW = ddW[:, 0]
    A = A[0]

    #reusable operators and traces
    a = A[-1] * rho_t
    e0 = cy_expect_rho_vec(A[0], rho_t, 1)
    b = A[0] * rho_t - e0 * rho_t
    TrAb = cy_expect_rho_vec(A[0], b, 1)
    Lb = A[0] * b - TrAb * rho_t - e0 * b
    TrALb = cy_expect_rho_vec(A[0], Lb, 1)
    TrAa = cy_expect_rho_vec(A[0], a, 1)

    drho_t = b * dW[0] 
    drho_t += Lb * dW[1] # Milstein term
    xx0 = (drho_t + a * dt) + rho_t #starting vector for the linear solver (Milstein prediction)
    drho_t += (0.5 * dt) * a

    # new terms: 
    drho_t += A[-1] * b * (dW[2] - 0.5*dW[0]*dt)
    drho_t += (A[0] * a - TrAa * rho_t - e0 * a - TrAb * b) * dW[3]

    drho_t += (A[0] * Lb - TrALb * rho_t - (2 * TrAb) * b - e0 * Lb) * dW[4]
    drho_t += rho_t

    v, check = sp.linalg.bicgstab(A[-2], drho_t, x0 = xx0, tol=args['tol'])

    return v
    
def _rhs_rho_pred_corr_homodyne_single(L, rho_t, t, A, dt, ddW, d1, d2,
                                           args):
    """
    1/2 predictor-corrector scheme for homodyne detection with 1 stochastic operator
    """
    dW = ddW[:, 0]
    
    #predictor

    d_vec = (A[0][0] * rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))

    a_pred = np.copy(d_vec[-1])
    b_pred = - e[0] * rho_t
    b_pred += d_vec[0]

    pred_rho_t = np.copy(a_pred)
    pred_rho_t += b_pred * dW[0]
    pred_rho_t += rho_t

    a_pred -= ((d_vec[1] - e[1] * rho_t) - (2.0 * e[0]) * b_pred) * (0.5 * dt)
    
    #corrector

    d_vec = (A[0][0] * pred_rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A[0][1], A[0][1]).trace(axis1=1, axis2=2))

    a_corr = d_vec[-1]
    b_corr = - e[0] * pred_rho_t
    b_corr += d_vec[0]

    a_corr -= ((d_vec[1] - e[1] * pred_rho_t) - (2.0 * e[0]) * b_corr) * (0.5 * dt)
    a_corr += a_pred
    a_corr *= 0.5

    b_corr += b_pred
    b_corr *= 0.5 * dW[0]

    corr_rho_t = a_corr
    corr_rho_t += b_corr
    corr_rho_t += rho_t

    return corr_rho_t
