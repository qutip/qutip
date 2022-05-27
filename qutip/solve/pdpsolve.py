# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import RandomState

from .. import (
    Qobj, expect, spre, spost, stack_columns, unstack_columns, liouvillian,
)
from ..core import data as _data
from .solver import Result, SolverOptions
from ..parallel import serial_map
from ..ui.progressbar import TextProgressBar
from ..settings import settings
debug = settings.core['debug']


class StochasticSolverOptions:
    """
    Class of options for stochastic (piecewse deterministic process) PDP
    solvers such as :func:`qutip.pdpsolve.ssepdpsolve`,
    :func:`qutip.pdpsolve.smepdpsolve`.
    Options can be specified either as arguments to the constructor::

        sso = StochasticSolverOptions(nsubsteps=100, ...)

    or by changing the class attributes after creation::

        sso = StochasticSolverOptions()
        sso.nsubsteps = 1000

    The stochastic solvers :func:`qutip.pdpsolve.ssepdpsolve` and
    :func:`qutip.pdpsolve.smepdpsolve` all take the same keyword arguments as
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

    distribution : string ('normal', 'poisson')
        The name of the distribution used for the stochastic increments.

    store_measurements : bool (default False)
        Whether or not to store the measurement results in the
        :class:`qutip.solver.Result` instance returned by the solver.

    noise : array
        Vector specifying the noise.

    normalize : bool (default True)
        Whether or not to normalize the wave function during the evolution.

    options : :class:`qutip.solver.SolverOptions`
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
            options = SolverOptions()

        if progress_bar is None:
            progress_bar = TextProgressBar()

        self.H = H
        self.d1 = d1
        self.d2 = d2
        self.d2_len = d2_len
        self.dW_factors = dW_factors  # if dW_factors else np.ones(d2_len)
        self.state0 = state0
        self.times = times
        self.c_ops = c_ops
        self.sc_ops = sc_ops
        self.e_ops = e_ops
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
        self.store_states = options['store_states']
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

        # Does any operator depend on time?
        self.td = False
        if not isinstance(H, Qobj):
            self.td = True
        for ops in c_ops:
            if not isinstance(ops, Qobj):
                self.td = True
        for ops in sc_ops:
            if not isinstance(ops, Qobj):
                self.td = True


def main_ssepdpsolve(H, psi0, times, c_ops, e_ops, **kwargs):
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

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
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


def main_smepdpsolve(H, rho0, times, c_ops, e_ops, **kwargs):
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
        to the equation of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
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
# Generic parameterized stochastic SE PDP solver
#
def _ssepdpsolve_generic(sso, options, progress_bar):
    """
    For internal use. See ssepdpsolve.
    """
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
        psi_t = _data.dense.fast_from_numpy(sso.state0.full().ravel())

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
    if options['average_states'] and np.any(data.states):
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


def _ssepdpsolve_single_trajectory(data, Heff, dt, times, N_store, N_substeps, psi_t, dims, c_ops, e_ops):
    """
    Internal function. See ssepdpsolve.
    """
    states_list = []

    phi_t = psi_t.copy()

    prng = RandomState()  # todo: seed it
    r_jump, r_op = prng.rand(2)

    jump_times = []
    jump_op_idx = []

    for t_idx, t in enumerate(times):

        if e_ops:
            for e_idx, e in enumerate(e_ops):
                s = _data.expect(e, psi_t)
                data.expect[e_idx, t_idx] += s
                data.ss[e_idx, t_idx] += s ** 2
        else:
            states_list.append(Qobj(psi_t.to_array(), dims=dims))

        for j in range(N_substeps):

            if _data.norm.l2(phi_t) ** 2 < r_jump:
                # jump occurs
                p = np.array([
                    _data.norm.l2(_data.matmul(c.data, psi_t)) ** 2
                    for c in c_ops
                ])
                p = np.cumsum(p / np.sum(p))
                n = np.where(p >= r_op)[0][0]

                # apply jump
                psi_t = _data.matmul(c_ops[n].data, psi_t)
                psi_t /= _data.norm.l2(psi_t)
                phi_t = psi_t.copy()

                # store info about jump
                jump_times.append(times[t_idx] + dt * j)
                jump_op_idx.append(n)

                # get new random numbers for next jump
                r_jump, r_op = prng.rand(2)

            # deterministic evolution wihtout correction for norm decay
            dphi_t = (-1j*dt) * _data.matmul(Heff.data, phi_t)

            # deterministic evolution with correction for norm decay
            dpsi_t = (-1j*dt) * _data.matmul(Heff.data, psi_t)
            A = 0.5 * np.sum([
                _data.norm.l2(_data.matmul(c.data, psi_t)) ** 2
                for c in c_ops
            ])
            dpsi_t = _data.add(dpsi_t, psi_t, scale=dt*A)

            # increment wavefunctions
            phi_t = _data.add(phi_t, dphi_t)
            psi_t = _data.add(psi_t, dpsi_t)

            # ensure that normalized wavefunction remains normalized
            # this allows larger time step than otherwise would be possible
            psi_t = _data.mul(psi_t, 1/_data.norm.l2(psi_t))

    return states_list, jump_times, jump_op_idx


# -----------------------------------------------------------------------------
# Generic parameterized stochastic ME PDP solver
#
def _smepdpsolve_generic(sso, options, progress_bar):
    """
    For internal use. See smepdpsolve.
    """
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
        rho_t = _data.dense.fast_from_numpy(sso.rho0.full())
        rho_t = _data.column_stack_dense(rho_t)

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
    if options['average_states'] and np.any(data.states):
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

def _smepdpsolve_single_trajectory(data, L, dt, times, N_store, N_substeps, rho_t, dims, c_ops, e_ops):
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
                data.expect[e_idx, t_idx] +=\
                    _data.expect_super(e, rho_t)
        else:
            states_list.append(Qobj(unstack_columns(rho_t), dims=dims))

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
            dsigma_t = _data.matmul(L.data, sigma_t) * dt

            # deterministic evolution with correction for norm decay
            drho_t = _data.matmul(L.data, rho_t) * dt

            # increment density matrices
            sigma_t = _data.add(sigma_t, dsigma_t)
            rho_t = _data.add(rho_t, drho_t)

    return states_list, jump_times, jump_op_idx
