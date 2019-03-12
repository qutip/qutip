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

__all__ = ['mcsolve']

import os
import numpy as np
from numpy.random import RandomState, randint
from scipy.integrate import ode
from scipy.integrate._ode import zvode

from types import FunctionType, BuiltinFunctionType
from functools import partial

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.parallel import parfor, parallel_map, serial_map
from qutip.cy.mcsolve import cy_mc_run_ode
# cy_mc_run_ode = None
from qutip.sesolve import sesolve
from qutip.solver import (Options, Result,
                          solver_safe, SolverSystem)
from qutip.settings import debug
from qutip.ui.progressbar import TextProgressBar, BaseProgressBar
import qutip.settings

if debug:
    import inspect

#
# Internal, global variables for storing references to dynamically loaded
# cython functions

# Todo: use real warning
def warn(text):
    print(text)


class SolverConfiguration():
    def __init__(self, H, psi0, tlist, c_ops, e_ops,
                 ntraj, args, options,
                 progress_bar, map_func, map_kwargs,
                 _safe_mode=True):

        if debug:
            print(inspect.stack()[0][3])

        # set general items
        self.tlist = np.asarray(tlist, dtype=float)

        try:
            self.ntraj = int(ntraj)
        except:
            try:
                self.ntraj = max(ntraj)
            except:
                self.ntraj = options.ntraj

        self.map_func = map_func
        self.map_kwargs = map_kwargs
        self.options = options
        # set num_cpus to the value given in qutip.settings if none in Options
        if not self.options.num_cpus:
            self.options.num_cpus = qutip.settings.num_cpus
            if self.options.num_cpus == 1:
                # fallback on serial_map if num_cpu == 1, since there is no
                # benefit of starting multiprocessing in this case
                self.map_func = serial_map

        # set options on ouput states
        if self.options.steady_state_average:
            self.options.average_states = True
        if self.options.average_states:
            self.options.store_states = True

        # set norm finding constants
        self.norm_tol = self.options.norm_tol
        self.norm_t_tol = self.options.norm_t_tol
        self.norm_steps = self.options.norm_steps

        if progress_bar:
            if progress_bar is True:
                self.progress_bar = TextProgressBar()
            else:
                self.progress_bar = progress_bar
        else:
            self.progress_bar = BaseProgressBar()

        # set the physics
        if not psi0.isket:
            raise Exception("Initial state must be a state vector.")

        # set initial value data
        if options.tidy:
            self.psi0 = psi0.tidyup(options.atol).full().ravel()
        else:
            self.psi0 = psi0.full().ravel()
        self.psi0_dims = psi0.dims
        self.psi0_shape = psi0.shape

        # take care of expectation values, if any
        if e_ops:
            self.e_num = len(e_ops)
            self.e_ops_cdata = np.array([e.data.cdata for e in e_ops],
                                        dtype=object)
            self.e_ops_isherm = [e.isherm for e in e_ops]
        else:
            self.e_num = 0
            self.options.store_states = True

        # SETUP ODE DATA IF NONE EXISTS OR NOT REUSING
        # --------------------------------------------


        dopri = (options.method == "dopri5")
        if not options.rhs_reuse or not config.H_is_set:
            # Find the type of time-dependence for H and c_ops
            # h_tflag
            #   1 : QobjEvo
            #   2 : H is a functions
            #   3 : QobjEvo with state
            #   3 : H functions with state

            # define :
            # self.rhs : the function called by the integrator
            # self.td_c_ops : to get needed c*vec function
            # self.td_n_ops : to get the expectation value of the c_op
            self.H_td = None
            self.Hc_td = None
            self.rhs = None
            self.Hc_rhs = None
            self.h_func = None
            self.h_func_args = {}

            self.c_num = len(c_ops)
            self.td_c_ops = []
            self.td_n_ops = []
            c_types = []
            for c in c_ops:
                cevo = QobjEvo(c, args, tlist)
                cevo.compile()
                cdc = cevo._cdc()
                cdc.compile()
                self.td_c_ops.append(cevo)
                self.td_n_ops.append(cdc)
            self.td_c_ops = np.array(self.td_c_ops, dtype=object)
            self.td_n_ops = np.array(self.td_n_ops, dtype=object)

            Hc_td = self.td_n_ops[0] * 0.
            for c in self.td_n_ops:
                Hc_td += -0.5 * c

            self.h_func_args = {}
            if not options.rhs_with_state:
                try:
                    # Can H be made into an QobjEvo?
                    H_td = QobjEvo(H, args, tlist)
                    H_td *= -1j
                    H_td += Hc_td
                    H_td.compile()
                    self.H_td = H_td
                    self.h_tflag = 1
                    self.rhs = H_td.compiled_qobjevo.mul_vec

                except:
                    self.h_tflag = 2
                    self.h_func = H
                    self.h_func_args = args
                    self.Hc_td = Hc_td
                    self.Hc_td.compile()
                    self.Hc_rhs = Hc_td.compiled_qobjevo.mul_vec
                    self.rhs = _funcrhs

            else:
                self.Hc_td = Hc_td
                self.Hc_td.compile()
                self.Hc_rhs = Hc_td.compiled_qobjevo.mul_vec
                try:
                    self.h_tflag = 3
                    # Can H be made into an QobjEvo?
                    self.rhs = _tdrhs_with_state
                    H_td = QobjEvo(H, args, tlist, raw_str=True)
                    H_td *= -1j
                    self.H_td = H_td
                    self.H_td.compile()
                    self.h_func = H_td.with_state
                except:
                    self.h_tflag = 4
                    self.h_func = H
                    self.h_func_args = args
                    self.rhs = _funcrhs_with_state

            self.H_is_set = True
            if _safe_mode:
                try:
                    for c_op in self.td_c_ops:
                        c_op(0.)
                except:
                    raise Exception("Error calculating c_ops ")

                try:
                    for c_op in self.td_c_ops:
                        c_op.rhs(0, self.psi0)
                except:
                    raise Exception("c_ops are not consistant with psi0")

                try:
                    if self.H_td and self.h_tflag != 3:
                        self.H_td(0.)
                except:
                    raise Exception("Error calculating H")

                try:
                    if self.h_tflag == 1:
                        self.rhs(0, self.psi0)
                    else:
                        self.rhs(0, self.psi0, config)
                except:
                    raise Exception("H is not consistant with psi0")

            ss = SolverSystem()
            ss.h_tflag = self.h_tflag
            ss.td_c_ops = self.td_c_ops
            ss.td_n_ops = self.td_n_ops
            ss.H_td = self.H_td
            ss.Hc_td = self.Hc_td
            ss.h_func = self.h_func
            ss.h_func_args = self.h_func_args
            solver_safe["mcsolve"] = ss

        else:
            # REUSING equation system
            if isinstance(H, SolverSystem):
                sys = H
            else:
                sys = solver_safe["mcsolve"]

            self.h_tflag = sys.h_tflag
            self.td_c_ops = sys.td_c_ops
            self.td_n_ops = sys.td_n_ops
            self.H_td = sys.H_td
            self.Hc_td = sys.Hc_td
            self.h_func = sys.h_func
            self.h_func_args = [args if args else sys.h_func_args]

            if args:
                [c_op.arguments(args) for c_op in self.td_c_ops]
                [c_op.arguments(args) for c_op in self.td_n_ops]
                if self.H_td is not None:
                    self.H_td.arguments(args)

            if self.h_tflag == 1:
                self.rhs = self.H_td.compiled_qobjevo.mul_vec

            elif self.h_tflag == 2:
                self.Hc_rhs = self.Hc_td.compiled_qobjevo.mul_vec
                self.rhs = _funcrhs

            elif self.h_tflag == 3:
                self.Hc_rhs = self.Hc_td.compiled_qobjevo.mul_vec
                self.h_func = self.H_td.with_state
                self.rhs = _tdrhs_with_state

            elif self.h_tflag == 4:
                self.Hc_rhs = self.Hc_td.compiled_qobjevo.mul_vec
                self.rhs = _funcrhs_with_state

        return self


class qutip_zvode(zvode):
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r


def mcsolve(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=None,
            args={}, options=Options(),
            progress_bar=True, map_func=parallel_map, map_kwargs={},
            _safe_mode=True):
    """Monte Carlo evolution of a state vector :math:`|\psi \\rangle` for a
    given Hamiltonian and sets of collapse operators, and possibly, operators
    for calculating expectation values. Options for the underlying ODE solver
    are given by the Options class.

    mcsolve supports time-dependent Hamiltonians and collapse operators using
    either Python functions of strings to represent time-dependent
    coefficients. Note that, the system Hamiltonian MUST have at least one
    constant term.

    As an example of a time-dependent problem, consider a Hamiltonian with two
    terms ``H0`` and ``H1``, where ``H1`` is time-dependent with coefficient
    ``sin(w*t)``, and collapse operators ``C0`` and ``C1``, where ``C1`` is
    time-dependent with coeffcient ``exp(-a*t)``.  Here, w and a are constant
    arguments with values ``W`` and ``A``.

    Using the Python function time-dependent format requires two Python
    functions, one for each collapse coefficient. Therefore, this problem could
    be expressed as::

        def H1_coeff(t,args):
            return sin(args['w']*t)

        def C1_coeff(t,args):
            return exp(-args['a']*t)

        H = [H0, [H1, H1_coeff]]

        c_ops = [C0, [C1, C1_coeff]]

        args={'a': A, 'w': W}

    or in String (Cython) format we could write::

        H = [H0, [H1, 'sin(w*t)']]

        c_ops = [C0, [C1, 'exp(-a*t)']]

        args={'a': A, 'w': W}

    Constant terms are preferably placed first in the Hamiltonian and collapse
    operator lists.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, ``list``
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector

    tlist : array_like
        Times at which results are recorded.

    ntraj : int
        Number of trajectories to run.

    c_ops : :class:`qutip.Qobj`, ``list``
        single collapse operator or a ``list`` of collapse operators.

    e_ops : :class:`qutip.Qobj`, ``list``
        single operator as Qobj or ``list`` or equivalent of Qobj operators
        for calculating expectation values.

    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : Options
        Instance of ODE solver options.

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation. Set to None to disable the
        progress bar.

    map_func: function
        A map function for managing the calls to the single-trajactory solver.

    map_kwargs: dictionary
        Optional keyword arguments to the map_func function.

    Returns
    -------
    results : :class:`qutip.solver.Result`
        Object storing all results from the simulation.

    .. note::

        It is possible to reuse the random number seeds from a previous run
        of the mcsolver by passing the output Result object seeds via the
        Options class, i.e. Options(seeds=prev_result.seeds).
    """
    if isinstance(c_ops, (Qobj, QobjEvo)):
        c_ops = [c_ops]

    if isinstance(H, SolverSystem):
        options.rhs_reuse = True

    if len(c_ops) == 0 and not options.rhs_reuse:
        warn("No c_ops, using sesolve")
        return sesolve(H, psi0, tlist, e_ops=e_ops, args=args,
                       options=options, progress_bar=progress_bar,
                       _safe_mode=_safe_mode)

    e_ops_dict = None
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]
    elif isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]

    config = SolverConfiguration(H, psi0, tlist, c_ops, e_ops, ntraj,
                                 args, options, progress_bar, map_func,
                                 map_kwargs, _safe_mode)

    # load monte carlo class
    mc = _MC(config)
    # Run the simulation
    mc.run()

    # AFTER MCSOLVER IS DONE
    # ----------------------

    # Store results in the Result object
    output = Result()
    output.solver = 'mcsolve'
    output.seeds = config.options.seeds

    # state vectors
    """if (mc.psi_out is not None and
            config.options.average_states and
            config.ntraj != 1):


    elif mc.psi_out is not None:
        output.states = mc.psi_out"""

    if mc.psi_out is not None:
        if config.options.steady_state_average:
            output.states = Qobj(np.mean(mc.psi_out, axis=0),
                                 [config.psi0_dims[0], config.psi0_dims[0]],
                                 [config.psi0_shape[0], config.psi0_shape[0]])

        elif config.options.average_states:
            psi_dm = np.empty((config.ntraj, len(config.tlist)), dtype=object)
            for i in range(config.ntraj):
                for j in range(len(config.tlist)):
                    tmp = Qobj(mc.psi_out[i,j,:].reshape(-1,1))
                    tmp = tmp*tmp.dag()
                    tmp.dims = [config.psi0_dims[0], config.psi0_dims[0]]
                    psi_dm[i,j] = tmp
                output.states = parfor(_mc_dm_avg, psi_dm.T)

        else:
            psi = np.empty((config.ntraj, len(config.tlist)), dtype=object)
            for i in range(config.ntraj):
                for j in range(len(config.tlist)):
                    tmp = Qobj(mc.psi_out[i,j,:].reshape(-1,1),
                               dims = config.psi0_dims[0])
                    psi[i,j] = tmp
            output.states = psi

    # expectation values
    if mc.expect_out is not None:
        _one_expects = np.empty((config.ntraj, len(config.tlist)), dtype=object)
        expect = [None]*config.e_num
        for i in range(config.e_num):
            if config.e_ops_isherm[i]:
                expect[i] = np.real(mc.expect_out[:,:,i])
            else:
                expect[i] = mc.expect_out[:,:,i]

            if config.options.average_expect:
                if isinstance(ntraj, int):
                    expect[i] = np.mean(expect[i], axis=0)

                else:
                    expt_data = []
                    for num in ntraj:
                        expt_data.append(np.mean(expect[i][:num,:], axis=0))
                    expect[i] = np.stack(expt_data)
        output.expect = expect

    # simulation parameters
    output.times = config.tlist
    output.num_expect = config.e_num
    output.num_collapse = config.c_num
    output.ntraj = config.ntraj
    output.col_times = mc.collapse_times_out
    output.col_which = mc.which_op_out

    if e_ops_dict:
        output.expect = {e: output.expect[n]
                         for n, e in enumerate(e_ops_dict.keys())}

    return output


# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class _MC():
    """
    Private class for solving Monte Carlo evolution from mcsolve
    """

    def __init__(self, config):
        self.config = config
        # set output variables, even if they are not used to simplify output
        # code.
        self.psi_out = None
        self.expect_out = None
        if config.options.store_states:
            self.psi_out = [None] * config.ntraj
        if config.e_num > 0:
            self.expect_out = [None] * config.ntraj
        self.collapse_times_out = np.zeros(config.ntraj, dtype=np.ndarray)
        self.which_op_out = np.zeros(config.ntraj, dtype=np.ndarray)

        # setup seeds array
        if self.config.options.seeds is None:
            step = 4294967295 // config.ntraj
            self.config.options.seeds = \
                randint(0, step-1, size=config.ntraj) + \
                np.arange(config.ntraj) * step
        else:
            # if ntraj was reduced but reusing seeds
            seed_length = len(config.options.seeds)
            if seed_length > config.ntraj:
                self.config.options.seeds = \
                    config.options.seeds[0:config.ntraj]
            # if ntraj was increased but reusing seeds
            elif seed_length < config.ntraj:
                len_new_seed = (config.ntraj - seed_length)
                step = 4294967295 // len_new_seed
                newseeds = randint(0, step - 1,
                                   size=(len_new_seed)) + \
                                   np.arange(len_new_seed) * step
                self.config.options.seeds = np.hstack(
                    (config.options.seeds, newseeds))

    def run(self):
        if debug:
            print(inspect.stack()[0][3])

        # set arguments for input to monte carlo
        map_kwargs = {'progress_bar': self.config.progress_bar,
                      'num_cpus': self.config.options.num_cpus}
        map_kwargs.update(self.config.map_kwargs)
        task_args = (self.config.options.seeds,)
        task_kwargs = {}
        results = self.config.map_func(self.single_traj,
                                       list(range(self.config.ntraj)),
                                       task_args, task_kwargs,
                                       **map_kwargs)

        for n, result in enumerate(results):
            state_out, expect_out, collapse_times, which_oper = result

            if self.config.options.store_states:
                self.psi_out[n] = state_out

            if self.config.e_num > 0:
                self.expect_out[n] = expect_out

            self.collapse_times_out[n] = collapse_times
            self.which_op_out[n] = which_oper

        self.psi_out = np.stack(self.psi_out) # psi_out[traj, t, :] = psi
        self.expect_out = np.stack(self.expect_out) # expect_out[traj, t, e_ops]


    # -----------------------------------------------------------------------------
    # single-trajectory for monte carlo
    # -----------------------------------------------------------------------------
    def single_traj(self, nt, seeds):
        """
        Monte Carlo algorithm returning state-vector or expectation values
        at times tlist for a single trajectory.
        """

        # SEED AND RNG AND GENERATE
        prng = RandomState(seeds[nt])
        config = self.config

        if False:
            pass
            # (config.h_tflag in (1,) and config.options.method == "dopri5"):
            # dopri5 solver to add later
            # states_out, expect_out, collapse_times, which_oper = cy_mc_run_fast(
            #     config, prng)
        else:
            ODE = _build_integration_func(config)

            tlist = config.tlist
            # set initial conditions
            ODE.set_initial_value(config.psi0, tlist[0])

            states_out, expect_out, collapse_times, which_oper = cy_mc_run_ode(ODE,
                     config, prng)

        # Run at end of mc_alg function
        # -----------------------------
        if config.options.steady_state_average:
            states_out /= float(len(tlist))

        return (states_out, expect_out,
                np.array(collapse_times, dtype=float),
                np.array(which_oper, dtype=int))

# -----------------------------------------------------------------------------
# CODES FOR PYTHON FUNCTION BASED TIME-DEPENDENT RHS
# -----------------------------------------------------------------------------
def _build_integration_func(config):
    """
    Create the integration function while fixing the parameters
    """
    if debug:
        print(inspect.stack()[0][3] + " in " + str(os.getpid()))

    ODE = ode(config.rhs)
    if config.h_tflag in (2, 3):
        ODE.set_f_params(config)
    # initialize ODE solver for RHS
    ODE.set_integrator('zvode', method="adams")
    opt = config.options
    ODE._integrator = qutip_zvode(
        method=opt.method, order=opt.order, atol=opt.atol,
        rtol=opt.rtol, nsteps=opt.nsteps, first_step=opt.first_step,
        min_step=opt.min_step, max_step=opt.max_step)

    if not len(ODE._y):
        ODE.t = 0.0
        ODE._y = np.array([0.0], complex)
    ODE._integrator.reset(len(ODE._y), ODE.jac is not None)
    return ODE


# RHS of ODE for python function Hamiltonian
def _funcrhs(t, psi, config):
    h_func_data = -1.0j * config.h_func(t, config.h_func_args).data
    h_func_term = h_func_data.mul_vec(psi)
    return h_func_term + config.Hc_rhs(t, psi)

def _funcrhs_with_state(t, psi, config):
    h_func_data = - 1.0j * config.h_func(t, psi, config.h_func_args).data
    h_func_term = h_func_data.mul_vec(psi)
    return h_func_term + config.Hc_rhs(t, psi)

def _tdrhs_with_state(t, psi, config):
    h_func_data = config.h_func(t, psi, config.h_func_args, data=1)
    h_func_term = h_func_data.mul_vec(psi)
    return h_func_term + config.Hc_rhs(t, psi)


def _mc_dm_avg(psi_list):
    """
    Private function that averages density matrices in parallel
    over all trajectories for a single time using parfor.
    """
    ln = len(psi_list)
    dims = psi_list[0].dims
    shape = psi_list[0].shape
    out_data = sum([psi.data for psi in psi_list]) / ln
    return Qobj(out_data, dims=dims, shape=shape, fast='mc-dm')
