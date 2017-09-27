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

__all__ = ['mcsolve_3']

import os
import numpy as np
from numpy.random import RandomState, randint
from scipy.integrate import ode
import scipy.sparse as sp
from scipy.integrate._ode import zvode

from types import FunctionType, BuiltinFunctionType
from functools import partial

from qutip.qobj import Qobj
from qutip.td_qobj import td_Qobj
from qutip.parallel import parfor, parallel_map, serial_map
from qutip.cy.spmatfuncs import spmv
from qutip.cy.mcsolvetd import cy_mc_run_ode, cy_mc_run_fast

from qutip.solver import Options, Result, _solver_safety_check
#from qutip.interpolate import Cubic_Spline
from qutip.settings import debug
from qutip.ui.progressbar import TextProgressBar, BaseProgressBar
import qutip.settings

if debug:
    import inspect

#
# Internal, global variables for storing references to dynamically loaded
# cython functions

global config_global

class SolverConfiguration():
    def __init__(self):
        pass

    def reset(self):
        pass

    def soft_reset(self):
        pass

config = SolverConfiguration()


class qutip_zvode(zvode):
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r


def mcsolve_3(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=None,
            args={}, options=None, progress_bar=True,
            map_func=None, map_kwargs=None,
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
    H : :class:`qutip.Qobj`
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector

    tlist : array_like
        Times at which results are recorded.

    ntraj : int
        Number of trajectories to run.

    c_ops : array_like
        single collapse operator or ``list`` or ``array`` of collapse
        operators.

    e_ops : array_like
        single operator or ``list`` or ``array`` of operators for calculating
        expectation values.

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

    if len(c_ops) == 0:
        print("No c_ops, using sesolve")
        sesolve(H, psi0, tlist, e_ops=e_ops, args=args, options=options,
                progress_bar=progress_bar, _safe_mode=_safe_mode)

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]
    elif isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    config = _mc_make_config(H, psi0, tlist, c_ops, e_ops, ntraj,
            args, options, progress_bar, map_func, map_kwargs, _safe_mode)
    options = config.options

    # load monte carlo class
    mc = _MC(config)
    # Run the simulation
    mc.run()

    # Remove RHS cython file if necessary
    #if not options.rhs_reuse and config.tdname:
    #    _cython_build_cleanup(config.tdname)

    # AFTER MCSOLVER IS DONE
    # ----------------------

    # Store results in the Result object
    output = Result()
    output.solver = 'mcsolve'
    output.seeds = config.options.seeds
    # state vectors
    if (mc.psi_out is not None and config.options.average_states
            and config.cflag and ntraj != 1):
        output.states = parfor(_mc_dm_avg, mc.psi_out.T)
    elif mc.psi_out is not None:
        output.states = mc.psi_out

    # expectation values
    if (mc.expect_out is not None and config.cflag
            and config.options.average_expect):
        # averaging if multiple trajectories
        if isinstance(ntraj, int):
            output.expect = [np.mean(np.array([mc.expect_out[nt][op]
                                               for nt in range(ntraj)],
                            dtype=object), axis=0) for op in range(config.e_num)]
        elif isinstance(ntraj, (list, np.ndarray)):
            output.expect = []
            for num in ntraj:
                expt_data = np.mean(mc.expect_out[:num], axis=0)
                data_list = []
                if any([not op.isherm for op in e_ops]):
                    for k in range(len(e_ops)):
                        if e_ops[k].isherm:
                            data_list.append(np.real(expt_data[k]))
                        else:
                            data_list.append(expt_data[k])
                else:
                    data_list = [data for data in expt_data]
                output.expect.append(data_list)
    else:
        # no averaging for single trajectory or if average_expect flag
        # (Options) is off
        if mc.expect_out is not None:
            output.expect = mc.expect_out

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


def _mc_make_config(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=None, args={},
            options=None, progress_bar=True, map_func=None,
            map_kwargs=None, _safe_mode=True, compile=True):

    if debug:
        print(inspect.stack()[0][3])

    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if _safe_mode:
        _solver_safety_check(H, psi0, c_ops, e_ops, args)

    if options is None:
        options = Options()

    config.soft_reset()
    # set general items
    if ntraj is None:
        ntraj = options.ntraj
    config.tlist = np.asarray(tlist)
    if isinstance(ntraj, (list, np.ndarray)):
        config.ntraj = np.sort(ntraj)[-1]
    else:
        config.ntraj = ntraj

    config.map_func = map_func if map_func is not None else parallel_map
    config.map_kwargs = map_kwargs if map_kwargs is not None else {}

    if not psi0.isket:
        raise Exception("Initial state must be a state vector.")

    config.options = options

    if progress_bar:
        if progress_bar is True:
            config.progress_bar = TextProgressBar()
        else:
            config.progress_bar = progress_bar
    else:
        config.progress_bar = BaseProgressBar()

    # set num_cpus to the value given in qutip.settings if none in Options
    if not config.options.num_cpus:
        config.options.num_cpus = qutip.settings.num_cpus
        if config.options.num_cpus == 1:
            # fallback on serial_map if num_cpu == 1, since there is no
            # benefit of starting multiprocessing in this case
            config.map_func = serial_map

    # set initial value data
    if options.tidy:
        config.psi0 = psi0.tidyup(options.atol).full().ravel()
    else:
        config.psi0 = psi0.full().ravel()

    config.psi0_dims = psi0.dims
    config.psi0_shape = psi0.shape

    # set options on ouput states
    if config.options.steady_state_average:
        config.options.average_states = True

    # set norm finding constants
    config.norm_tol = options.norm_tol
    config.norm_steps = options.norm_steps

    # take care of expectation values, if any
    if any(e_ops):
        config.e_num = len(e_ops)
        config.e_ops_data = []
        config.e_ops_ind = []
        config.e_ops_ptr = []
        config.e_ops_isherm = []
        for op in e_ops:
            if isinstance(op, list):
                op = op[0]
            config.e_ops_data.append(op.data.data)
            config.e_ops_ind.append(op.data.indices)
            config.e_ops_ptr.append(op.data.indptr)
            config.e_ops_isherm.append(op.isherm)
        config.e_ops_data = np.array(config.e_ops_data)
        config.e_ops_ind = np.array(config.e_ops_ind)
        config.e_ops_ptr = np.array(config.e_ops_ptr)
        config.e_ops_isherm = np.array(config.e_ops_isherm)

    # SETUP ODE DATA IF NONE EXISTS OR NOT REUSING
    # --------------------------------------------
    dopri = (options.method == "dopri5")
    if not options.rhs_reuse or not config.tdfunc:
        # reset config collapse and time-dependence flags to default values


        # Find the type of time-dependence for H and c_ops
        # h_tflag
        #   0 : cte
        #   1 : td that can be compiled to cython (str, numpy)
        #   2 : td that cannot be compiled (func)
        #   3 : H is a functions
        #   4 : H functions with state
        #
        # c_tflag
        #   0 : cte
        #   1 : td that can be compiled to cython (str, numpy)
        #   2 : td that cannot be compiled (func)

        # define :
        # config.rhs : the function called by the integrator
        # config.td_c_ops : to get needed spmv function
        # config.td_n_ops : to get the expectation value of the c_op
        # config.rhs_ptr : pointer to the H rhs function

        config.c_num = len(c_ops)
        if len(c_ops) > 0:
            config.cflag = True
        else:
            config.cflag = False
            raise Exception("Should use sesolve instead of mcsolve")

        c_types = []
        config.td_c_ops = []
        config.td_n_ops = []
        for c in c_ops:
            config.td_c_ops += [td_Qobj(c, args, tlist)]
            config.td_n_ops += [config.td_c_ops[-1].norm()]
            if config.td_c_ops[-1].cte:
                c_types += [0]
            else:
                c_types += [1]
        config.c_tflag = max(c_types)
        [c_op.compile() for c_op in config.td_c_ops]
        [c_op.compile() for c_op in config.td_n_ops]
        #if config.c_tflag in (0,1):
            #Compile str, fast cte/numpy
        #    [c_ops.get_rhs_func() for c_ops in config.td_c_ops]
        #    [c_ops.get_rhs_func() for c_ops in config.td_n_ops]
        #elif config.c_tflag == 2:
        #    for c_ops in config.td_c_ops:
        #         c_ops.fast = False
        #    for c_ops in config.td_n_ops:
        #         c_ops.fast = False

        Hc_td = td_Qobj(config.td_c_ops[0](0)*0., args, tlist)
        for c in config.td_n_ops:
            Hc_td += -0.5 * c

        config.h_func_args = {}
        if not options.rhs_with_state:
            if not isinstance(H, (FunctionType, BuiltinFunctionType, partial)):
                H_td = td_Qobj(H, args, tlist)
                H_td *= -1j
                H_td += Hc_td
                if options.tidy:
                    # Compiling tidyup anyway
                    H_td.tidyup(options.atol)
                config.H_td = H_td
                config.H_td.compile()
                config.h_tflag = 1
                config.rhs = H_td.get_rhs_func()

                #if H_td.fast:
                #    config.h_tflag = 1
                #    config.rhs = H_td.get_rhs_func()
                #    if dopri:
                #        config.rhs_ptr = H_td._get_rhs_ptr()
                #else:
                #    config.h_tflag = 2
                #    config.rhs = H_td.get_rhs_func()
            else:
                config.h_tflag = 3
                config.h_func = H
                config.h_func_args = args
                #Hc_td *= -1j
                config.Hc_td = Hc_td
                config.Hc_td.compile()
                config.Hc_rhs = Hc_td.get_rhs_func()
                config.rhs = _tdrhs

        else:
            config.h_tflag = 4
            #Hc_td *= -1j
            config.Hc_td = Hc_td
            config.Hc_td.compile()
            config.Hc_rhs = Hc_td.get_rhs_func()
            config.rhs = _tdrhs_with_state
            if isinstance(H, (FunctionType, BuiltinFunctionType, partial)):
                config.h_func = H
                config.h_func_args = args
            else:
                H_td = td_Qobj(H, args, tlist)
                config.H_td = H_td
                config.H_td.compile()
                config.h_func = H_td.with_args # compiled



    else:
        # setup args for new parameters when rhs_reuse=True and tdfunc is given
        # string based
        raise NotImplementedError("How would this be used?")
    #if config.h_tflag >= 2 or config.c_tflag ==2:
    #    config.options.method = "adams"
    return config

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
        self.expect_out = []
        self.collapse_times_out = None
        self.which_op_out = None

        # FOR EVOLUTION WITH COLLAPSE OPERATORS
        if config.cflag:
            # preallocate ntraj arrays for state vectors,
            # collapse times, and which operator
            self.collapse_times_out = np.zeros(config.ntraj, dtype=np.ndarray)
            self.which_op_out = np.zeros(config.ntraj, dtype=np.ndarray)
            if config.e_num == 0 or config.options.store_states:
                self.psi_out = [None] * config.ntraj
            if config.e_num > 0:
                self.expect_out = [None] * config.ntraj

            # setup seeds array
            if self.config.options.seeds is None:
                step = 4294967295 // config.ntraj
                self.config.options.seeds = \
                    randint(0, step -1, size=config.ntraj) + \
                    np.arange(config.ntraj) * step
            else:
                # if ntraj was reduced but reusing seeds
                seed_length = len(config.options.seeds)
                if seed_length > config.ntraj:
                    self.config.options.seeds = \
                        config.options.seeds[0:config.ntraj]
                # if ntraj was increased but reusing seeds
                elif seed_length < config.ntraj:
                    step = 4294967295 // (config.ntraj - seed_length)
                    newseeds = randint(0, step -1,
                                       size = (config.ntraj - seed_length)) + \
                                       np.arange(config.ntraj) * step
                    self.config.options.seeds = np.hstack(
                        (config.options.seeds, newseeds))
        else:
            raise Exception("mcsolve without collapse!")

    def run(self):
        global config_global
        if debug:
            print(inspect.stack()[0][3])

        if not self.config.cflag:
            raise Exception("Should use sesolve")
            self.config.ntraj = 1
            if self.config.e_num == 0 or self.config.options.store_states:
                self.expect_out, self.psi_out = \
                    _evolve_no_collapse_psi_out(self.config)
            else:
                self.expect_out = _evolve_no_collapse_expect_out(self.config)

        else:
            # set arguments for input to monte carlo
            #print(config.rhs)
            map_kwargs = {'progress_bar': self.config.progress_bar,
                          'num_cpus': self.config.options.num_cpus}
            map_kwargs.update(self.config.map_kwargs)
            #task_args = (self.config, self.config.options,
            #             self.config.options.seeds)
            task_args = (self.config.options.seeds,)
            task_kwargs = {}
            config_global = self.config
            print( config_global.h_tflag in (1,) and config_global.options.method == "dopri5" )
            results = config.map_func(_mc_alg_evolve,
                                      list(range(config.ntraj)),
                                      task_args, task_kwargs,
                                      **map_kwargs)
            config_global = None
            for n, result in enumerate(results):
                state_out, expect_out, collapse_times, which_oper = result

                if self.config.e_num == 0 or self.config.options.store_states:
                    self.psi_out[n] = state_out

                if self.config.e_num > 0:
                    self.expect_out[n] = expect_out

                self.collapse_times_out[n] = collapse_times
                self.which_op_out[n] = which_oper

            self.psi_out = np.asarray(self.psi_out, dtype=object)


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
    if config.h_tflag in (3,4):
        ODE.set_f_params(config)
    #else:
        #ODE.set_f_params(None)
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
def _tdrhs(t, psi, config):
    h_func_data = -1.0j * config.h_func(t, config.h_func_args).data
    h_func_term = spmv(h_func_data, psi)
    return h_func_term + config.Hc_rhs(t, psi)


def _tdrhs_with_state(t, psi, config):
    h_func_data = - 1.0j * config.h_func(t, psi, config.h_func_args).data
    h_func_term = spmv(h_func_data, psi)
    return h_func_term + config.Hc_rhs(t, psi)


# -----------------------------------------------------------------------------
# single-trajectory for monte carlo
# -----------------------------------------------------------------------------
#def _mc_alg_evolve(nt, config, opt, seeds):
def _mc_alg_evolve(nt, seeds):
    global config_global
    config = config_global
    opt = config_global.options
    """
    Monte Carlo algorithm returning state-vector or expectation values
    at times tlist for a single trajectory.
    """

    # SEED AND RNG AND GENERATE
    prng = RandomState(seeds[nt])

    #print(config.h_tflag, config.options.method)
    #print(config.H_td, config.H_td.compiled_Qobj)
    #print(config.rhs)
    #print(config.rhs is config.H_td.compiled_Qobj.rhs)

    if ( config.h_tflag in (1,) and config.options.method == "dopri5" ):
        states_out, expect_out, collapse_times, which_oper = cy_mc_run_fast(
            config, prng)
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
        states_out = np.array([Qobj(states_out[0] / float(len(tlist)),
                              [config.psi0_dims[0], config.psi0_dims[0]],
                              [config.psi0_shape[0], config.psi0_shape[0]],
                              fast='mc-dm')])

    return (states_out, expect_out,
            np.array(collapse_times, dtype=float),
            np.array(which_oper, dtype=int))


def _mc_dm_avg(psi_list):
    """
    Private function that averages density matrices in parallel
    over all trajectories for a single time using parfor.
    """
    ln = len(psi_list)
    dims = psi_list[0].dims
    shape = psi_list[0].shape
    out_data = np.sum([psi.data for psi in psi_list]) / ln
    return Qobj(out_data, dims=dims, shape=shape, fast='mc-dm')
