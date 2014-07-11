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

import sys
import os
import time
import copy
import numpy as np
from types import FunctionType
from multiprocessing import Pool, cpu_count

from numpy.random import RandomState, random_integers
from numpy import arange, array, cumsum, mean, ndarray, setdiff1d, sort, zeros

from scipy.integrate import ode
from scipy.linalg import norm
import scipy.sparse as sp
from scipy.linalg.blas import get_blas_funcs
dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)

from qutip.qobj import *
from qutip.expect import *
from qutip.states import ket2dm
from qutip.parfor import parfor
from qutip.cy.spmatfuncs import cy_ode_rhs, cy_expect_psi_csr, spmv, spmv_csr
from qutip.cy.codegen import Codegen
from qutip.solver import Options, Result, config
from qutip.rhs_generate import _td_format_check, _td_wrap_array_str
import qutip.settings
from qutip.settings import debug
from qutip.ui.progressbar import TextProgressBar

if debug:
    import inspect

#
# Internal, global variables for storing references to dynamically loaded
# cython functions
#
_cy_col_spmv_func = None
_cy_col_expect_func = None
_cy_col_spmv_call_func = None
_cy_col_expect_call_func = None
_cy_rhs_func = None


def mcsolve(H, psi0, tlist, c_ops, e_ops, ntraj=None,
            args={}, options=Options()):
    """Monte-Carlo evolution of a state vector :math:`|\psi \\rangle` for a
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

        H=[H0,[H1,H1_coeff]]

        c_op_list=[C0,[C1,C1_coeff]]

        args={'a':A,'w':W}

    or in String (Cython) format we could write::

        H=[H0,[H1,'sin(w*t)']]

        c_op_list=[C0,[C1,'exp(-a*t)']]

        args={'a':A,'w':W}

    Constant terms are preferably placed first in the Hamiltonian and collapse
    operator lists.

    Parameters
    ----------
    H : qobj
        System Hamiltonian.
    psi0 : qobj
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

    Returns
    -------
    results : Result
        Object storing all results from simulation.

    """

    if debug:
        print(inspect.stack()[0][3])

    if ntraj is None:
        ntraj = options.ntraj

    if not psi0.isket:
        raise Exception("Initial state must be a state vector.")

    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    config.options = options
    if isinstance(ntraj, list):
        config.progress_bar = TextProgressBar(max(ntraj))
    else:
        config.progress_bar = TextProgressBar(ntraj)

    # set num_cpus to the value given in qutip.settings if none in Options
    if not config.options.num_cpus:
        config.options.num_cpus = qutip.settings.num_cpus

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

    # set general items
    config.tlist = tlist
    if isinstance(ntraj, (list, ndarray)):
        config.ntraj = sort(ntraj)[-1]
    else:
        config.ntraj = ntraj

    # set norm finding constants
    config.norm_tol = options.norm_tol
    config.norm_steps = options.norm_steps

    # convert array based time-dependence to string format
    H, c_ops, args = _td_wrap_array_str(H, c_ops, args, tlist)

    # ----------------------------------------------
    # SETUP ODE DATA IF NONE EXISTS OR NOT REUSING
    # ----------------------------------------------
    if (not options.rhs_reuse) or (not config.tdfunc):
        # reset config collapse and time-dependence flags to default values
        config.soft_reset()

        # check for type of time-dependence (if any)
        time_type, h_stuff, c_stuff = _td_format_check(H, c_ops, 'mc')
        h_terms = len(h_stuff[0]) + len(h_stuff[1]) + len(h_stuff[2])
        c_terms = len(c_stuff[0]) + len(c_stuff[1]) + len(c_stuff[2])
        # set time_type for use in multiprocessing
        config.tflag = time_type

        # check for collapse operators
        if c_terms > 0:
            config.cflag = 1
        else:
            config.cflag = 0

        # Configure data
        _mc_data_config(H, psi0, h_stuff, c_ops, c_stuff, args, e_ops,
                        options, config)

        # compile and load cython functions if necessary
        _mc_func_load(config)

    else:
        # setup args for new parameters when rhs_reuse=True and tdfunc is given
        # string based
        if config.tflag in array([1, 10, 11]):
            if any(args):
                config.c_args = []
                arg_items = args.items()
                for k in range(len(args)):
                    config.c_args.append(arg_items[k][1])
        # function based
        elif config.tflag in array([2, 3, 20, 22]):
            config.h_func_args = args

    # load monte-carlo class
    mc = _MC_class(config)

    # RUN THE SIMULATION
    mc.run()

    # remove RHS cython file if necessary
    if not options.rhs_reuse and config.tdname:
        try:
            os.remove(config.tdname + ".pyx")
        except:
            pass

    # AFTER MCSOLVER IS DONE --------------------------------------
    # ------- COLLECT AND RETURN OUTPUT DATA IN ODEDATA OBJECT --------------
    output = Result()
    output.solver = 'mcsolve'
    # state vectors
    if (mc.psi_out is not None and config.options.average_states
            and config.cflag and ntraj != 1):
        output.states = parfor(_mc_dm_avg, mc.psi_out.T)
    elif mc.psi_out is not None:
        output.states = mc.psi_out
    # expectation values
    elif (mc.expect_out is not None and config.cflag
            and config.options.average_expect):
        # averaging if multiple trajectories
        if isinstance(ntraj, int):
            output.expect = [mean([mc.expect_out[nt][op]
                                   for nt in range(ntraj)], axis=0)
                             for op in range(config.e_num)]
        elif isinstance(ntraj, (list, ndarray)):
            output.expect = []
            for num in ntraj:
                expt_data = mean(mc.expect_out[:num], axis=0)
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
        # no averaging for single trajectory or if average_states flag
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


# --------------------------------------------------------------
# MONTE-CARLO CLASS
# --------------------------------------------------------------
class _MC_class():
    """
    Private class for solving Monte-Carlo evolution from mcsolve
    """

    def __init__(self, config):

        # -----------------------------------
        # INIT MC CLASS
        # -----------------------------------

        self.config = config

        # ----MAIN OBJECT PROPERTIES--------------------
        # holds instance of the ProgressBar class
        self.bar = None
        # holds instance of the Pthread class
        self.thread = None
        # Number of completed trajectories
        self.count = 0
        # step-size for count attribute
        self.step = 1
        # Percent of trajectories completed
        self.percent = 0.0
        # used in implimenting the command line progress ouput
        self.level = 0.1
        # times at which to output state vectors or expectation values
        # number of time steps in tlist
        self.num_times = len(self.config.tlist)
        # holds seed for random number generator
        self.seeds = self.config.options.seeds
        # number of cpus to be used
        self.cpus = self.config.options.num_cpus
        # set output variables, even if they are not used to simplify output
        # code.
        self.psi_out = None
        self.expect_out = []
        self.collapse_times_out = None
        self.which_op_out = None

        # FOR EVOLUTION FOR NO COLLAPSE OPERATORS
        if config.c_num == 0:
            if config.e_num == 0:
                # Output array of state vectors calculated at times in tlist
                self.psi_out = array([Qobj()] * self.num_times)
            elif config.e_num != 0:  # no collapse expectation values
                # List of output expectation values calculated at times in
                # tlist
                self.expect_out = []
                for i in range(config.e_num):
                    if config.e_ops_isherm[i]:
                        # preallocate real array of zeros
                        self.expect_out.append(zeros(self.num_times))
                    else:  # preallocate complex array of zeros
                        self.expect_out.append(
                            zeros(self.num_times, dtype=complex))
                    self.expect_out[i][0] = cy_expect_psi_csr(
                        config.e_ops_data[i], config.e_ops_ind[i],
                        config.e_ops_ptr[i], config.psi0,
                        config.e_ops_isherm[i])

        # FOR EVOLUTION WITH COLLAPSE OPERATORS
        elif config.c_num != 0:
            # preallocate #ntraj arrays for state vectors, collapse times, and
            # which operator
            self.collapse_times_out = zeros((config.ntraj), dtype=ndarray)
            self.which_op_out = zeros((config.ntraj), dtype=ndarray)
            if config.e_num == 0:
                # if no expectation operators, preallocate #ntraj arrays
                # for state vectors
                if self.config.options.steady_state_average:
                    self.psi_out = array([zeros((1), dtype=object)
                                          for q in range(config.ntraj)])
                else:
                    self.psi_out = array([zeros((self.num_times), dtype=object)
                                          for q in range(config.ntraj)])

            else:  # preallocate array of lists for expectation values
                self.expect_out = [[] for x in range(config.ntraj)]

    # -------------------------------------------------
    # CLASS METHODS
    # -------------------------------------------------
    def callback(self, results):
        r = results[0]
        if self.config.e_num == 0:  # output state-vector
            self.psi_out[r] = results[1]
        else:  # output expectation values
            self.expect_out[r] = results[1]

        self.collapse_times_out[r] = results[2]
        self.which_op_out[r] = results[3]
        self.count += self.step

        if self.config.ntraj != 1:
            self.config.progress_bar.update(self.count)

    def serial(self, args, top):

        if debug:
            print(inspect.stack()[0][3])

        for nt in range(self.config.ntraj):
            top.callback(_mc_alg_evolve(nt, args, self.config))

    def parallel(self, args, top):

        if debug:
            print(inspect.stack()[0][3])

        if self.cpus == 1 or self.config.ntraj == 1:
            self.serial(args, top)
            return

        pl = Pool(processes=self.cpus)
        for nt in range(self.config.ntraj):
            pl.apply_async(_mc_alg_evolve, args=(nt, args, self.config),
                           callback=top.callback)
        pl.close()

        try:
            pl.join()
        except KeyboardInterrupt:
            print("Cancel all MC threads on keyboard interrupt")
            pl.terminate()
            pl.join()
        return

    def run(self):

        if debug:
            print(inspect.stack()[0][3])

        if self.config.c_num == 0:
            if self.config.ntraj != 1:
                # ntraj != 1 is pointless for no collapse operators
                self.config.ntraj = 1
            if self.config.e_num == 0:  # return psi at each requested time
                self.psi_out = _no_collapse_psi_out(
                    self.num_times, self.psi_out, self.config)
            else:  # return expectation values of requested operators
                self.expect_out = _no_collapse_expect_out(
                    self.num_times, self.expect_out, self.config)

        elif self.config.c_num != 0:

            if self.seeds is None:
                self.seeds = random_integers(1e8, size=self.config.ntraj)
            # else:
            #    if len(self.seeds) != self.config.ntraj:
            #        raise "Incompatible size of seeds vector in config."

            if self.config.e_num == 0:
                if config.options.steady_state_average:
                    mc_alg_out = zeros((1), dtype=object)
                else:
                    mc_alg_out = zeros((self.num_times), dtype=object)
                temp = sp.csr_matrix(
                    np.reshape(self.config.psi0,
                               (self.config.psi0.shape[0], 1)),
                    dtype=complex)
                if (self.config.options.average_states and
                        not config.options.steady_state_average):
                    # output is averaged states, so use dm
                    mc_alg_out[0] = Qobj(temp*temp.conj().transpose(),
                                         [config.psi0_dims[0],
                                          config.psi0_dims[0]],
                                         [config.psi0_shape[0],
                                          config.psi0_shape[0]],
                                         fast='mc-dm')
                elif (not self.config.options.average_states and
                      not config.options.steady_state_average):
                    # output is not averaged, so write state vectors
                    mc_alg_out[0] = Qobj(temp, config.psi0_dims,
                                         config.psi0_shape, fast='mc')
                elif config.options.steady_state_average:
                    mc_alg_out[0] = temp * temp.conj().transpose()

            else:
                # PRE-GENERATE LIST OF EXPECTATION VALUES
                mc_alg_out = []
                for i in range(self.config.e_num):
                    if self.config.e_ops_isherm[i]:
                        # preallocate real array of zeros
                        mc_alg_out.append(zeros(self.num_times))
                    else:
                        # preallocate complex array of zeros
                        mc_alg_out.append(zeros(self.num_times, dtype=complex))

                    mc_alg_out[i][0] = \
                        cy_expect_psi_csr(self.config.e_ops_data[i],
                                          self.config.e_ops_ind[i],
                                          self.config.e_ops_ptr[i],
                                          self.config.psi0,
                                          self.config.e_ops_isherm[i])

            # set arguments for input to monte-carlo
            args = (mc_alg_out, self.config.options,
                    self.config.tlist, self.num_times, self.seeds)

            self.config.progress_bar.start(self.config.ntraj)
            self.parallel(args, self)
            self.config.progress_bar.finished()


# ----------------------------------------------------
# CODES FOR PYTHON FUNCTION BASED TIME-DEPENDENT RHS
# ----------------------------------------------------

# RHS of ODE for time-dependent systems with no collapse operators
def _tdRHS(t, psi, config):
    h_data = config.h_func(t, config.h_func_args).data
    return spmv(h_data, psi)


# RHS of ODE for constant Hamiltonian and at least one function based
# collapse operator
def _cRHStd(t, psi, config):
    sys = cy_ode_rhs(t, psi, config.h_data,
                     config.h_ind, config.h_ptr)
    col = array([np.abs(config.c_funcs[j](t, config.c_func_args)) ** 2 *
                 spmv_csr(config.n_ops_data[j],
                          config.n_ops_ind[j],
                          config.n_ops_ptr[j], psi)
                for j in config.c_td_inds])
    return sys - 0.5 * np.sum(col, 0)


# RHS of ODE for list-function based Hamiltonian
def _tdRHStd(t, psi, config):
    const_term = spmv_csr(config.h_data,
                          config.h_ind,
                          config.h_ptr, psi)
    h_func_term = array([config.h_funcs[j](t, config.h_func_args) *
                         spmv_csr(config.h_td_data[j],
                                  config.h_td_ind[j],
                                  config.h_td_ptr[j], psi)
                         for j in config.h_td_inds])
    col_func_terms = array([np.abs(
        config.c_funcs[j](t, config.c_func_args)) ** 2 *
        spmv_csr(config.n_ops_data[j],
                 config.n_ops_ind[j],
                 config.n_ops_ptr[j],
                 psi)
        for j in config.c_td_inds])
    return (const_term - np.sum(h_func_term, 0)
            - 0.5 * np.sum(col_func_terms, 0))


def _tdRHStd_with_state(t, psi, config):

    const_term = spmv_csr(config.h_data,
                          config.h_ind,
                          config.h_ptr, psi)

    h_func_term = array([
        config.h_funcs[j](t, psi, config.h_func_args) *
        spmv_csr(config.h_td_data[j],
                 config.h_td_ind[j],
                 config.h_td_ptr[j], psi)
        for j in config.h_td_inds])

    col_func_terms = array([
        np.abs(config.c_funcs[j](t, config.c_func_args)) ** 2 *
        spmv_csr(config.n_ops_data[j],
                 config.n_ops_ind[j],
                 config.n_ops_ptr[j], psi)
        for j in config.c_td_inds])

    return (const_term - np.sum(h_func_term, 0)
            - 0.5 * np.sum(col_func_terms, 0))


# RHS of ODE for python function Hamiltonian
def _pyRHSc(t, psi, config):
    h_func_data = - 1.0j * config.h_funcs(t, config.h_func_args)
    h_func_term = spmv(h_func_data, psi)
    const_col_term = 0
    if len(config.c_const_inds) > 0:
        const_col_term = spmv_csr(config.h_data, config.h_ind,
                                  config.h_ptr, psi)

    return h_func_term + const_col_term


def _pyRHSc_with_state(t, psi, config):
    h_func_data = - 1.0j * config.h_funcs(t, psi, config.h_func_args)
    h_func_term = spmv(h_func_data, psi)
    const_col_term = 0
    if len(config.c_const_inds) > 0:
        const_col_term = spmv_csr(config.h_data, config.h_ind,
                                  config.h_ptr, psi)

    return h_func_term + const_col_term


# ----------------------------------------------------
# END PYTHON FUNCTION RHS
# ----------------------------------------------------

# --- return psi at requested times for no collapse operators
def _no_collapse_psi_out(num_times, psi_out, config):
    """
    Calculates state vectors at times tlist if no collapse AND no
    expectation values are given.
    """

    global _cy_rhs_func
    global _cy_col_spmv_func, _cy_col_expect_func
    global _cy_col_spmv_call_func, _cy_col_expect_call_func

    if debug:
        print(inspect.stack()[0][3])

    if not _cy_rhs_func:
        _mc_func_load(config)

    opt = config.options
    if config.tflag in array([1, 10, 11]):
        ODE = ode(_cy_rhs_func)
        code = compile('ODE.set_f_params(' + config.string + ')',
                       '<string>', 'exec')
        exec(code)
    elif config.tflag == 2:
        ODE = ode(_cRHStd)
        ODE.set_f_params(config)
    elif config.tflag in array([20, 22]):
        if config.options.rhs_with_state:
            ODE = ode(_tdRHStd_with_state)
        else:
            ODE = ode(_tdRHStd)
        ODE.set_f_params(config)
    elif config.tflag == 3:
        if config.options.rhs_with_state:
            ODE = ode(_pyRHSc_with_state)
        else:
            ODE = ode(_pyRHSc)
        ODE.set_f_params(config)
    else:
        ODE = ode(cy_ode_rhs)
        ODE.set_f_params(config.h_data, config.h_ind, config.h_ptr)

    # initialize ODE solver for RHS
    ODE.set_integrator('zvode', method=opt.method, order=opt.order,
                       atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                       first_step=opt.first_step, min_step=opt.min_step,
                       max_step=opt.max_step)
    # set initial conditions
    ODE.set_initial_value(config.psi0, config.tlist[0])
    psi_out[0] = Qobj(config.psi0, config.psi0_dims,
                      config.psi0_shape)
    for k in range(1, num_times):
        ODE.integrate(config.tlist[k], step=0)  # integrate up to tlist[k]
        if ODE.successful():
            psi_out[k] = Qobj(ODE.y / dznrm2(ODE.y), config.psi0_dims,
                              config.psi0_shape)
        else:
            raise ValueError('Error in ODE solver')
    return psi_out


# ---- return expectation values at requested times for no collapse oper
def _no_collapse_expect_out(num_times, expect_out, config):
    """
    Calculates expect.values at times tlist if no collapse ops. given
    """

    global _cy_rhs_func
    global _cy_col_spmv_func, _cy_col_expect_func
    global _cy_col_spmv_call_func, _cy_col_expect_call_func

    if debug:
        print(inspect.stack()[0][3])

    if not _cy_rhs_func:
        _mc_func_load(config)

    opt = config.options
    if config.tflag in array([1, 10, 11]):
        ODE = ode(_cy_rhs_func)
        code = compile('ODE.set_f_params(' + config.string + ')',
                       '<string>', 'exec')
        exec(code)
    elif config.tflag == 2:
        ODE = ode(_cRHStd)
        ODE.set_f_params(config)
    elif config.tflag in array([20, 22]):
        if config.options.rhs_with_state:
            ODE = ode(_tdRHStd_with_state)
        else:
            ODE = ode(_tdRHStd)
        ODE.set_f_params(config)
    elif config.tflag == 3:
        if config.options.rhs_with_state:
            ODE = ode(_pyRHSc_with_state)
        else:
            ODE = ode(_pyRHSc)
        ODE.set_f_params(config)
    else:
        ODE = ode(cy_ode_rhs)
        ODE.set_f_params(config.h_data, config.h_ind, config.h_ptr)

    ODE.set_integrator('zvode', method=opt.method, order=opt.order,
                       atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                       first_step=opt.first_step, min_step=opt.min_step,
                       max_step=opt.max_step)
    ODE.set_initial_value(config.psi0, config.tlist[0])
    for jj in range(config.e_num):
        expect_out[jj][0] = cy_expect_psi_csr(
            config.e_ops_data[jj], config.e_ops_ind[jj],
            config.e_ops_ptr[jj], config.psi0,
            config.e_ops_isherm[jj])

    for k in range(1, num_times):
        ODE.integrate(config.tlist[k], step=0)  # integrate up to tlist[k]
        if ODE.successful():
            state = ODE.y / dznrm2(ODE.y)
            for jj in range(config.e_num):
                expect_out[jj][k] = cy_expect_psi_csr(
                    config.e_ops_data[jj], config.e_ops_ind[jj],
                    config.e_ops_ptr[jj], state,
                    config.e_ops_isherm[jj])
        else:
            raise ValueError('Error in ODE solver')

    return expect_out


# ---single-trajectory for monte-carlo---
def _mc_alg_evolve(nt, args, config):
    """
    Monte-Carlo algorithm returning state-vector or expectation values
    at times tlist for a single trajectory.
    """

    global _cy_rhs_func
    global _cy_col_spmv_func, _cy_col_expect_func
    global _cy_col_spmv_call_func, _cy_col_expect_call_func

    if not _cy_rhs_func:
        _mc_func_load(config)

    try:
        # get input data
        mc_alg_out, opt, tlist, num_times, seeds = args

        collapse_times = np.array([], dtype=float)  # times of collapses
        which_oper = array([], dtype=float)  # which operator did the collapse

        # SEED AND RNG AND GENERATE
        prng = RandomState(seeds[nt])
        # first rand is collapse norm, second is which operator
        rand_vals = prng.rand(2)

        # CREATE ODE OBJECT CORRESPONDING TO DESIRED TIME-DEPENDENCE
        if config.tflag in array([1, 10, 11]):
            ODE = ode(_cy_rhs_func)
            code = compile('ODE.set_f_params(' + config.string + ')',
                           '<string>', 'exec')
            exec(code)
        elif config.tflag == 2:
            ODE = ode(_cRHStd)
            ODE.set_f_params(config)
        elif config.tflag in array([20, 22]):
            if config.options.rhs_with_state:
                ODE = ode(_tdRHStd_with_state)
            else:
                ODE = ode(_tdRHStd)
            ODE.set_f_params(config)
        elif config.tflag == 3:
            if config.options.rhs_with_state:
                ODE = ode(_pyRHSc_with_state)
            else:
                ODE = ode(_pyRHSc)
            ODE.set_f_params(config)
        else:
            ODE = ode(_cy_rhs_func)
            ODE.set_f_params(config.h_data, config.h_ind,
                             config.h_ptr)

        # initialize ODE solver for RHS
        ODE.set_integrator('zvode', method=opt.method, order=opt.order,
                           atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                           first_step=opt.first_step, min_step=opt.min_step,
                           max_step=opt.max_step)

        # set initial conditions
        ODE.set_initial_value(config.psi0, tlist[0])

        # make array for collapse operator inds
        cinds = np.arange(config.c_num)

        # RUN ODE UNTIL EACH TIME IN TLIST
        for k in range(1, num_times):
            # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
            while ODE.t < tlist[k]:
                t_prev = ODE.t
                y_prev = ODE.y
                norm2_prev = dznrm2(ODE.y) ** 2
                # integrate up to tlist[k], one step at a time.
                ODE.integrate(tlist[k], step=1)
                if not ODE.successful():
                    raise Exception("ZVODE failed!")
                # check if ODE jumped over tlist[k], if so,
                # integrate until tlist exactly
                if ODE.t > tlist[k]:
                    ODE.set_initial_value(y_prev, t_prev)
                    ODE.integrate(tlist[k], step=0)
                    if not ODE.successful():
                        raise Exception("ZVODE failed!")
                norm2_psi = dznrm2(ODE.y) ** 2
                if norm2_psi <= rand_vals[0]:  # <== collapse has occured
                    # find collapse time to within specified tolerance
                    # ---------------------------------------------------
                    ii = 0
                    t_final = ODE.t
                    while ii < config.norm_steps:
                        ii += 1
                        t_guess = t_prev + np.log(norm2_prev / rand_vals[0]) / \
                            np.log(norm2_prev / norm2_psi) * (t_final - t_prev)
                        ODE.set_initial_value(y_prev, t_prev)
                        ODE.integrate(t_guess, step=0)
                        if not ODE.successful():
                            raise Exception(
                                "ZVODE failed after adjusting step size!")
                        norm2_guess = dznrm2(ODE.y)**2
                        if (np.abs(rand_vals[0] - norm2_guess) <
                                config.norm_tol * rand_vals[0]):
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
                    # ---------------------------------------------------
                    np.append(collapse_times, ODE.t)
                    # some string based collapse operators
                    if config.tflag in array([1, 11]):
                        n_dp = [cy_expect_psi_csr(config.n_ops_data[i],
                                                  config.n_ops_ind[i],
                                                  config.n_ops_ptr[i],
                                                  ODE.y, 1)
                                for i in config.c_const_inds]

                        _locals = locals()
                        # calculates the expectation values for time-dependent
                        # norm collapse operators
                        exec(_cy_col_expect_call_func, globals(), _locals)
                        n_dp = array(_locals['n_dp'])

                    # some Python function based collapse operators
                    elif config.tflag in array([2, 20, 22]):
                        n_dp = [cy_expect_psi_csr(config.n_ops_data[i],
                                                  config.n_ops_ind[i],
                                                  config.n_ops_ptr[i],
                                                  ODE.y, 1)
                                for i in config.c_const_inds]
                        n_dp += [abs(config.c_funcs[i](
                                     ODE.t, config.c_func_args)) ** 2 *
                                 cy_expect_psi_csr(config.n_ops_data[i],
                                                   config.n_ops_ind[i],
                                                   config.n_ops_ptr[i],
                                                   ODE.y, 1)
                                 for i in config.c_td_inds]
                        n_dp = array(n_dp)
                    # all constant collapse operators.
                    else:
                        n_dp = array(
                            [cy_expect_psi_csr(config.n_ops_data[i],
                                               config.n_ops_ind[i],
                                               config.n_ops_ptr[i],
                                               ODE.y, 1)
                             for i in range(config.c_num)])

                    # determine which operator does collapse and store it
                    kk = cumsum(n_dp / sum(n_dp))
                    j = cinds[kk >= rand_vals[1]][0]
                    np.append(which_oper, j)
                    if j in config.c_const_inds:
                        state = spmv_csr(config.c_ops_data[j],
                                         config.c_ops_ind[j],
                                         config.c_ops_ptr[j], ODE.y)
                    else:
                        if config.tflag in array([1, 11]):
                            _locals = locals()
                            # calculates the state vector for collapse by a
                            # time-dependent collapse operator
                            exec(_cy_col_spmv_call_func, globals(), _locals)
                            state = _locals['state']
                        else:
                            state = \
                                config.c_funcs[j](ODE.t,
                                                  config.c_func_args) * \
                                spmv_csr(config.c_ops_data[j],
                                         config.c_ops_ind[j],
                                         config.c_ops_ptr[j], ODE.y)
                    state = state / dznrm2(state)
                    ODE.set_initial_value(state, ODE.t)
                    rand_vals = prng.rand(2)
            # -------------------------------------------------------

            # -- after while loop --
            out_psi = ODE.y / dznrm2(ODE.y)
            if config.e_num == 0:
                out_psi = sp.csr_matrix(np.reshape(out_psi,
                                                   (out_psi.shape[0], 1)),
                                        dtype=complex)
                if (config.options.average_states and
                        not config.options.steady_state_average):
                    mc_alg_out[k] = Qobj(out_psi * out_psi.conj().transpose(),
                                         [config.psi0_dims[0],
                                          config.psi0_dims[0]],
                                         [config.psi0_shape[0],
                                          config.psi0_shape[0]],
                                         fast='mc-dm')

                elif config.options.steady_state_average:
                    mc_alg_out[0] = (mc_alg_out[0] +
                                     (out_psi * out_psi.conj().transpose()))

                else:
                    mc_alg_out[k] = Qobj(out_psi, config.psi0_dims,
                                         config.psi0_shape, fast='mc')
            else:
                for jj in range(config.e_num):
                    mc_alg_out[jj][k] = cy_expect_psi_csr(
                        config.e_ops_data[jj], config.e_ops_ind[jj],
                        config.e_ops_ptr[jj], out_psi,
                        config.e_ops_isherm[jj])

        # Run at end of mc_alg function
        # ------------------------------
        if config.options.steady_state_average:
            mc_alg_out = array([Qobj(mc_alg_out[0] / float(len(tlist)),
                                [config.psi0_dims[0],
                                 config.psi0_dims[0]],
                                [config.psi0_shape[0],
                                 config.psi0_shape[0]],
                                fast='mc-dm')])

        return nt, mc_alg_out, collapse_times, which_oper

    except Exception as e:
        print("failed to run _mc_alg_evolve: " + str(e))


def _mc_func_load(config):
    """Load cython functions"""

    global _cy_rhs_func
    global _cy_col_spmv_func, _cy_col_expect_func
    global _cy_col_spmv_call_func, _cy_col_expect_call_func

    if debug:
        print(inspect.stack()[0][3] + " in " + str(os.getpid()))

    if config.tflag in array([1, 10, 11]):
        # compile time-depdendent RHS code
        if config.tflag in array([1, 11]):
            code = compile('from ' + config.tdname +
                           ' import cy_td_ode_rhs, col_spmv, col_expect',
                           '<string>', 'exec')
            exec(code, globals())
            _cy_rhs_func = cy_td_ode_rhs
            _cy_col_spmv_func = col_spmv
            _cy_col_expect_func = col_expect
        else:
            code = compile('from ' + config.tdname +
                           ' import cy_td_ode_rhs', '<string>', 'exec')
            exec(code, globals())
            _cy_rhs_func = cy_td_ode_rhs

        # compile wrapper functions for calling cython spmv and expect
        if config.col_spmv_code:
            _cy_col_spmv_call_func = compile(
                config.col_spmv_code, '<string>', 'exec')

        if config.col_expect_code:
            _cy_col_expect_call_func = compile(
                config.col_expect_code, '<string>', 'exec')

    elif config.tflag == 0:
        _cy_rhs_func = cy_ode_rhs


def _mc_data_config(H, psi0, h_stuff, c_ops, c_stuff, args, e_ops,
                    options, config):
    """Creates the appropriate data structures for the monte carlo solver
    based on the given time-dependent, or indepdendent, format.
    """

    if debug:
        print(inspect.stack()[0][3])

    config.soft_reset()

    # take care of expectation values, if any
    if any(e_ops):
        config.e_num = len(e_ops)
        for op in e_ops:
            if isinstance(op, list):
                op = op[0]
            config.e_ops_data.append(op.data.data)
            config.e_ops_ind.append(op.data.indices)
            config.e_ops_ptr.append(op.data.indptr)
            config.e_ops_isherm.append(op.isherm)

        config.e_ops_data = array(config.e_ops_data)
        config.e_ops_ind = array(config.e_ops_ind)
        config.e_ops_ptr = array(config.e_ops_ptr)
        config.e_ops_isherm = array(config.e_ops_isherm)
    # ----

    # take care of collapse operators, if any
    if any(c_ops):
        config.c_num = len(c_ops)
        for c_op in c_ops:
            if isinstance(c_op, list):
                c_op = c_op[0]
            n_op = c_op.dag() * c_op
            config.c_ops_data.append(c_op.data.data)
            config.c_ops_ind.append(c_op.data.indices)
            config.c_ops_ptr.append(c_op.data.indptr)
            # norm ops
            config.n_ops_data.append(n_op.data.data)
            config.n_ops_ind.append(n_op.data.indices)
            config.n_ops_ptr.append(n_op.data.indptr)
        # to array
        config.c_ops_data = array(config.c_ops_data)
        config.c_ops_ind = array(config.c_ops_ind)
        config.c_ops_ptr = array(config.c_ops_ptr)

        config.n_ops_data = array(config.n_ops_data)
        config.n_ops_ind = array(config.n_ops_ind)
        config.n_ops_ptr = array(config.n_ops_ptr)
    # ----

    # --------------------------------------------
    # START CONSTANT H & C_OPS CODE
    # --------------------------------------------
    if config.tflag == 0:
        if config.cflag:
            config.c_const_inds = arange(len(c_ops))
            for c_op in c_ops:
                n_op = c_op.dag() * c_op
                H -= 0.5j * \
                    n_op  # combine Hamiltonian and collapse terms into one
        # construct Hamiltonian data structures
        if options.tidy:
            H = H.tidyup(options.atol)
        config.h_data = -1.0j * H.data.data
        config.h_ind = H.data.indices
        config.h_ptr = H.data.indptr
    # ----

    # --------------------------------------------
    # START STRING BASED TIME-DEPENDENCE
    # --------------------------------------------
    elif config.tflag in array([1, 10, 11]):
        # take care of arguments for collapse operators, if any
        if any(args):
            for item in args.items():
                config.c_args.append(item[1])
        # constant Hamiltonian / string-type collapse operators
        if config.tflag == 1:
            H_inds = arange(1)
            H_tdterms = 0
            len_h = 1
            C_inds = arange(config.c_num)
            # find inds of time-dependent terms
            C_td_inds = array(c_stuff[2])
            # find inds of constant terms
            C_const_inds = setdiff1d(C_inds, C_td_inds)
            # extract time-dependent coefficients (strings)
            C_tdterms = [c_ops[k][1] for k in C_td_inds]
            # store indicies of constant collapse terms
            config.c_const_inds = C_const_inds
            # store indicies of time-dependent collapse terms
            config.c_td_inds = C_td_inds

            for k in config.c_const_inds:
                H -= 0.5j * (c_ops[k].dag() * c_ops[k])
            if options.tidy:
                H = H.tidyup(options.atol)
            config.h_data = [H.data.data]
            config.h_ind = [H.data.indices]
            config.h_ptr = [H.data.indptr]
            for k in config.c_td_inds:
                op = c_ops[k][0].dag() * c_ops[k][0]
                config.h_data.append(-0.5j * op.data.data)
                config.h_ind.append(op.data.indices)
                config.h_ptr.append(op.data.indptr)
            config.h_data = -1.0j * array(config.h_data)
            config.h_ind = array(config.h_ind)
            config.h_ptr = array(config.h_ptr)
            # --------------------------------------------
            # END OF IF STATEMENT
            # --------------------------------------------

        # string-type Hamiltonian & at least one string-type collapse operator
        else:
            H_inds = arange(len(H))
            # find inds of time-dependent terms
            H_td_inds = array(h_stuff[2])
            # find inds of constant terms
            H_const_inds = setdiff1d(H_inds, H_td_inds)
            # extract time-dependent coefficients (strings or functions)
            H_tdterms = [H[k][1] for k in H_td_inds]
            # combine time-INDEPENDENT terms into one.
            H = array([sum(H[k] for k in H_const_inds)] +
                      [H[k][0] for k in H_td_inds])
            len_h = len(H)
            H_inds = arange(len_h)
            # store indicies of time-dependent Hamiltonian terms
            config.h_td_inds = arange(1, len_h)
            # if there are any collapse operators
            if config.c_num > 0:
                if config.tflag == 10:
                    # constant collapse operators
                    config.c_const_inds = arange(config.c_num)
                    for k in config.c_const_inds:
                        H[0] -= 0.5j * (c_ops[k].dag() * c_ops[k])
                    C_inds = arange(config.c_num)
                    C_tdterms = array([])
                else:
                    # some time-dependent collapse terms
                    C_inds = arange(config.c_num)
                    # find inds of time-dependent terms
                    C_td_inds = array(c_stuff[2])
                    # find inds of constant terms
                    C_const_inds = setdiff1d(C_inds, C_td_inds)
                    C_tdterms = [c_ops[k][1] for k in C_td_inds]
                    # extract time-dependent coefficients (strings)
                    # store indicies of constant collapse terms
                    config.c_const_inds = C_const_inds
                    # store indicies of time-dependent collapse terms
                    config.c_td_inds = C_td_inds
                    for k in config.c_const_inds:
                        H[0] -= 0.5j * (c_ops[k].dag() * c_ops[k])
            else:
                # set empty objects if no collapse operators
                C_const_inds = arange(config.c_num)
                config.c_const_inds = arange(config.c_num)
                config.c_td_inds = array([])
                C_tdterms = array([])
                C_inds = array([])

            # tidyup
            if options.tidy:
                H = array([H[k].tidyup(options.atol) for k in range(len_h)])
            # construct data sets
            config.h_data = [H[k].data.data for k in range(len_h)]
            config.h_ind = [H[k].data.indices for k in range(len_h)]
            config.h_ptr = [H[k].data.indptr for k in range(len_h)]
            for k in config.c_td_inds:
                config.h_data.append(-0.5j * config.n_ops_data[k])
                config.h_ind.append(config.n_ops_ind[k])
                config.h_ptr.append(config.n_ops_ptr[k])
            config.h_data = -1.0j * array(config.h_data)
            config.h_ind = array(config.h_ind)
            config.h_ptr = array(config.h_ptr)
            # --------------------------------------------
            # END OF ELSE STATEMENT
            # --------------------------------------------

        # set execuatble code for collapse expectation values and spmv
        col_spmv_code = ("state = _cy_col_spmv_func(j, ODE.t, " +
                         "config.c_ops_data[j], config.c_ops_ind[j], " +
                         "config.c_ops_ptr[j], ODE.y")
        col_expect_code = ("for i in config.c_td_inds: " +
                           "n_dp.append(_cy_col_expect_func(i, ODE.t, " +
                           "config.n_ops_data[i], " +
                           "config.n_ops_ind[i], " +
                           "config.n_ops_ptr[i], ODE.y")
        for kk in range(len(config.c_args)):
            col_spmv_code += ",config.c_args[" + str(kk) + "]"
            col_expect_code += ",config.c_args[" + str(kk) + "]"
        col_spmv_code += ")"
        col_expect_code += "))"

        config.col_spmv_code = col_spmv_code
        config.col_expect_code = col_expect_code

        # setup ode args string
        config.string = ""
        data_range = range(len(config.h_data))
        for k in data_range:
            config.string += ("config.h_data[" + str(k) +
                              "], config.h_ind[" + str(k) +
                              "], config.h_ptr[" + str(k) + "]")
            if k != data_range[-1]:
                config.string += ","
        # attach args to ode args string
        if len(config.c_args) > 0:
            for kk in range(len(config.c_args)):
                config.string += "," + "config.c_args[" + str(kk) + "]"
        # ----

        name = "rhs" + str(os.getpid()) + str(config.cgen_num)
        config.tdname = name
        cgen = Codegen(H_inds, H_tdterms, config.h_td_inds, args,
                       C_inds, C_tdterms, config.c_td_inds, type='mc',
                       config=config)
        cgen.generate(name + ".pyx")

    # -------------------------------------------------
    # START PYTHON LIST-FUNCTION BASED TIME-DEPENDENCE
    # -------------------------------------------------
    elif config.tflag in array([2, 20, 22]):

        # take care of Hamiltonian
        if config.tflag == 2:
            # constant Hamiltonian, at least one function based collapse
            # operators
            H_inds = array([0])
            H_tdterms = 0
            len_h = 1
        else:
            # function based Hamiltonian
            H_inds = arange(len(H))
            H_td_inds = array(h_stuff[1])  # find inds of time-dependent terms
            H_const_inds = setdiff1d(H_inds, H_td_inds)  # inds of const. terms
            config.h_funcs = array([H[k][1] for k in H_td_inds])
            config.h_func_args = args
            Htd = array([H[k][0] for k in H_td_inds])
            config.h_td_inds = arange(len(Htd))
            H = sum(H[k] for k in H_const_inds)

        # take care of collapse operators
        C_inds = arange(config.c_num)
        # find inds of time-dependent terms
        C_td_inds = array(c_stuff[1])
        # find inds of constant terms
        C_const_inds = setdiff1d(C_inds, C_td_inds)
        # store indicies of constant collapse terms
        config.c_const_inds = C_const_inds
        # store indicies of time-dependent collapse terms
        config.c_td_inds = C_td_inds
        config.c_funcs = zeros(config.c_num, dtype=FunctionType)
        for k in config.c_td_inds:
            config.c_funcs[k] = c_ops[k][1]
        config.c_func_args = args

        # combine constant collapse terms with constant H and construct data
        for k in config.c_const_inds:
            H -= 0.5j * (c_ops[k].dag() * c_ops[k])
        if options.tidy:
            H = H.tidyup(options.atol)
            Htd = array([Htd[j].tidyup(options.atol)
                         for j in config.h_td_inds])
        # setup constant H terms data
        config.h_data = -1.0j * H.data.data
        config.h_ind = H.data.indices
        config.h_ptr = H.data.indptr

        # setup td H terms data
        config.h_td_data = array(
            [-1.0j * Htd[k].data.data for k in config.h_td_inds])
        config.h_td_ind = array(
            [Htd[k].data.indices for k in config.h_td_inds])
        config.h_td_ptr = array(
            [Htd[k].data.indptr for k in config.h_td_inds])

    # --------------------------------------------
    # START PYTHON FUNCTION BASED HAMILTONIAN
    # --------------------------------------------
    elif config.tflag == 3:
        # take care of Hamiltonian
        config.h_funcs = H
        config.h_func_args = args

        # take care of collapse operators
        config.c_const_inds = arange(config.c_num)
        config.c_td_inds = array([])  # find inds of time-dependent terms
        if len(config.c_const_inds) > 0:
            H = 0
            for k in config.c_const_inds:
                H -= 0.5j * (c_ops[k].dag() * c_ops[k])
            if options.tidy:
                H = H.tidyup(options.atol)
            config.h_data = -1.0j * H.data.data
            config.h_ind = H.data.indices
            config.h_ptr = H.data.indptr


def _mc_dm_avg(psi_list):
    """
    Private function that averages density matrices in parallel
    over all trajectories for a single time using parfor.
    """
    ln = len(psi_list)
    dims = psi_list[0].dims
    shape = psi_list[0].shape
    out_data = mean([psi.data for psi in psi_list])
    return Qobj(out_data, dims=dims, shape=shape, fast='mc-dm')
