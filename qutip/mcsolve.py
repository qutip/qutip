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
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import sys
import os
import time
import numpy as np
import datetime
from types import FunctionType
from multiprocessing import Pool, cpu_count
from numpy.random import RandomState, random_integers
from scipy import arange, array, cumsum, mean, ndarray, setdiff1d, sort, zeros
from scipy.integrate import ode
from scipy.linalg import norm
from qutip.qobj import *
from qutip.expect import *
from qutip.states import ket2dm
from qutip.parfor import parfor
from qutip.odeoptions import Odeoptions
from qutip.odeconfig import odeconfig
from qutip.cyQ.spmatfuncs import cy_ode_rhs, cy_expect, spmv, spmv1d
from qutip.cyQ.codegen import Codegen
from qutip.odedata import Odedata
from qutip.odechecks import _ode_checks
import qutip.settings
from qutip.settings import debug

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
            args={}, options=Odeoptions()):
    """Monte-Carlo evolution of a state vector :math:`|\psi \\rangle` for a
    given Hamiltonian and sets of collapse operators, and possibly, operators
    for calculating expectation values. Options for the underlying ODE solver
    are given by the Odeoptions class.

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
    options : Odeoptions
        Instance of ODE solver options.

    Returns
    -------
    results : Odedata
        Object storing all results from simulation.

    """

    if debug:
        print(inspect.stack()[0][3])

    if ntraj is None:
        ntraj = options.ntraj

    # if single operator is passed for c_ops or e_ops, convert it to
    # list containing only that operator
    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if psi0.type != 'ket':
        raise Exception("Initial state must be a state vector.")
    odeconfig.options = options
    # set num_cpus to the value given in qutip.settings if none in Odeoptions
    if not odeconfig.options.num_cpus:
        odeconfig.options.num_cpus = qutip.settings.num_cpus
    # set initial value data
    if options.tidy:
        odeconfig.psi0 = psi0.tidyup(options.atol).full()
    else:
        odeconfig.psi0 = psi0.full()
    odeconfig.psi0_dims = psi0.dims
    odeconfig.psi0_shape = psi0.shape
    # set general items
    odeconfig.tlist = tlist
    if isinstance(ntraj, (list, ndarray)):
        odeconfig.ntraj = sort(ntraj)[-1]
    else:
        odeconfig.ntraj = ntraj
    # set norm finding constants
    odeconfig.norm_tol = options.norm_tol
    odeconfig.norm_steps = options.norm_steps
    #----

    #----------------------------------------------
    # SETUP ODE DATA IF NONE EXISTS OR NOT REUSING
    #----------------------------------------------
    if (not options.rhs_reuse) or (not odeconfig.tdfunc):
        # reset odeconfig collapse and time-dependence flags to default values
        odeconfig.soft_reset()

        # check for type of time-dependence (if any)
        time_type, h_stuff, c_stuff = _ode_checks(H, c_ops, 'mc')
        h_terms = len(h_stuff[0]) + len(h_stuff[1]) + len(h_stuff[2])
        c_terms = len(c_stuff[0]) + len(c_stuff[1]) + len(c_stuff[2])
        # set time_type for use in multiprocessing
        odeconfig.tflag = time_type

        #-Check for PyObjC on Mac platforms
        if sys.platform == 'darwin' and odeconfig.options.gui:
            try:
                import Foundation
            except:
                odeconfig.options.gui = False

        # check if running in iPython and using Cython compiling (then no GUI
        # to work around error)
        if odeconfig.options.gui and odeconfig.tflag in array([1, 10, 11]):
            try:
                __IPYTHON__
            except:
                pass
            else:
                odeconfig.options.gui = False
        if qutip.settings.qutip_gui == "NONE":
            odeconfig.options.gui = False

        # check for collapse operators
        if c_terms > 0:
            odeconfig.cflag = 1
        else:
            odeconfig.cflag = 0

        # Configure data
        _mc_data_config(H, psi0, h_stuff, c_ops, c_stuff, args, e_ops,
                        options, odeconfig)

        # compile and load cython functions if necessary
        _mc_func_load(odeconfig)

    else:
        # setup args for new parameters when rhs_reuse=True and tdfunc is given
        # string based
        if odeconfig.tflag in array([1, 10, 11]):
            if any(args):
                odeconfig.c_args = []
                arg_items = args.items()
                for k in range(len(args)):
                    odeconfig.c_args.append(arg_items[k][1])
        # function based
        elif odeconfig.tflag in array([2, 3, 20, 22]):
            odeconfig.h_func_args = args

    # load monte-carlo class
    mc = _MC_class(odeconfig)

    # RUN THE SIMULATION
    mc.run()

    # remove RHS cython file if necessary
    if not options.rhs_reuse and odeconfig.tdname:
        try:
            os.remove(odeconfig.tdname + ".pyx")
        except Exception as e:
            print("Error removing pyx file: " + str(e))

    # AFTER MCSOLVER IS DONE --------------------------------------
    #-------COLLECT AND RETURN OUTPUT DATA IN ODEDATA OBJECT --------------#
    output = Odedata()
    output.solver = 'mcsolve'
    # state vectors
    if mc.psi_out is not None and odeconfig.options.mc_avg and odeconfig.cflag:
        output.states = parfor(_mc_dm_avg, mc.psi_out.T)
    elif mc.psi_out is not None:
        output.states = mc.psi_out
    # expectation values
    elif (mc.expect_out is not None and odeconfig.cflag
            and odeconfig.options.mc_avg):
        # averaging if multiple trajectories
        if isinstance(ntraj, int):
            output.expect = [mean([mc.expect_out[nt][op]
                                   for nt in range(ntraj)], axis=0)
                             for op in range(odeconfig.e_num)]
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
        # no averaging for single trajectory or if mc_avg flag
        # (Odeoptions) is off
        if mc.expect_out is not None:
            output.expect = mc.expect_out

    # simulation parameters
    output.times = odeconfig.tlist
    output.num_expect = odeconfig.e_num
    output.num_collapse = odeconfig.c_num
    output.ntraj = odeconfig.ntraj
    output.col_times = mc.collapse_times_out
    output.col_which = mc.which_op_out
    return output


#--------------------------------------------------------------
# MONTE-CARLO CLASS                                           #
#--------------------------------------------------------------
class _MC_class():
    """
    Private class for solving Monte-Carlo evolution from mcsolve
    """

    def __init__(self, odeconfig):

        #-----------------------------------#
        # INIT MC CLASS
        #-----------------------------------#

        self.odeconfig = odeconfig

        #----MAIN OBJECT PROPERTIES--------------------#
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
        self.num_times = len(self.odeconfig.tlist)
        # holds seed for random number generator
        self.seed = None
        # holds expected time to completion
        self.st = None
        # number of cpus to be used
        self.cpus = self.odeconfig.options.num_cpus
        # set output variables, even if they are not used to simplify output
        # code.
        self.psi_out = None
        self.expect_out = []
        self.collapse_times_out = None
        self.which_op_out = None

        # FOR EVOLUTION FOR NO COLLAPSE OPERATORS
        if odeconfig.c_num == 0:
            if odeconfig.e_num == 0:
                # Output array of state vectors calculated at times in tlist
                self.psi_out = array([Qobj()] * self.num_times)
            elif odeconfig.e_num != 0:  # no collpase expectation values
                # List of output expectation values calculated at times in
                # tlist
                self.expect_out = []
                for i in range(odeconfig.e_num):
                    if odeconfig.e_ops_isherm[i]:
                        # preallocate real array of zeros
                        self.expect_out.append(zeros(self.num_times))
                    else:  # preallocate complex array of zeros
                        self.expect_out.append(
                            zeros(self.num_times, dtype=complex))
                    self.expect_out[i][0] = cy_expect(
                        odeconfig.e_ops_data[i], odeconfig.e_ops_ind[i],
                        odeconfig.e_ops_ptr[i], odeconfig.e_ops_isherm[i],
                        odeconfig.psi0)

        # FOR EVOLUTION WITH COLLAPSE OPERATORS
        elif odeconfig.c_num != 0:
            # preallocate #ntraj arrays for state vectors, collapse times, and
            # which operator
            self.collapse_times_out = zeros((odeconfig.ntraj), dtype=ndarray)
            self.which_op_out = zeros((odeconfig.ntraj), dtype=ndarray)
            if odeconfig.e_num == 0:
                # if no expectation operators, preallocate #ntraj arrays
                # for state vectors
                self.psi_out = array([zeros((self.num_times), dtype=object)
                                     for q in range(odeconfig.ntraj)])
            else:  # preallocate array of lists for expectation values
                self.expect_out = [[] for x in range(odeconfig.ntraj)]

    #-------------------------------------------------#
    # CLASS METHODS
    #-------------------------------------------------#
    def callback(self, results):
        r = results[0]
        if self.odeconfig.e_num == 0:  # output state-vector
            self.psi_out[r] = results[1]
        else:  # output expectation values
            self.expect_out[r] = results[1]
        self.collapse_times_out[r] = results[2]
        self.which_op_out[r] = results[3]
        self.count += self.step
        if (not self.odeconfig.options.gui and self.odeconfig.ntraj != 1):
            # print to term
            self.percent = self.count / (1.0 * self.odeconfig.ntraj)
            if self.count / float(self.odeconfig.ntraj) >= self.level:
                # calls function to determine simulation time remaining
                self.level = _time_remaining(
                    self.st, self.odeconfig.ntraj, self.count, self.level)
    #-----

    def parallel(self, args, top=None):

        if debug:
            print(inspect.stack()[0][3])

        self.st = datetime.datetime.now()  # set simulation starting time
        pl = Pool(processes=self.cpus)
        [pl.apply_async(_mc_alg_evolve,
                        args=(nt, args, self.odeconfig),
                        callback=top.callback)
         for nt in range(0, self.odeconfig.ntraj)]
        pl.close()
        try:
            pl.join()
        except KeyboardInterrupt:
            print("Cancel all MC threads on keyboard interrupt")
            pl.terminate()
            pl.join()
        return
    #-----

    def run(self):

        if debug:
            print(inspect.stack()[0][3])

        if self.odeconfig.c_num == 0:
            if self.odeconfig.ntraj != 1:
                # Ntraj != 1 IS pointless for no collapse operators
                self.odeconfig.ntraj = 1
                print('No collapse operators specified.\n' +
                      'Running a single trajectory only.\n')
            if self.odeconfig.e_num == 0:  # return psi at each requested time
                self.psi_out = _no_collapse_psi_out(
                    self.num_times, self.psi_out, self.odeconfig)
            else:  # return expectation values of requested operators
                self.expect_out = _no_collapse_expect_out(
                    self.num_times, self.expect_out, self.odeconfig)
        elif self.odeconfig.c_num != 0:
            self.seed = random_integers(1e8, size=self.odeconfig.ntraj)
            if self.odeconfig.e_num == 0:
                mc_alg_out = zeros((self.num_times), dtype=ndarray)
                if self.odeconfig.options.mc_avg:
                    # output is averaged states, so use dm
                    mc_alg_out[0] = \
                        self.odeconfig.psi0 * self.odeconfig.psi0.conj().T
                else:
                    # output is not averaged, so write state vectors
                    mc_alg_out[0] = self.odeconfig.psi0
            else:
                # PRE-GENERATE LIST OF EXPECTATION VALUES
                mc_alg_out = []
                for i in range(self.odeconfig.e_num):
                    if self.odeconfig.e_ops_isherm[i]:
                        # preallocate real array of zeros
                        mc_alg_out.append(zeros(self.num_times))
                    else:
                        # preallocate complex array of zeros
                        mc_alg_out.append(zeros(self.num_times, dtype=complex))
                    mc_alg_out[i][0] = \
                        cy_expect(self.odeconfig.e_ops_data[i],
                                  self.odeconfig.e_ops_ind[i],
                                  self.odeconfig.e_ops_ptr[i],
                                  self.odeconfig.e_ops_isherm[i],
                                  self.odeconfig.psi0)
            # set arguments for input to monte-carlo
            args = (mc_alg_out, self.odeconfig.options,
                    self.odeconfig.tlist, self.num_times, self.seed)
            if not self.odeconfig.options.gui:
                self.parallel(args, self)
            else:
                if qutip.settings.qutip_gui == "PYSIDE":
                    from PySide import QtGui, QtCore
                elif qutip.settings.qutip_gui == "PYQT4":
                    from PyQt4 import QtGui, QtCore
                from gui.ProgressBar import ProgressBar, Pthread
                # checks if QApplication already exists (needed for iPython)
                app = QtGui.QApplication.instance()
                if not app:  # create QApplication if it doesnt exist
                    app = QtGui.QApplication(sys.argv)
                thread = Pthread(target=self.parallel, args=args, top=self)
                self.bar = ProgressBar(
                    self, thread, self.odeconfig.ntraj, self.cpus)
                QtCore.QTimer.singleShot(0, self.bar.run)
                self.bar.show()
                self.bar.activateWindow()
                self.bar.raise_()
                app.exec_()
                return


#----------------------------------------------------
# CODES FOR PYTHON FUNCTION BASED TIME-DEPENDENT RHS
#----------------------------------------------------

# RHS of ODE for time-dependent systems with no collapse operators
def _tdRHS(t, psi, odeconfig):
    h_data = odeconfig.h_func(t, odeconfig.h_func_args).data
    return spmv1d(h_data.data, h_data.indices, h_data.indptr, psi)


# RHS of ODE for constant Hamiltonian and at least one function based
# collapse operator
def _cRHStd(t, psi, odeconfig):
    sys = cy_ode_rhs(t, psi, odeconfig.h_data,
                     odeconfig.h_ind, odeconfig.h_ptr)
    col = array([np.abs(odeconfig.c_funcs[j](t, odeconfig.c_func_args)) ** 2 *
                 spmv1d(odeconfig.n_ops_data[j],
                        odeconfig.n_ops_ind[j],
                        odeconfig.n_ops_ptr[j], psi)
                for j in odeconfig.c_td_inds])
    return sys - 0.5 * np.sum(col, 0)


# RHS of ODE for function-list based Hamiltonian
def _tdRHStd(t, psi, odeconfig):
    const_term = spmv1d(odeconfig.h_data,
                        odeconfig.h_ind,
                        odeconfig.h_ptr, psi)
    h_func_term = array([odeconfig.h_funcs[j](t, odeconfig.h_func_args) *
                         spmv1d(odeconfig.h_td_data[j],
                                odeconfig.h_td_ind[j],
                                odeconfig.h_td_ptr[j], psi)
                         for j in odeconfig.h_td_inds])
    col_func_terms = array([np.abs(
        odeconfig.c_funcs[j](t, odeconfig.c_func_args)) ** 2 *
        spmv1d(odeconfig.n_ops_data[j],
               odeconfig.n_ops_ind[j],
               odeconfig.n_ops_ptr[j],
               psi)
        for j in odeconfig.c_td_inds])
    return (const_term - np.sum(h_func_term, 0)
            - 0.5 * np.sum(col_func_terms, 0))


# RHS of ODE for python function Hamiltonian
def _pyRHSc(t, psi, odeconfig):
    h_func_data = odeconfig.h_funcs(t, odeconfig.h_func_args).data
    h_func_term = spmv1d(h_func_data.data, h_func_data.indices,
                         h_func_data.indptr, psi)
    const_col_term = 0
    if len(odeconfig.c_const_inds) > 0:
        const_col_term = spmv1d(
            odeconfig.h_data, odeconfig.h_ind, odeconfig.h_ptr, psi)
    return h_func_term + const_col_term


#----------------------------------------------------
# END PYTHON FUNCTION RHS
#----------------------------------------------------

######---return psi at requested times for no collapse operators---######
def _no_collapse_psi_out(num_times, psi_out, odeconfig):
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
        _mc_func_load(odeconfig)

    opt = odeconfig.options
    if odeconfig.tflag in array([1, 10, 11]):
        ODE = ode(_cy_rhs_func)
        code = compile('ODE.set_f_params(' + odeconfig.string + ')',
                       '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag == 2:
        ODE = ode(_cRHStd)
        ODE.set_f_params(odeconfig)
    elif odeconfig.tflag in array([20, 22]):
        ODE = ode(_tdRHStd)
        ODE.set_f_params(odeconfig)
    elif odeconfig.tflag == 3:
        ODE = ode(_pyRHSc)
        ODE.set_f_params(odeconfig)
    else:
        ODE = ode(cy_ode_rhs)
        ODE.set_f_params(odeconfig.h_data, odeconfig.h_ind, odeconfig.h_ptr)

    # initialize ODE solver for RHS
    ODE.set_integrator('zvode', method=opt.method, order=opt.order,
                       atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                       first_step=opt.first_step, min_step=opt.min_step,
                       max_step=opt.max_step)
    # set initial conditions
    ODE.set_initial_value(odeconfig.psi0, odeconfig.tlist[0])
    psi_out[0] = Qobj(
        odeconfig.psi0, odeconfig.psi0_dims, odeconfig.psi0_shape, 'ket')
    for k in range(1, num_times):
        ODE.integrate(odeconfig.tlist[k], step=0)  # integrate up to tlist[k]
        if ODE.successful():
            psi_out[k] = Qobj(ODE.y / norm(
                ODE.y, 2), odeconfig.psi0_dims, odeconfig.psi0_shape, 'ket')
        else:
            raise ValueError('Error in ODE solver')
    return psi_out
#------------------------------------------------------------------------


######---return expectation values at requested times for no collapse oper
def _no_collapse_expect_out(num_times, expect_out, odeconfig):
    """
    Calculates expect.values at times tlist if no collapse ops. given
    """

    global _cy_rhs_func
    global _cy_col_spmv_func, _cy_col_expect_func
    global _cy_col_spmv_call_func, _cy_col_expect_call_func

    if debug:
        print(inspect.stack()[0][3])

    if not _cy_rhs_func:
        _mc_func_load(odeconfig)

    opt = odeconfig.options
    if odeconfig.tflag in array([1, 10, 11]):
        ODE = ode(_cy_rhs_func)
        code = compile('ODE.set_f_params(' + odeconfig.string + ')',
                       '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag == 2:
        ODE = ode(_cRHStd)
        ODE.set_f_params(odeconfig)
    elif odeconfig.tflag in array([20, 22]):
        ODE = ode(_tdRHStd)
        ODE.set_f_params(odeconfig)
    elif odeconfig.tflag == 3:
        ODE = ode(_pyRHSc)
        ODE.set_f_params(odeconfig)
    else:
        ODE = ode(cy_ode_rhs)
        ODE.set_f_params(odeconfig.h_data, odeconfig.h_ind, odeconfig.h_ptr)

    ODE.set_integrator('zvode', method=opt.method, order=opt.order,
                       atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                       first_step=opt.first_step, min_step=opt.min_step,
                       max_step=opt.max_step)  # initialize ODE solver for RHS
    ODE.set_initial_value(
        odeconfig.psi0, odeconfig.tlist[0])  # set initial conditions
    for jj in range(odeconfig.e_num):
        expect_out[jj][0] = cy_expect(odeconfig.e_ops_data[jj],
                                      odeconfig.e_ops_ind[jj],
                                      odeconfig.e_ops_ptr[jj],
                                      odeconfig.e_ops_isherm[jj],
                                      odeconfig.psi0)
    for k in range(1, num_times):
        ODE.integrate(odeconfig.tlist[k], step=0)  # integrate up to tlist[k]
        if ODE.successful():
            state = ODE.y / norm(ODE.y)
            for jj in range(odeconfig.e_num):
                expect_out[jj][k] = cy_expect(odeconfig.e_ops_data[jj],
                                              odeconfig.e_ops_ind[jj],
                                              odeconfig.e_ops_ptr[jj],
                                              odeconfig.e_ops_isherm[jj],
                                              state)
        else:
            raise ValueError('Error in ODE solver')
    return expect_out  # return times and expectiation values
#------------------------------------------------------------------------


#---single-trajectory for monte-carlo---
def _mc_alg_evolve(nt, args, odeconfig):
    """
    Monte-Carlo algorithm returning state-vector or expectation values
    at times tlist for a single trajectory.
    """

    global _cy_rhs_func
    global _cy_col_spmv_func, _cy_col_expect_func
    global _cy_col_spmv_call_func, _cy_col_expect_call_func

    if not _cy_rhs_func:
        _mc_func_load(odeconfig)

    try:
        # get input data
        mc_alg_out, opt, tlist, num_times, seeds = args

        collapse_times = []  # times at which collapse occurs
        which_oper = []  # which operator did the collapse

        # SEED AND RNG AND GENERATE
        prng = RandomState(seeds[nt])
        rand_vals = prng.rand(2)  # first rand is collapse norm,
                                  # second is which operator

        # CREATE ODE OBJECT CORRESPONDING TO DESIRED TIME-DEPENDENCE
        if odeconfig.tflag in array([1, 10, 11]):
            ODE = ode(_cy_rhs_func)
            code = compile('ODE.set_f_params(' + odeconfig.string + ')',
                           '<string>', 'exec')
            exec(code)
        elif odeconfig.tflag == 2:
            ODE = ode(_cRHStd)
            ODE.set_f_params(odeconfig)
        elif odeconfig.tflag in array([20, 22]):
            ODE = ode(_tdRHStd)
            ODE.set_f_params(odeconfig)
        elif odeconfig.tflag == 3:
            ODE = ode(_pyRHSc)
            ODE.set_f_params(odeconfig)
        else:
            ODE = ode(_cy_rhs_func)
            ODE.set_f_params(odeconfig.h_data, odeconfig.h_ind,
                             odeconfig.h_ptr)

        # initialize ODE solver for RHS
        ODE.set_integrator('zvode', method=opt.method, order=opt.order,
                           atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                           first_step=opt.first_step, min_step=opt.min_step,
                           max_step=opt.max_step)

        # set initial conditions
        ODE.set_initial_value(odeconfig.psi0, tlist[0])

        # make array for collapse operator inds
        cinds = arange(odeconfig.c_num)

        # RUN ODE UNTIL EACH TIME IN TLIST
        for k in range(1, num_times):
            # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
            while ODE.t < tlist[k]:
                t_prev = ODE.t
                y_prev = ODE.y
                norm2_prev = norm(ODE.y, 2) ** 2
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
                norm2_psi = norm(ODE.y, 2) ** 2
                if norm2_psi <= rand_vals[0]:  # <== collpase has occured
                    # find collpase time to within specified tolerance
                    #---------------------------------------------------
                    ii = 0
                    t_final = ODE.t
                    while ii < odeconfig.norm_steps:
                        ii += 1
                        t_guess = t_prev + log(norm2_prev / rand_vals[0]) / \
                            log(norm2_prev / norm2_psi) * (t_final - t_prev)
                        ODE.set_initial_value(y_prev, t_prev)
                        ODE.integrate(t_guess, step=0)
                        if not ODE.successful():
                            raise Exception(
                                "ZVODE failed after adjusting step size!")
                        norm2_guess = norm(ODE.y, 2) ** 2
                        if (abs(rand_vals[0] - norm2_guess) <
                                odeconfig.norm_tol * rand_vals[0]):
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
                    if ii > odeconfig.norm_steps:
                        raise Exception("Norm tolerance not reached. " +
                                        "Increase accuracy of ODE solver or " +
                                        "Odeoptions.norm_steps.")
                    #---------------------------------------------------
                    collapse_times.append(ODE.t)
                    # some string based collapse operators
                    if odeconfig.tflag in array([1, 11]):
                        n_dp = [cy_expect(odeconfig.n_ops_data[i],
                                          odeconfig.n_ops_ind[i],
                                          odeconfig.n_ops_ptr[i], 1, ODE.y)
                                for i in odeconfig.c_const_inds]

                        _locals = locals()
                        # calculates the expectation values for time-dependent
                        # norm collapse operators
                        exec(_cy_col_expect_call_func, globals(), _locals)
                        n_dp = array(_locals['n_dp'])

                    # some Python function based collapse operators
                    elif odeconfig.tflag in array([2, 20, 22]):
                        n_dp = [cy_expect(odeconfig.n_ops_data[i],
                                          odeconfig.n_ops_ind[i],
                                          odeconfig.n_ops_ptr[i], 1, ODE.y)
                                for i in odeconfig.c_const_inds]
                        n_dp += [abs(odeconfig.c_funcs[i](
                                     ODE.t, odeconfig.c_func_args)) ** 2 *
                                 cy_expect(odeconfig.n_ops_data[i],
                                           odeconfig.n_ops_ind[i],
                                           odeconfig.n_ops_ptr[i], 1, ODE.y)
                                 for i in odeconfig.c_td_inds]
                        n_dp = array(n_dp)
                    # all constant collapse operators.
                    else:
                        n_dp = array([cy_expect(odeconfig.n_ops_data[i],
                                                odeconfig.n_ops_ind[i],
                                                odeconfig.n_ops_ptr[i],
                                                1, ODE.y)
                                      for i in range(odeconfig.c_num)])

                    # determine which operator does collapse
                    kk = cumsum(n_dp / sum(n_dp))
                    j = cinds[kk >= rand_vals[1]][0]
                    which_oper.append(j)  # record which operator did collapse
                    if j in odeconfig.c_const_inds:
                        state = spmv(odeconfig.c_ops_data[j],
                                     odeconfig.c_ops_ind[j],
                                     odeconfig.c_ops_ptr[j], ODE.y)
                    else:
                        if odeconfig.tflag in array([1, 11]):
                            _locals = locals()
                            # calculates the state vector for collapse by a
                            # time-dependent collapse operator
                            exec(_cy_col_spmv_call_func, globals(), _locals)
                            state = _locals['state']
                        else:
                            state = \
                                odeconfig.c_funcs[j](ODE.t,
                                                     odeconfig.c_func_args) * \
                                spmv(odeconfig.c_ops_data[j],
                                     odeconfig.c_ops_ind[j],
                                     odeconfig.c_ops_ptr[j], ODE.y)
                    state = state / norm(state, 2)
                    ODE.set_initial_value(state, ODE.t)
                    rand_vals = prng.rand(2)
            #-------------------------------------------------------

            ###--after while loop--####
            out_psi = ODE.y / norm(ODE.y, 2)
            if odeconfig.e_num == 0:
                if odeconfig.options.mc_avg:
                    mc_alg_out[k] = out_psi * out_psi.conj().T
                else:
                    mc_alg_out[k] = out_psi
            else:
                for jj in range(odeconfig.e_num):
                    mc_alg_out[jj][k] = cy_expect(odeconfig.e_ops_data[jj],
                                                  odeconfig.e_ops_ind[jj],
                                                  odeconfig.e_ops_ptr[jj],
                                                  odeconfig.e_ops_isherm[jj],
                                                  out_psi)

        # RETURN VALUES
        if odeconfig.e_num == 0:
            if odeconfig.options.mc_avg:
                mc_alg_out = array([Qobj(k, [odeconfig.psi0_dims[0],
                                             odeconfig.psi0_dims[0]],
                                            [odeconfig.psi0_shape[0],
                                             odeconfig.psi0_shape[0]],
                                         fast='mc-dm')
                                    for k in mc_alg_out])
            else:
                mc_alg_out = array([Qobj(k, odeconfig.psi0_dims,
                                         odeconfig.psi0_shape, fast='mc')
                                    for k in mc_alg_out])
            return nt, mc_alg_out, array(collapse_times), array(which_oper)
        else:
            return nt, mc_alg_out, array(collapse_times), array(which_oper)

    except Expection as e:
        print("failed to run _mc_alg_evolve: " + str(e))


def _time_remaining(st, ntraj, count, level):
    """
    Private function that determines, and prints, how much simulation
    time is remaining.
    """
    nwt = datetime.datetime.now()
    diff = ((nwt.day - st.day) * 86400 +
            (nwt.hour - st.hour) * (60 ** 2) +
            (nwt.minute - st.minute) * 60 +
            (nwt.second - st.second)) * (ntraj - count) / (1.0 * count)
    secs = datetime.timedelta(seconds=ceil(diff))
    dd = datetime.datetime(1, 1, 1) + secs
    time_string = "%02d:%02d:%02d:%02d" % \
        (dd.day - 1, dd.hour, dd.minute, dd.second)
    print(str(floor(count / float(ntraj) * 100)) + '%  (' + str(count) + '/' +
          str(ntraj) + ')' + '  Est. time remaining: ' + time_string)
    level += 0.1
    return level


def _mc_func_load(odeconfig):
    """Load cython functions"""

    global _cy_rhs_func
    global _cy_col_spmv_func, _cy_col_expect_func
    global _cy_col_spmv_call_func, _cy_col_expect_call_func

    if debug:
        print(inspect.stack()[0][3] + " in " + str(os.getpid()))

    if odeconfig.tflag in array([1, 10, 11]):
        # compile time-depdendent RHS code
        if odeconfig.tflag in array([1, 11]):
            code = compile('from ' + odeconfig.tdname +
                           ' import cyq_td_ode_rhs, col_spmv, col_expect',
                           '<string>', 'exec')
            exec(code, globals())
            _cy_rhs_func = cyq_td_ode_rhs
            _cy_col_spmv_func = col_spmv
            _cy_col_expect_func = col_expect
        else:
            code = compile('from ' + odeconfig.tdname +
                           ' import cyq_td_ode_rhs', '<string>', 'exec')
            exec(code, globals())
            _cy_rhs_func = cyq_td_ode_rhs

        # compile wrapper functions for calling cython spmv and expect
        if odeconfig.col_spmv_code:
            _cy_col_spmv_call_func = compile(
                odeconfig.col_spmv_code, '<string>', 'exec')

        if odeconfig.col_expect_code:
            _cy_col_expect_call_func = compile(
                odeconfig.col_expect_code, '<string>', 'exec')

    elif odeconfig.tflag == 0:
        _cy_rhs_func = cy_ode_rhs


def _mc_data_config(H, psi0, h_stuff, c_ops, c_stuff, args, e_ops,
                    options, odeconfig):
    """Creates the appropriate data structures for the monte carlo solver
    based on the given time-dependent, or indepdendent, format.
    """

    if debug:
        print(inspect.stack()[0][3])

    odeconfig.soft_reset()

    # take care of expectation values, if any
    if any(e_ops):
        odeconfig.e_num = len(e_ops)
        for op in e_ops:
            if isinstance(op, list):
                op = op[0]
            odeconfig.e_ops_data.append(op.data.data)
            odeconfig.e_ops_ind.append(op.data.indices)
            odeconfig.e_ops_ptr.append(op.data.indptr)
            odeconfig.e_ops_isherm.append(op.isherm)

        odeconfig.e_ops_data = array(odeconfig.e_ops_data)
        odeconfig.e_ops_ind = array(odeconfig.e_ops_ind)
        odeconfig.e_ops_ptr = array(odeconfig.e_ops_ptr)
        odeconfig.e_ops_isherm = array(odeconfig.e_ops_isherm)
    #----

    # take care of collapse operators, if any
    if any(c_ops):
        odeconfig.c_num = len(c_ops)
        for c_op in c_ops:
            if isinstance(c_op, list):
                c_op = c_op[0]
            n_op = c_op.dag() * c_op
            odeconfig.c_ops_data.append(c_op.data.data)
            odeconfig.c_ops_ind.append(c_op.data.indices)
            odeconfig.c_ops_ptr.append(c_op.data.indptr)
            # norm ops
            odeconfig.n_ops_data.append(n_op.data.data)
            odeconfig.n_ops_ind.append(n_op.data.indices)
            odeconfig.n_ops_ptr.append(n_op.data.indptr)
        # to array
        odeconfig.c_ops_data = array(odeconfig.c_ops_data)
        odeconfig.c_ops_ind = array(odeconfig.c_ops_ind)
        odeconfig.c_ops_ptr = array(odeconfig.c_ops_ptr)

        odeconfig.n_ops_data = array(odeconfig.n_ops_data)
        odeconfig.n_ops_ind = array(odeconfig.n_ops_ind)
        odeconfig.n_ops_ptr = array(odeconfig.n_ops_ptr)
    #----

    #--------------------------------------------
    # START CONSTANT H & C_OPS CODE
    #--------------------------------------------
    if odeconfig.tflag == 0:
        if odeconfig.cflag:
            odeconfig.c_const_inds = arange(len(c_ops))
            for c_op in c_ops:
                n_op = c_op.dag() * c_op
                H -= 0.5j * \
                    n_op  # combine Hamiltonian and collapse terms into one
        # construct Hamiltonian data structures
        if options.tidy:
            H = H.tidyup(options.atol)
        odeconfig.h_data = -1.0j * H.data.data
        odeconfig.h_ind = H.data.indices
        odeconfig.h_ptr = H.data.indptr
    #----

    #--------------------------------------------
    # START STRING BASED TIME-DEPENDENCE
    #--------------------------------------------
    elif odeconfig.tflag in array([1, 10, 11]):
        # take care of arguments for collapse operators, if any
        if any(args):
            for item in args.items():
                odeconfig.c_args.append(item[1])
        # constant Hamiltonian / string-type collapse operators
        if odeconfig.tflag == 1:
            H_inds = arange(1)
            H_tdterms = 0
            len_h = 1
            C_inds = arange(odeconfig.c_num)
            # find inds of time-dependent terms
            C_td_inds = array(c_stuff[2])
            # find inds of constant terms
            C_const_inds = setdiff1d(C_inds, C_td_inds)
            # extract time-dependent coefficients (strings)
            C_tdterms = [c_ops[k][1] for k in C_td_inds]
            # store indicies of constant collapse terms
            odeconfig.c_const_inds = C_const_inds
            # store indicies of time-dependent collapse terms
            odeconfig.c_td_inds = C_td_inds

            for k in odeconfig.c_const_inds:
                H -= 0.5j * (c_ops[k].dag() * c_ops[k])
            if options.tidy:
                H = H.tidyup(options.atol)
            odeconfig.h_data = [H.data.data]
            odeconfig.h_ind = [H.data.indices]
            odeconfig.h_ptr = [H.data.indptr]
            for k in odeconfig.c_td_inds:
                op = c_ops[k][0].dag() * c_ops[k][0]
                odeconfig.h_data.append(-0.5j * op.data.data)
                odeconfig.h_ind.append(op.data.indices)
                odeconfig.h_ptr.append(op.data.indptr)
            odeconfig.h_data = -1.0j * array(odeconfig.h_data)
            odeconfig.h_ind = array(odeconfig.h_ind)
            odeconfig.h_ptr = array(odeconfig.h_ptr)
            #--------------------------------------------
            # END OF IF STATEMENT
            #--------------------------------------------

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
            odeconfig.h_td_inds = arange(1, len_h)
            # if there are any collpase operators
            if odeconfig.c_num > 0:
                if odeconfig.tflag == 10:
                    # constant collapse operators
                    odeconfig.c_const_inds = arange(odeconfig.c_num)
                    for k in odeconfig.c_const_inds:
                        H[0] -= 0.5j * (c_ops[k].dag() * c_ops[k])
                    C_inds = arange(odeconfig.c_num)
                    C_tdterms = array([])
                else:
                    # some time-dependent collapse terms
                    C_inds = arange(odeconfig.c_num)
                    # find inds of time-dependent terms
                    C_td_inds = array(c_stuff[2])
                    # find inds of constant terms
                    C_const_inds = setdiff1d(C_inds, C_td_inds)
                    C_tdterms = [c_ops[k][1] for k in C_td_inds]
                    # extract time-dependent coefficients (strings)
                    # store indicies of constant collapse terms
                    odeconfig.c_const_inds = C_const_inds
                    # store indicies of time-dependent collapse terms
                    odeconfig.c_td_inds = C_td_inds
                    for k in odeconfig.c_const_inds:
                        H[0] -= 0.5j * (c_ops[k].dag() * c_ops[k])
            else:
                # set empty objects if no collapse operators
                C_const_inds = arange(odeconfig.c_num)
                odeconfig.c_const_inds = arange(odeconfig.c_num)
                odeconfig.c_td_inds = array([])
                C_tdterms = array([])
                C_inds = array([])

            # tidyup
            if options.tidy:
                H = array([H[k].tidyup(options.atol) for k in range(len_h)])
            # construct data sets
            odeconfig.h_data = [H[k].data.data for k in range(len_h)]
            odeconfig.h_ind = [H[k].data.indices for k in range(len_h)]
            odeconfig.h_ptr = [H[k].data.indptr for k in range(len_h)]
            for k in odeconfig.c_td_inds:
                odeconfig.h_data.append(-0.5j * odeconfig.n_ops_data[k])
                odeconfig.h_ind.append(odeconfig.n_ops_ind[k])
                odeconfig.h_ptr.append(odeconfig.n_ops_ptr[k])
            odeconfig.h_data = -1.0j * array(odeconfig.h_data)
            odeconfig.h_ind = array(odeconfig.h_ind)
            odeconfig.h_ptr = array(odeconfig.h_ptr)
            #--------------------------------------------
            # END OF ELSE STATEMENT
            #--------------------------------------------

        # set execuatble code for collapse expectation values and spmv
        col_spmv_code = ("state = _cy_col_spmv_func(j, ODE.t, " +
                         "odeconfig.c_ops_data[j], odeconfig.c_ops_ind[j], " +
                         "odeconfig.c_ops_ptr[j], ODE.y")
        col_expect_code = ("for i in odeconfig.c_td_inds: " +
                           "n_dp.append(_cy_col_expect_func(i, ODE.t, " +
                           "odeconfig.n_ops_data[i], " +
                           "odeconfig.n_ops_ind[i], " +
                           "odeconfig.n_ops_ptr[i], ODE.y")
        for kk in range(len(odeconfig.c_args)):
            col_spmv_code += ",odeconfig.c_args[" + str(kk) + "]"
            col_expect_code += ",odeconfig.c_args[" + str(kk) + "]"
        col_spmv_code += ")"
        col_expect_code += "))"

        odeconfig.col_spmv_code = col_spmv_code
        odeconfig.col_expect_code = col_expect_code

        # setup ode args string
        odeconfig.string = ""
        data_range = range(len(odeconfig.h_data))
        for k in data_range:
            odeconfig.string += ("odeconfig.h_data[" + str(k) +
                                 "],odeconfig.h_ind[" + str(k) +
                                 "],odeconfig.h_ptr[" + str(k) + "]")
            if k != data_range[-1]:
                odeconfig.string += ","
        # attach args to ode args string
        if len(odeconfig.c_args) > 0:
            for kk in range(len(odeconfig.c_args)):
                odeconfig.string += "," + "odeconfig.c_args[" + str(kk) + "]"
        #----

        name = "rhs" + str(odeconfig.cgen_num)
        odeconfig.tdname = name
        cgen = Codegen(H_inds, H_tdterms, odeconfig.h_td_inds, args,
                       C_inds, C_tdterms, odeconfig.c_td_inds, type='mc',
                       odeconfig=odeconfig)
        cgen.generate(name + ".pyx")

    #--------------------------------------------
    # END OF STRING TYPE TIME DEPENDENT CODE
    #--------------------------------------------

    #--------------------------------------------
    # START PYTHON FUNCTION BASED TIME-DEPENDENCE
    #--------------------------------------------
    elif odeconfig.tflag in array([2, 20, 22]):

        # take care of Hamiltonian
        if odeconfig.tflag == 2:
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
            odeconfig.h_funcs = array([H[k][1] for k in H_td_inds])
            odeconfig.h_func_args = args
            Htd = array([H[k][0] for k in H_td_inds])
            odeconfig.h_td_inds = arange(len(Htd))
            H = sum(H[k] for k in H_const_inds)

        # take care of collapse operators
        C_inds = arange(odeconfig.c_num)
        # find inds of time-dependent terms
        C_td_inds = array(c_stuff[1])
        # find inds of constant terms
        C_const_inds = setdiff1d(C_inds, C_td_inds)
        # store indicies of constant collapse terms
        odeconfig.c_const_inds = C_const_inds
        # store indicies of time-dependent collapse terms
        odeconfig.c_td_inds = C_td_inds
        odeconfig.c_funcs = zeros(odeconfig.c_num, dtype=FunctionType)
        for k in odeconfig.c_td_inds:
            odeconfig.c_funcs[k] = c_ops[k][1]
        odeconfig.c_func_args = args

        # combine constant collapse terms with constant H and construct data
        for k in odeconfig.c_const_inds:
            H -= 0.5j * (c_ops[k].dag() * c_ops[k])
        if options.tidy:
            H = H.tidyup(options.atol)
            Htd = array([Htd[j].tidyup(options.atol)
                         for j in odeconfig.h_td_inds])
        # setup constant H terms data
        odeconfig.h_data = -1.0j * H.data.data
        odeconfig.h_ind = H.data.indices
        odeconfig.h_ptr = H.data.indptr

        # setup td H terms data
        odeconfig.h_td_data = array(
            [-1.0j * Htd[k].data.data for k in odeconfig.h_td_inds])
        odeconfig.h_td_ind = array(
            [Htd[k].data.indices for k in odeconfig.h_td_inds])
        odeconfig.h_td_ptr = array(
            [Htd[k].data.indptr for k in odeconfig.h_td_inds])
        #--------------------------------------------
        # END PYTHON FUNCTION BASED TIME-DEPENDENCE
        #--------------------------------------------

    #--------------------------------------------
    # START PYTHON FUNCTION BASED HAMILTONIAN
    #--------------------------------------------
    elif odeconfig.tflag == 3:
        # take care of Hamiltonian
        odeconfig.h_funcs = H
        odeconfig.h_func_args = args

        # take care of collapse operators
        odeconfig.c_const_inds = arange(odeconfig.c_num)
        odeconfig.c_td_inds = array([])  # find inds of time-dependent terms
        if len(odeconfig.c_const_inds) > 0:
            H = 0
            for k in odeconfig.c_const_inds:
                H -= 0.5j * (c_ops[k].dag() * c_ops[k])
            if options.tidy:
                H = H.tidyup(options.atol)
            odeconfig.h_data = -1.0j * H.data.data
            odeconfig.h_ind = H.data.indices
            odeconfig.h_ptr = H.data.indptr


def _mc_dm_avg(psi_list):
    """
    Private function that averages density matrices in parallel
    over all trajectores for a single time using parfor.
    """
    ln = len(psi_list)
    dims = psi_list[0].dims
    shape = psi_list[0].shape
    out_data = mean([psi.data for psi in psi_list])
    return Qobj(out_data, dims=dims, shape=shape, fast='mc-dm')
