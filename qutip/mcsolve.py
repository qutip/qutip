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
import scipy.sparse as sp
from scipy.integrate import ode
from scipy.integrate._ode import zvode

from types import FunctionType, BuiltinFunctionType
from functools import partial
from qutip.fastsparse import csr2fast
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.parallel import parfor, parallel_map, serial_map
from qutip.cy.mcsolve import CyMcOde, CyMcOdeDiag
from qutip.cy.spconvert import dense1D_to_fastcsr_ket
from qutip.sesolve import sesolve
from qutip.solver import (Options, Result, ExpectOps,
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

class qutip_zvode(zvode):
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r

def mcsolve(H, psi0, tlist, c_ops=[], e_ops=[], ntraj=0,
            args={}, options=Options(),
            progress_bar=True, map_func=parallel_map, map_kwargs={},
            _safe_mode=True, _exp=False):
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

    if options.rhs_reuse and not isinstance(H, SolverSystem):
        # TODO: deprecate when going to class based solver.
        if "mcsolve" in solver_safe:
            # print(" ")
            H = solver_safe["mcsolve"]
        else:
            pass
            # raise Exception("Could not find the Hamiltonian to reuse.")

    if not ntraj:
        ntraj = options.ntraj

    if len(c_ops) == 0 and not options.rhs_reuse:
        warn("No c_ops, using sesolve")
        return sesolve(H, psi0, tlist, e_ops=e_ops, args=args,
                       options=options, progress_bar=progress_bar,
                       _safe_mode=_safe_mode)

    try:
        num_traj = int(ntraj)
    except:
        num_traj = max(ntraj)

    # set the physics
    if not psi0.isket:
        raise Exception("Initial state must be a state vector.")

    # load monte carlo class
    mc = _MC(options, _exp)


    if isinstance(H, SolverSystem):
        mc.ss = H
    else:
        mc.make_system(H, c_ops, tlist, args, options)

    mc.reset(tlist[0], psi0)

    mc.set_e_ops(e_ops)

    if options.seeds is not None:
        mc.seed(num_traj, options.seeds)

    if _safe_mode:
        mc.run_test()

    # Run the simulation
    mc.run(num_traj=num_traj, tlist=tlist,
           progress_bar=progress_bar,
           map_func=map_func, map_kwargs=map_kwargs)

    return mc.get_result(ntraj)


# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class _MC():
    """
    Private class for solving Monte Carlo evolution from mcsolve
    """
    def __init__(self, options=Options(), _exp=False):
        self.options = options
        self.ss = None
        self.tlist = None
        self.e_ops = None
        self.ran = False
        self.psi0 = None
        self.seeds = []
        self.t = 0.
        self.num_traj = 0
        self.args_col = None

        self._psi_out = []
        self._expect_out = []
        self._collapse = []
        self._ss_out = []

        # Flag
        self._experimental = _exp

    def reset(self, t=0., psi0=None):
        if psi0 is not None:
            self.psi0 = psi0
        if self.psi0 is not None:
            self.initial_vector = self.psi0.full().ravel("F")
            if self.ss is not None and self.ss.type == "Diagonal":
                self.initial_vector = np.dot(self.ss.Ud, self.initial_vector)

        self.t = t
        self.ran = False
        self._psi_out = []
        self._expect_out = []
        self._collapse = []
        self._ss_out = []

    def seed(self, ntraj, seeds=[]):
        # setup seeds array
        np.random.seed()
        try:
            seed = int(seeds)
            np.random.seed(seed)
            seeds = []
        except TypeError:
            pass

        if len(seeds) < ntraj:
            self.seeds = seeds + list(randint(0, 2**31-1, size=ntraj-len(seeds)))
        else:
            self.seeds = seeds[:ntraj]

    def make_system(self, H, c_ops, tlist=None, args={}, options=None):
        if options is None:
            options = self.options
        else:
            self.options = options
        var = _collapse_args(args)

        ss = SolverSystem()
        ss.td_c_ops = []
        ss.td_n_ops = []
        ss.args = args
        ss.col_args = var
        for c in c_ops:
            cevo = QobjEvo(c, args, tlist)
            cdc = cevo._cdc()
            cevo.compile()
            cdc.compile()
            ss.td_c_ops.append(cevo)
            ss.td_n_ops.append(cdc)

        try:
            H_td = QobjEvo(H, args, tlist)
            H_td *= -1j
            for c in ss.td_n_ops:
                H_td += -0.5 * c
            if options.rhs_with_state:
                H_td._check_old_with_state()
            H_td.compile()
            ss.H_td = H_td
            ss.makefunc = _qobjevo_set
            ss.set_args = _qobjevo_args
            ss.type = "QobjEvo"

        except:
            ss.h_func = H
            ss.Hc_td = -0.5 * sum(ss.td_n_ops)
            ss.Hc_td.compile()
            ss.with_state = options.rhs_with_state
            ss.makefunc = _func_set
            ss.set_args = _func_args
            ss.type = "callback"

        solver_safe["mcsolve"] = ss
        self.ss = ss
        self.reset()

    def set_e_ops(self, e_ops=[]):
        if e_ops:
            self.e_ops = ExpectOps(e_ops)
        else:
            self.e_ops = ExpectOps([])

        ss = self.ss
        if ss is not None and ss.type == "Diagonal" and not self.e_ops.isfunc:
            e_op = [Qobj(ss.Ud @ e.full() @ ss.U, dims=e.dims) for e in self.e_ops.e_ops]
            self.e_ops = ExpectOps(e_ops)

        if not self.e_ops:
            self.options.store_states = True

    def run_test(self):
        try:
            for c_op in self.ss.td_c_ops:
                c_op.mul_vec(0, self.psi0)
        except:
            raise Exception("c_ops are not consistant with psi0")

        if self.ss.type == "QobjEvo":
            try:
                self.ss.H_td.mul_vec(0., self.psi0)
            except:
                raise Exception("Error calculating H")
        else:
            try:
                rhs, ode_args = self.ss.makefunc(ss)
                rhs(t, self.psi0.full().ravel(), ode_args)
            except:
                raise Exception("Error calculating H")

    def run(self, num_traj=0, psi0=None, tlist=None,
            args={}, e_ops=None, options=None,
            progress_bar=True,
            map_func=parallel_map, map_kwargs={}):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 4 situation for run:
        # - first run
        # - change parameters
        # - add  trajectories
        #       (self.add_traj)      Not Implemented
        # - continue from the last time and states
        #       (self.continue_runs) Not Implemented
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        options = options if options is not None else self.options

        if self.ran and tlist[0] == self.t:
            # psi0 is ignored since we restart from a
            # different states for each trajectories
            self.continue_runs(num_traj, tlist, args, e_ops, options,
                               progress_bar, map_func, map_kwargs)
            return

        if args != self.ss.args:
            self.ss.set_args(self.ss, args)
            self.reset()

        if e_ops and e_ops != self.e_ops:
            self.set_e_ops(e_ops)
            self.reset()

        if psi0 is not None and psi0 != self.psi0:
            self.psi0 = psi0
            self.reset()

        tlist = np.array(tlist)
        if tlist is not None and np.all(tlist != self.tlist):
            self.tlist = tlist
            self.reset()

        if self.ran:
            if options.store_states and self._psi_out[0].shape[0] == 1:
                self.reset()
            else:
                # if not reset here, add trajectories
                self.add_traj(num_traj, progress_bar, map_func, map_kwargs)
                return

        if not num_traj:
            num_traj = options.ntraj

        if options.num_cpus == 1 or num_traj == 1:
            map_func = serial_map

        if len(self.seeds) != num_traj:
            self.seed(num_traj, self.seeds)

        if not progress_bar:
            progress_bar = BaseProgressBar()
        elif progress_bar is True:
            progress_bar = TextProgressBar()

        # set arguments for input to monte carlo
        map_kwargs = {'progress_bar': progress_bar,
                      'num_cpus': options.num_cpus}
        map_kwargs.update(map_kwargs)

        if self.e_ops is None:
            self.set_e_ops()

        if self.ss.type == "Diagonal":
            results = map_func(self._single_traj_diag, list(range(num_traj)), **map_kwargs)
        else:
            results = map_func(self._single_traj, list(range(num_traj)), **map_kwargs)

        self.t = self.tlist[-1]
        self.num_traj = num_traj
        self.ran = True

        for result in results:
            state_out, ss_out, expect, collapse = result
            self._psi_out.append(state_out)
            self._ss_out.append(ss_out)
            self._expect_out.append(expect)
            self._collapse.append(collapse)
        self._psi_out = np.stack(self._psi_out)
        self._ss_out = np.stack(self._ss_out)

    def add_traj(self, num_traj,
                 progress_bar=True,
                 map_func=parallel_map, map_kwargs={}):
        raise NotImplementedError

    def continue_runs(self, num_traj, tlist, args={}, e_ops=[], options=None,
                      progress_bar=True,
                      map_func=parallel_map, map_kwargs={}):
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # results functions
    # --------------------------------------------------------------------------
    @property
    def states(self):
        dims = self.psi0.dims[0]
        len_ = self._psi_out.shape[2]
        if self._psi_out.shape[1] == 1:
            dm_t = np.zeros((len_, len_), dtype=complex)
            for i in range(self.num_traj):
                vec = self._psi_out[i,0,:] # .reshape((-1,1))
                dm_t += np.outer(vec, vec.conj())
            return Qobj(dm_t/self.num_traj, dims=[dims, dims])
        else:
            states = np.empty((len(self.tlist)), dtype=object)
            for j in range(len(self.tlist)):
                dm_t = np.zeros((len_, len_), dtype=complex)
                for i in range(self.num_traj):
                    vec = self._psi_out[i,j,:] # .reshape((-1,1))
                    dm_t += np.outer(vec, vec.conj())
                states[j] = Qobj(dm_t/self.num_traj, dims=[dims, dims])
            return states

    @property
    def final_state(self):
        dims = self.psi0.dims[0]
        len_ = self._psi_out.shape[2]
        dm_t = np.zeros((len_, len_), dtype=complex)
        for i in range(self.num_traj):
            vec = self._psi_out[i,-1,:]
            dm_t += np.outer(vec, vec.conj())
        return Qobj(dm_t/self.num_traj, dims=[dims, dims])

    @property
    def runs_final_states(self):
        dims = self.psi0.dims[0]
        psis = np.empty((self.num_traj), dtype=object)
        for i in range(self.num_traj):
            psis[i] = Qobj(dense1D_to_fastcsr_ket(self._psi_out[i,-1,:]),
                           dims=dims, fast='mc')
        return psis

    @property
    def expect(self):
        return self.expect_traj_avg()

    @property
    def runs_expect(self):
        return [expt.finish() for expt in self._expect_out]

    def expect_traj_avg(self, ntraj=0):
        if not ntraj:
            ntraj = len(self._expect_out)
        expect = np.stack([expt.raw_out for expt in self._expect_out[:ntraj]])
        expect = np.mean(expect, axis=0)

        result = []
        for ii in range(self.e_ops.e_num):
            if self.e_ops.e_ops_isherm[ii]:
                result.append(np.real(expect[ii, :]))
            else:
                result.append(expect[ii, :])

        if self.e_ops.e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(self.e_ops.e_ops_dict.keys())}
        return result

    @property
    def steady_state(self):
        if self._ss_out is not None:
            dims = self.psi0.dims[0]
            len_ = self.psi0.shape[0]
            return Qobj(np.mean(self._ss_out, axis=0),
                            [dims, dims], [len_, len_])
        # TO-DO rebuild steady_state from _psi_out if needed
        # elif self._psi_out is not None:
        #     return sum(self.state_average) / self.num_traj
        else:
            return None

    @property
    def runs_states(self):
        dims = self.psi0.dims
        psis = np.empty((self.num_traj, len(self.tlist)), dtype=object)
        for i in range(self.num_traj):
            for j in range(len(self.tlist)):
                psis[i,j] = Qobj(dense1D_to_fastcsr_ket(self._psi_out[i,j,:]),
                                 dims=dims, fast='mc')
        return psis

    @property
    def collapse(self):
        return self._collapse

    @property
    def collapse_times(self):
        out = []
        for col_ in self._collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[0])
            out.append( np.array(col) )
        return out
        return [np.array(list(zip(*col_))[0]) for col_ in self._collapse]

    @property
    def collapse_which(self):
        out = []
        for col_ in self._collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[1])
            out.append( np.array(col) )
        return out
        return [np.array(list(zip(*col_))[1]) for col_ in self._collapse]

    def get_result(self, ntraj=[]):
        # Store results in the Result object
        if not ntraj:
            ntraj = [self.num_traj]
        elif not isinstance(ntraj, list):
            ntraj = [ntraj]

        output = Result()
        output.solver = 'mcsolve'
        output.seeds = self.seeds

        options = self.options
        output.options = options

        if options.steady_state_average:
            output.states = self.steady_state
        elif options.average_states and options.store_states:
            output.states = self.states
        elif options.store_states:
            output.states = self.runs_states

        if options.store_final_state:
            if not self._experimental or options.average_states:
                output.final_state = self.final_state
            else:
                output.final_state = self.runs_final_states

        if options.average_expect:
            output.expect = [self.expect_traj_avg(n) for n in ntraj]
            if len(output.expect) == 1:
                output.expect = output.expect[0]
        else:
            output.expect = self.runs_expect

        # simulation parameters
        output.times = self.tlist
        output.num_expect = self.e_ops.e_num
        output.num_collapse = len(self.ss.td_c_ops)
        output.ntraj = self.num_traj
        output.col_times = self.collapse_times
        output.col_which = self.collapse_which

        return output

    # --------------------------------------------------------------------------
    # single-trajectory for monte carlo
    # --------------------------------------------------------------------------
    def _single_traj(self, nt):
        """
        Monte Carlo algorithm returning state-vector or expectation values
        at times tlist for a single trajectory.
        """
        # SEED AND RNG AND GENERATE
        prng = RandomState(self.seeds[nt])
        opt = self.options

        # set initial conditions
        ss = self.ss
        tlist = self.tlist
        e_ops = self.e_ops.copy()
        opt = self.options
        rhs, ode_args = self.ss.makefunc(ss)
        ODE = self._build_integration_func(rhs, ode_args, opt)
        ODE.set_initial_value(self.initial_vector, tlist[0])
        e_ops.init(tlist)

        cymc = CyMcOde(ss, opt)
        states_out, ss_out, collapses = cymc.run_ode(ODE, tlist, e_ops, prng)

        # Run at end of mc_alg function
        # -----------------------------
        if opt.steady_state_average:
            ss_out /= float(len(tlist))

        return (states_out, ss_out, e_ops, collapses)

    def _build_integration_func(self, rhs, ode_args, opt):
        """
        Create the integration function while fixing the parameters
        """
        ODE = ode(rhs)
        if ode_args:
            ODE.set_f_params(ode_args)
        # initialize ODE solver for RHS
        ODE.set_integrator('zvode', method="adams")
        ODE._integrator = qutip_zvode(
            method=opt.method, order=opt.order, atol=opt.atol,
            rtol=opt.rtol, nsteps=opt.nsteps, first_step=opt.first_step,
            min_step=opt.min_step, max_step=opt.max_step)
        return ODE

    # --------------------------------------------------------------------------
    # In development diagonalize the Hamiltonian before solving
    # Same seeds give same evolution
    # 3~5 time faster
    # constant system only.
    # --------------------------------------------------------------------------
    def make_diag_system(self, H, c_ops):
        ss = SolverSystem()
        ss.td_c_ops = []
        ss.td_n_ops = []

        H_ = H.copy()
        H_ *= -1j
        for c in c_ops:
            H_ += -0.5 * c.dag() * c

        w, v = np.linalg.eig(H_.full())
        arg = np.argsort(np.abs(w))
        eig = w[arg]
        U = v.T[arg].T
        Ud = U.T.conj()

        for c in c_ops:
            c_diag = Qobj(Ud @ c.full() @ U, dims=c.dims)
            cevo = QobjEvo(c_diag)
            cdc = cevo._cdc()
            cevo.compile()
            cdc.compile()
            ss.td_c_ops.append(cevo)
            ss.td_n_ops.append(cdc)

        ss.H_diag = eig
        ss.Ud = Ud
        ss.U = U
        ss.args = {}
        ss.type = "Diagonal"
        solver_safe["mcsolve"] = ss

        if self.e_ops and not self.e_ops.isfunc:
            e_op = [Qobj(Ud @ e.full() @ U, dims=e.dims) for e in self.e_ops.e_ops]
            self.e_ops = ExpectOps(e_ops)
        self.ss = ss
        self.reset()

    def _single_traj_diag(self, nt):
        """
        Monte Carlo algorithm returning state-vector or expectation values
        at times tlist for a single trajectory.
        """
        # SEED AND RNG AND GENERATE
        prng = RandomState(self.seeds[nt])
        opt = self.options

        ss = self.ss
        tlist = self.tlist
        e_ops = self.e_ops.copy()
        opt = self.options
        e_ops.init(tlist)

        cymc = CyMcOdeDiag(ss, opt)
        states_out, ss_out, collapses = cymc.run_ode(self.initial_vector, tlist,
                                                     e_ops, prng)

        if opt.steady_state_average:
            ss_out = ss.U @ ss_out @ ss.Ud
        states_out = np.inner(ss.U, states_out).T
        if opt.steady_state_average:
            ss_out /= float(len(tlist))
        return (states_out, ss_out, e_ops, collapses)

# -----------------------------------------------------------------------------
# CODES FOR PYTHON FUNCTION BASED TIME-DEPENDENT RHS
# -----------------------------------------------------------------------------
def _qobjevo_set(ss, psi0=None, args={}, opt=None):
    if args:
        self.set_args(args)
    rhs = ss.H_td.compiled_qobjevo.mul_vec
    return rhs, ()

def _qobjevo_args(ss, args):
    var = _collapse_args(args)
    ss.col_args = var
    ss.args = args
    ss.H_td.arguments(args)
    for c in ss.td_c_ops:
        c.arguments(args)
    for c in ss.td_n_ops:
        c.arguments(args)

def _func_set(HS, psi0=None, args={}, opt=None):
    if args:
        self.set_args(args)
    else:
        args = ss.args
    if ss.with_state:
        rhs = _funcrhs
    else:
        rhs = _funcrhs_with_state
    return rhs, (ss.h_func, ss.Hc_td, args)

def _func_args(ss, args):
    var = _collapse_args(args)
    ss.col_args = var
    ss.args = args
    for c in ss.td_c_ops:
        c.arguments(args)
    for c in ss.td_n_ops:
        c.arguments(args)
    return rhs, (ss.h_func, ss.Hc_td, args)


# RHS of ODE for python function Hamiltonian
def _funcrhs(t, psi, h_func, Hc_td, args):
    h_func_data = -1.0j * h_func(t, args).data
    h_func_term = h_func_data * psi
    return h_func_term + Hc_td.mul_vec(t, psi)

def _funcrhs_with_state(t, psi, h_func, Hc_td, args):
    h_func_data = - 1.0j * h_func(t, psi, args).data
    h_func_term = h_func_data * psi
    return h_func_term + Hc_td.mul_vec(t, psi)

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

def _collapse_args(args):
    to_rm = ""
    for k in args:
        if "=" in k and k.split("=")[1] == "collapse":
            to_rm.append(k)
            var = k.split("=")[0]
            if isinstance(args[k], list):
                list_ = args[k]
            else:
                list_ = []
            to_rm = k
            break
    if to_rm:
        del args[k]
        args[var] = list_
        return var
    else:
        return ""
