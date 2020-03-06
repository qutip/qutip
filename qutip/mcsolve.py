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

__all__ = ['mcsolve', 'McSolver']

import os
import numpy as np
from numpy.random import RandomState, randint
import scipy.sparse as sp
from scipy.integrate import ode


from types import FunctionType, BuiltinFunctionType
from functools import partial
from qutip.fastsparse import csr2fast
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.qobjevo_maker import qobjevo_maker
from qutip.parallel import parfor, parallel_map, serial_map
#from qutip.cy.mcsolve import CyMcOde, CyMcOdeDiag
from qutip.cy.spconvert import dense1D_to_fastcsr_ket
from qutip.sesolve import sesolve
from qutip.solver import (Options, Result, ExpectOps, Solver,
                          solver_safe, SolverSystem)
from qutip.settings import debug
from qutip.ui.progressbar import TextProgressBar, BaseProgressBar
import qutip.settings

from .mcsolverode import McOdeScipyZvode, McOdeQutipDiag

if debug:
    import inspect

#
# Internal, global variables for storing references to dynamically loaded
# cython functions

# Todo: use real warning
def warn(text):
    print(text)


def mcsolve(H, psi0, tlist, c_ops, e_ops=[], ntraj=0,
            args={}, options=None, progress_bar=True, seeds=None,
            parallel=True, map_func=None, map_kwargs={}, _safe_mode=True):
    """Monte Carlo evolution of a state vector :math:`|\psi \\rangle` for a
    given Hamiltonian and sets of collapse operators, and possibly, operators
    for calculating expectation values. Options for the underlying ODE solver
    are given by the Options class.

    mcsolve supports time-dependent Hamiltonians and collapse operators using
    either Python functions or list with functions or string to represent
    time-dependent coefficients.

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

    The Hamiltonian and collapse operators can also be represented a function,
    but this is usually slower than the list format::
        def H(t, args):
            return H0 + H1 * sin(args['w']*t)

        def C1(t, args):
            return c1 * exp(-args['a']*t)

        c_ops = [C0, C1]

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

    parallel : bool
        True to run in parallel, map_func has priority.

    seeds : int, list
        list of seeds for random number for each trajectory or unique seed for
        all generators.

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
    if isinstance(c_ops, (Qobj, QobjEvo)): c_ops = [c_ops]

    if len(c_ops) == 0:
        warn("No c_ops, using sesolve")
        return sesolve(H, psi0, tlist, e_ops=e_ops, args=args,
                       options=options, progress_bar=progress_bar,
                       _safe_mode=_safe_mode)
    if options is None: options=Options()
    if ntraj == 0: ntraj =  options.ntraj
    nums_traj = ntraj if isinstance(ntraj, list) else [ntraj]
    if seeds is None: seeds = options.seeds
    if map_func is parallel_map:
        parallel = True
    elif map_func is serial_map:
        parallel = False

    if options is not None and options.rhs_reuse:
        raise DeprecationWarning
        warn("'rhs_reuse' of Options will be deprecated. "
             "Use the object interface of instead: 'McSolver'")
        if "mcsolve" in solver_safe:
            solver = solver_safe["mcsolve"]
            if e_ops: solver.e_ops = e_ops
            if options is not None: solver.options = options
            solver.progress_bar = progress_bar
        else:
            solver = McSolver(H, c_ops, args, psi0, tlist, e_ops,
                              options, progress_bar, parallel)
            solver_safe["mcsolve"] = solver
    else:
        solver = McSolver(H, c_ops, args, psi0, tlist, e_ops,
                          options, progress_bar, parallel)

    # Run the simulation
    return solver.run(psi0, tlist, nums_traj, e_ops, args, seeds,
                      _safe_mode=_safe_mode)


class McSolver(Solver):
    """Monte Carlo solver for a given Hamiltonian and sets of collapse
    operators. Options for the underlying ODE solver are given by the Options
    class.

    McSolver supports time-dependent Hamiltonians and collapse operators using
    either Python functions of list based format.

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

    The Hamiltonian and collapse operators can also be represented a function,
    but this is usually slower than the list format::
        def H(t, args):
            return H0 + H1 * sin(args['w']*t)

        def C1(t, args):
            return c1 * exp(-args['a']*t)

        c_ops = [C0, C1]

    Parameters
    ----------
    H : :class:`qutip.Qobj`, ``list``
        System Hamiltonian.

    c_ops : :class:`qutip.Qobj`, ``list``
        single collapse operator or a ``list`` of collapse operators.

    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    psi0 : :class:`qutip.Qobj`
        Initial state vector

    tlist : array_like
        Times at which results are recorded.

    e_ops : :class:`qutip.Qobj`, ``list``
        single operator as Qobj or ``list`` or equivalent of Qobj operators
        for calculating expectation values.

    options : Options
        Instance of ODE solver options.

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation. Set to None to disable the
        progress bar.

    parallel : bool
        True to run in parallel, map_func has priority.

    outtype: [Qobj, dense, sparse]
        Type of output states.

    Attributes
    ----------
    options: :class:`qutip.Options`
        options for the evolution, some to pass to scipy's solver, some for
        the output control.

    progress_bar: :class:`qutip.BaseProgressBar`
        How to show the evolution's progress.

    states: list of :class:`qutip.Qobj`
        States computed during the last run averaged over trajectories as
        density matrix.

    final_state: :class:`qutip.Qobj`
        Averaged stated at last time as a density matrix.

    runs_final_states: list of :class:`qutip.Qobj`
        kets at the last time for each trajectories

    runs_states: list of list of :class:`qutip.Qobj`
        kets for all trajectories and all times.

    steady_state: :class:`qutip.Qobj`
        estimation of the steady_state of the system.

    expect:
        expectation values averaged over each trajectories

    runs_expect: list of list (or dict) of np.array
        expectation values for each trajectories, e_ops and times.
        # Todo? Could be one np.array if complex even if e_ops is hermitian

    collapse: list of list of tuple
        List of collapses as pairs (times, which c_op) for each trajectories.

    collapse_times: list of np.array
        times of each collapse for each trajectories

    collapse_which: list of np.array
        Which collapse operator correspond to each collapse for
        all trajectories.

    measurement:
        photocurrent corresponding to each c_ops.

    Methods
    -------
    run(psi0, tlist, num_traj, e_ops=None, args=None, seed=None)
        Compute num_traj trajectories returning a Result object containing.
        the expectation values if e_ops is defined. If not defined or
        options.store_states is set, states are also included in results.
    get_result()
        Return the results of the last run.
    expect_traj_avg(ntraj=0)
        Return the expectation values averaged over ntraj. 0 means all
        trajectories of the last run.

    """
    def __init__(self, H, c_ops, args=None,
                 psi0=None, tlist=None, e_ops=None, options=None,
                 progress_bar=None, parallel=True, outtype=Qobj):
        self.progress_bar = progress_bar
        self.options = options
        self._outtype = outtype
        self._e_ops = e_ops
        self._args = args

        self.H = qobjevo_maker(H, self._args, tlist=tlist,
                               e_ops=e_ops, state=psi0)
        self.c_ops = [qobjevo_maker(op, self._args, tlist=tlist,
                                    e_ops=e_ops, state=psi0) for op in c_ops]

        self._feedback = (self.H.feedback and
                          all(c.feedback for c in self.c_ops))
        self._const = self.H.const and all(c.const for c in self.c_ops)
        self._dims = self.H.cte.dims
        self._psi0 = None
        self._psi = None
        self._solver = None
        self._parallel = parallel

        if psi0 is not None:
            self._set_psi(psi0)

    def seed(self, ntraj, seeds=None):
        # setup seeds array
        if seeds is None:
            np.random.seed()
            seeds = []
        elif isinstance(seeds, int):
            np.random.seed(seed)
            seeds = []
        if len(seeds) < ntraj:
            self.seeds = seeds + list(randint(0, 2**31-1,
                                              size=ntraj-len(seeds)))
        else:
            self.seeds = seeds[:ntraj]

    def _get_solver(self):
        solver = self.options.solver
        if self._solver and self._solver.name == solver:
            self._solver.update_args(self._args)
            return self._solver
        if solver in ["scipy_ivp_mc", "scipy_ivp", "ivp"]:
            self.options.solver = "scipy_ivp_mc"
            return McOdeScipyIVP(self.H, self.c_ops,
                                 self.options, self._parallel,
                                 self.progress_bar)
        elif solver in ["qutip_diag_mc", "diagonal"]:
            if not self._const:
                raise ValueError("diagonal solver can only be used "
                                 "for constant system")
            self.options.solver = "qutip_diag_mc"
            return McOdeQutipDiag(self.H, self.c_ops,
                                  self.options, self._parallel,
                                  self.progress_bar)
        elif solver in ["scipy_zvode_mc", "zvode", "scipy_zvode"]:
            self.options.solver = "scipy_zvode_mc"
            return McOdeScipyZvode(self.H, self.c_ops,
                                   self.options, self._parallel,
                                   self.progress_bar)
        else:
            raise ValueError("Invalid solver")

    def _set_psi(self, psi0):
        self._psi_out = []
        self._expect_out = []
        self._collapse = []
        self._ss_out = []
        self._psi0 = psi0
        self._state_dims = psi0.dims
        self._state_shape = psi0.shape
        self._psi = psi0.full().ravel("F")
        self._solver = self._get_solver()

    def _check(self, psi0):
        if not psi0.isket and (self._dims[1] == psi0.dims[0]):
            raise ValueError("The dimension of psi0 does not "
                             "fit the Hamiltonian")

    def run(self, psi0, tlist, num_traj, e_ops=None,
            args=None, seed=None, _safe_mode=False):
        if args is not None:
            self._args = args
        opt = self.options
        old_ss = opt.store_states
        e_ops = ExpectOps(e_ops)
        if not e_ops:
            opt.store_states = True
        self.set(psi0, tlist[0])
        if _safe_mode: self._check(self._psi0)

        self.tlist = tlist
        self.nums_traj = num_traj if isinstance(num_traj, list) else [num_traj]
        self.num_traj = max(self.nums_traj)
        self.seed(self.num_traj, seed)
        results = self._solver.run(self._psi, self.num_traj, self.seeds, tlist,
                                  {}, e_ops)

        for result in results:
            state_out, ss_out, expect, collapse = result
            self._psi_out.append(state_out)
            self._ss_out.append(ss_out)
            self._expect_out.append(expect)
            self._collapse.append(collapse)
        self._psi_out = np.stack(self._psi_out)
        self._ss_out = np.stack(self._ss_out)
        self._expect_out = np.stack(self._expect_out)
        self._expect = e_ops
        opt.store_states = old_ss
        return self.get_result()

    def step(self, t, args=None, e_ops=[]):
        raise NotImplementedError

    def set(self, psi0=None, t0=0):
        self.t = t0
        psi0 = psi0 if psi0 is not None else self._psi0
        self._set_psi(psi0)

    def _state2dm(self, t_idx):
        dims = self._state_dims[0]
        len_ = self._state_shape[0]
        dm_t = np.einsum("ri,rj->ij", self._psi_out[:,t_idx,:],
                         self._psi_out[:,t_idx,:].conj())
        dm_t = self.transform(dm_t/self.num_traj,
                              dims=[dims, dims], shape=(len_, len_))
        return dm_t

    @property
    def states(self):
        if self._psi_out.shape[1] == 1:
            return self._state2dm(-1)
        else:
            return [self._state2dm(j) for j in range(len(self.tlist))]

    @property
    def final_state(self):
        return self._state2dm(-1)

    @property
    def runs_final_states(self):
        return [self.transform(self._psi_out[i, -1, :])
                for i in range(self.num_traj)]

    @property
    def runs_states(self):
        return [[self.transform(self._psi_out[i, t, :])
                 for t in range(len(self.tlist))]
                 for i in range(self.num_traj)]

    @property
    def steady_state(self):
        if len(self._ss_out):
            dm_t = np.mean(self._ss_out, axis=0)
        else:
            dm_t = np.einsum("rti,rtj->ij", self._psi_out, self._psi_out.conj())
            dm_t /= len(self.tlist) * len(self.num_traj)
        dims = self._state_dims[0]
        len_ = self._state_shape[0]
        return self.transform(dm_t, dims=[dims, dims], shape=(len_, len_))

    @property
    def expect(self):
        return self.expect_traj_avg()

    @property
    def runs_expect(self):
        result = []
        for jj in range(self.num_traj):
            res_traj = []
            for ii in range(self._expect.e_num):
                if self._expect.e_ops_isherm[ii]:
                    res_traj.append(np.real(self._expect_out[jj, ii, :]))
                else:
                    res_traj.append(self._expect_out[jj, ii, :])
            result.append(res_traj)

        if self._expect.e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(self._expect.e_ops_dict.keys())}
        return result

    def expect_traj_avg(self, ntraj=0):
        if not ntraj:
            ntraj = self.num_traj
        expect = np.mean(self._expect_out[:ntraj,:,:], axis=0)

        result = []
        for ii in range(self._expect.e_num):
            if self._expect.e_ops_isherm[ii]:
                result.append(np.real(expect[ii, :]))
            else:
                result.append(expect[ii, :])

        if self._expect.e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(self._expect.e_ops_dict.keys())}
        return result

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

    @property
    def measurement(self):
        times_per_op = [[] for _ in range(len(self.c_ops))]
        for collapses in self._collapse:
            for time, op in collapses:
                times_per_op[op].append(time)
        return [np.histogram(times, bins=tlist) / self.num_traj
                for times in times_per_op]

    def get_result(self, ntraj=[]):
        # Store results in the Result object
        output = Result()
        output.solver = 'mcsolve'
        output.seeds = self.seeds

        options = self.options
        output.options = options

        if options.steady_state_average:
            output.steady_state = self.steady_state
        if options.average_states and options.store_states:
            output.states = np.array(self.states, dtype=object)
        elif options.store_states:
            output.states = np.array(self.runs_states, dtype=object)

        if options.store_final_state:
            if options.average_states:
                output.final_state = self.final_state
            else:
                output.final_state = self.runs_final_states

        if options.average_expect:
            output.expect = [self.expect_traj_avg(n) for n in self.nums_traj]
            if len(output.expect) == 1:
                output.expect = output.expect[0]
        else:
            output.expect = self.runs_expect

        # simulation parameters
        output.times = self.tlist
        output.num_expect = self._expect.e_num
        output.num_collapse = len(self.c_ops)
        output.ntraj = self.num_traj
        output.col_times = self.collapse_times
        output.col_which = self.collapse_which

        return output
