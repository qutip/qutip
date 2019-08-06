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
"""
This module provides solvers for the unitary Schrodinger equation.
"""

__all__ = ['sesolve']

import os
import types
import numpy as np
import scipy.integrate
import qutip.settings as qset
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from scipy.linalg import norm as la_norm
from qutip.parallel import parallel_map, serial_map
from qutip.cy.spconvert import dense1D_to_fastcsr_ket, dense2D_to_fastcsr_fmode
from qutip.cy.spmatfuncs import (cy_expect_psi, cy_ode_psi_func_td,
                                cy_ode_psi_func_td_with_state, normalize_inplace,
                                normalize_op_inplace, normalize_mixed)
from qutip.solver import (Result, Options, config, solver_safe,
                          SolverSystem, Solver, ExpectOps)
from qutip.superoperator import vec2mat
from qutip.ui.progressbar import (BaseProgressBar, TextProgressBar)
from qutip.cy.openmp.utilities import check_use_openmp, openmp_components
from itertools import product


def stack_ket(kets):
    out = np.zeros((kets[0].shape[0], len(kets)), dtype=complex)
    for i, ket in enumerate(kets):
        out[:,i] = ket.full().ravel()
    return Qobj(out)


class SESolver(Solver):
    """Stochastic Equation Solver

    """
    def __init__(self, H, args={}, tlist=[], options=None):
        if options is None:
            options = Options()

        super().__init__()
        if isinstance(H, (list, Qobj, QobjEvo)):
            ss = _sesolve_QobjEvo(H, tlist, args, options)
        elif callable(H):
            ss = _sesolve_func_td(H, args, options)
        else:
            raise Exception("Invalid H type")

        self.H = H
        self.ss = ss
        self.dims = None
        self.tlist = tlist
        self._args = args
        self.options = options
        self.optimization = {"period":0}

    def set_initial_value(self, psi0, tlist=[]):
        self.state0 = psi0
        self.dims = psi0.dims
        if tlist:
            self.tlist = tlist

    def optimization(self, period=0):
        self.optimization["period"] = period
        raise NotImplementedError

    def run(self, progress_bar=True):
        if progress_bar is True:
            progress_bar = TextProgressBar()

        func, ode_args = self.ss.makefunc(self.ss, self.state0,
                                          self._args, self.options)
        if not self.e_ops:
            self._options.store_states = True

        if not self._options.normalize_output:
            normalize_func = None
        elif self.state0.isket:
            normalize_func = normalize_inplace
        else:
            normalize_func = normalize_op_inplace
        self._e_ops.init(self._tlist)
        self._state_out = self._generic_ode_solve(func, ode_args, self.state0,
                                                  self._tlist, self._e_ops,
                                                  normalize_func, self._options,
                                                  progress_bar)

    def batch_run(self, states=[], args_sets=[],
                  progress_bar=True, map_func=parallel_map):
        N_states0 = len(states)
        if not states:
            states = [self.state0]
        vec_len = self.H.shape[0]
        N_vecs = [state.shape[1] for state in states]
        all_ket = all([n == 1 for n in N_vecs])
        all_op = all([n == vec_len for n in N_vecs])
        if not (all_ket or all_op):
            raise ValueError("Input state must be all ket or operator")

        N_args = len(args_sets)
        if not args_sets:
            args_sets = [self._args]

        if progress_bar is True:
            progress_bar = TextProgressBar()
        map_kwargs = {'progress_bar': progress_bar,
                      'num_cpus': self.options.num_cpus}

        if all_ket and self.ss.with_state:
            res = self._batch_run_ket(states, args_sets, map_func, map_kwargs)
        elif all_ket and (len(N_vecs) > vec_len):
            res = self._batch_run_prop_ket(states, args_sets, map_func, map_kwargs)
        elif all_ket and N_states0 >= 2:
            res = self._batch_run_merged_ket(states, args_sets, map_func, map_kwargs)
        elif all_ket:
            res = self._batch_run_ket(states, args_sets, map_func, map_kwargs)
        elif ss.with_state:
            res = self._batch_run_oper(states, args_sets, map_func, map_kwargs)
        else:
            res = self._batch_run_prop_oper(states, args_sets, map_func, map_kwargs)
        return res

    def _batch_run_ket(self, kets, args_sets, map_func, map_kwargs):
        N_states0 = len(kets)
        N_args = len(args_sets)
        states_out = np.empty((N_states0, N_args), dtype=object)
        expect_out = np.empty((N_states0, N_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        values = list(product(kets, args_sets))

        normalize_func = normalize_inplace
        if not self._options.normalize_output:
            normalize_func = False

        map_func(self._one_run_ket, values, (normalize_func,), **map_kwargs)

        for i, (state, expect) in enumerate(results):
            args_n, state_n = divmod(i, N_states0)
            if self._e_ops:
                expect_out[state_n, args_n] = expect.finish()
            if store_states:
                states_out[state_n, args_n] = state

        return states_out, expect_out

    def _batch_run_prop_ket(self, kets, args_sets, map_func, map_kwargs):
        N_states0 = len(kets)
        N_args = len(args_sets)
        nt = len(self._tlist)
        vec_len = kets[0].shape[0]

        states_out = np.empty((N_states0, N_args), dtype=object)
        expect_out = np.empty((N_states0, N_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        computed_state = [qt.qeye(vec_len)]
        values = list(product(computed_state, args_sets))

        normalize_func = normalize_op_inplace
        if not self._options.normalize_output:
            normalize_func = False

        map_func(self._one_run_ket, values, (normalize_func,), **map_kwargs)

        for i, (prop, _) in enumerate(results):
            args_n, state_n = divmod(i, N_states0)
            for ket in kets:
                e_op = self._e_ops.copy()
                e_op.init(self._tlist)
                state = np.zeros((nt, vec_len), dtype=np.float)
                for i, t in self._tlist:
                    state[i,:] = prop[t,:,:] * ket
                    e_op.step(i, state[i,:])
                if self._e_ops:
                    expect_out[state_n, args_n] = e_op.finish()
                if store_states:
                    states_out[state_n, args_n] = state
        return states_out, expect_out

    def _batch_run_merged_ket(self, kets, args_sets, map_func, map_kwargs):
        N_states0 = len(kets)
        N_args = len(args_sets)
        nt = len(self._tlist)

        states_out = np.empty((N_states0, N_args), dtype=object)
        expect_out = np.empty((N_states0, N_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        values = list(product(stack_ket(kets), args_sets))

        normalize_func = normalize_op_inplace
        if not self._options.normalize_output:
            normalize_func = False

        map_func(self._one_run_ket, values, (normalize_func,), **map_kwargs)

        for i, (state, _) in enumerate(results):
            args_n, state_n = divmod(i, N_states0)
            for i in range(state.shape[2]):
                vecs = state[:,:,i]
                e_op = self._e_ops.copy()
                e_op.init(self._tlist)
                for i, t in self._tlist:
                    e_op.step(i, vecs[t,:])
                if self._e_ops:
                    expect_out[state_n, args_n] = e_op.finish()
                if store_states:
                    states_out[state_n, args_n] = vecs
        return states_out, expect_out

    def _batch_run_oper(self, opers, args_sets, map_func, map_kwargs):
        N_states0 = len(opers)
        N_args = len(args_sets)
        states_out = np.empty((N_states0, N_args), dtype=object)
        expect_out = np.empty((N_states0, N_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        values = list(product(opers, args_sets))

        normalize_func = normalize_inplace
        if not self._options.normalize_output:
            normalize_func = False

        map_func(self._one_run_oper, values, (normalize_func,), **map_kwargs)

        for i, (state, expect) in enumerate(results):
            args_n, state_n = divmod(i, N_states0)
            if self._e_ops:
                expect_out[state_n, args_n] = expect.finish()
            if store_states:
                states_out[state_n, args_n] = state

        return states_out, expect_out

    def _batch_run_prop_ket(self, opers, args_sets, map_func, map_kwargs):
        N_states0 = len(opers)
        N_args = len(args_sets)
        nt = len(self._tlist)
        vec_len = opers[0].shape[0]

        states_out = np.empty((N_states0, N_args), dtype=object)
        expect_out = np.empty((N_states0, N_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        computed_state = [qt.qeye(vec_len)]
        values = list(product(computed_state, args_sets))

        normalize_func = normalize_op_inplace
        if not self._options.normalize_output:
            normalize_func = False

        map_func(self._one_run_ket, values, (normalize_func,), **map_kwargs)

        for i, (prop, _) in enumerate(results):
            args_n, state_n = divmod(i, N_states0)
            for oper in opers:
                e_op = self._e_ops.copy()
                e_op.init(self._tlist)
                state = np.zeros((nt, vec_len, vec_len), dtype=np.float)
                for i, t in self._tlist:
                    state[i,:,:] = np.conj(prop[t,:,:].T) @ oper @ prop[t,:,:]
                    e_op.step(i, state[i,:,:])
                if self._e_ops:
                    expect_out[state_n, args_n] = e_op.finish()
                if store_states:
                    states_out[state_n, args_n] = state
        return states_out, expect_out

    def _one_run_ket(self, run_data, normalize_func):
        opt = self._options
        state0, args = run_data
        func, ode_args = self.ss.makefunc(self.ss, state0, args, opt)

        if state0.isket:
            e_ops = self._e_ops.copy()
            e_ops.init(self._tlist)
        else:
            e_ops = ExpectOps([])
        state = self._generic_ode_solve(func, ode_args, state0, self._tlist,
                                        e_ops, normalize_func, opt,
                                        BaseProgressBar())
        return state, e_ops

    def _one_run_oper(self, run_data, normalize_func):
        opt = self._options
        state0, args = run_data
        func, ode_args = self.ss.makeoper(self.ss, state0, args, opt)

        e_ops = self._e_ops.copy()
        e_ops.init(self._tlist)
        state = self._generic_ode_solve(func, ode_args, state0, self._tlist,
                                        e_ops, normalize_func, opt,
                                        BaseProgressBar())
        return state, e_ops


def sesolve(H, psi0, tlist, e_ops=[], args={}, options=Options(),
            progress_bar=BaseProgressBar(), _safe_mode=True):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Evolve the state vector (`psi0`) using a given
    Hamiltonian (`H`), by integrating the set of ordinary differential
    equations that define the system. Alternatively evolve a unitary matrix in
    solving the Schrodinger operator equation.

    The output is either the state vector or unitary matrix at arbitrary points
    in time (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    Parameters
    ----------

    H : :class:`qutip.qobj`, :class:`qutip.qobjevo`, *list*, *callable*
        system Hamiltonian as a Qobj, list of Qobj and coefficient, QobjEvo,
        or a callback function for time-dependent Hamiltonians.
        list format and options can be found in QobjEvo's description.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        single operator or list of operators for which to evaluate
        expectation values.
        For list operator evolution, the overlapse is computed:
            tr(e_ops[i].dag()*op(t))

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : :class:`qutip.Qdeoptions`
        with options for the ODE solver.

    progress_bar : BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`, or
        an *array* or state vectors corresponding to the
        times in `tlist` [if `e_ops` is an empty list], or
        nothing if a callback function was given inplace of operators for
        which to calculate the expectation values.

    """
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]
    elif isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if progress_bar is True:
        progress_bar = TextProgressBar()

    if not (psi0.isket or psi0.isunitary):
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")

    if options.rhs_reuse and not isinstance(H, SolverSystem):
        # TODO: deprecate when going to class based solver.
        if "sesolve" in solver_safe:
            # print(" ")
            H = solver_safe["sesolve"]
        else:
            pass
            # raise Exception("Could not find the Hamiltonian to reuse.")

    #check if should use OPENMP
    check_use_openmp(options)

    if isinstance(H, SolverSystem):
        ss = H
    elif isinstance(H, (list, Qobj, QobjEvo)):
        ss = _sesolve_QobjEvo(H, tlist, args, options)
    elif callable(H):
        ss = _sesolve_func_td(H, args, options)
    else:
        raise Exception("Invalid H type")

    func, ode_args = ss.makefunc(ss, psi0, args, options)

    if _safe_mode:
        v = psi0.full().ravel('F')
        func(0., v, *ode_args) + v

    res = _se_ode_solve(func, ode_args, psi0, tlist, e_ops,
                        options, progress_bar, dims=psi0.dims)
    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    res.SolverSystem = ss
    return res


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#
def _sesolve_QobjEvo(H, tlist, args, opt):
    """
    Prepare the system for the solver, H can be an QobjEvo.
    """
    H_td = -1.0j * QobjEvo(H, args, tlist)
    if opt.rhs_with_state:
        H_td._check_old_with_state()
    nthread = opt.openmp_threads if opt.use_openmp else 0
    H_td.compile(omp=nthread)

    ss = SolverSystem()
    ss.type = "QobjEvo"
    ss.H = H_td
    ss.shape = H_td.cte.shape
    ss.dims = H_td.cte.dims
    ss.with_state = bool(H_td.dynamics_args)
    ss.makefunc = _qobjevo_set
    ss.makeoper = _qobjevo_set_oper
    solver_safe["sesolve"] = ss
    return ss

def _qobjevo_set(HS, psi, args, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.arguments(args)

    if psi.isket:
        func = H_td.compiled_qobjevo.mul_vec
    elif psi.isunitary:
        func = H_td.compiled_qobjevo.ode_mul_mat_f_vec
    else:
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")
    return func, ()

def _qobjevo_set_oper(HS, psi, args, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.arguments(args)
    N = psi.shape[0]

    def _oper_evolution(t, mat):
        oper = H_td.compiled_qobjevo.ode_mul_mat_f_vec(t, mat)
        out = mat.reshape((N,N)).T @ H_td.call(t, 1) - oper.reshape((N,N)).T
        return out.ravel("F")

    return _oper_evolution, ()

# -----------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians.
#
def _sesolve_func_td(H_func, args, opt):
    """
    Prepare the system for the solver, H is a function.
    """
    ss = SolverSystem()
    ss.type = "func"
    ss.H = H_func
    ss.makefunc = _Hfunc_set
    ss.makeoper = _Hfunc_set_oper
    solver_safe["sesolve"] = ss
    if not opt.rhs_with_state:
        ss.shape = h_func(0., args).shape
        ss.dims = h_func(0., args).dims
        ss.with_state = False
    else:
        ss.shape = None
        ss.dims = [[1],[1]]
        ss.with_state = True
    return ss

def _Hfunc_set(HS, psi, args, opt):
    """
    From the system, get the ode function and args
    """
    H_func = HS.H
    if psi.isunitary:
        if not opt.rhs_with_state:
            print("_ode_oper_func_td")
            func = _ode_oper_func_td
        else:
            print("_ode_oper_func_td_with_state")
            func = _ode_oper_func_td_with_state
    else:
        if not opt.rhs_with_state:
            print("cy_ode_psi_func_td")
            func = cy_ode_psi_func_td
        else:
            print("cy_ode_psi_func_td_with_state")
            func = cy_ode_psi_func_td_with_state

    return func, (H_func, args)

def _Hfunc_set_oper(HS, psi, args, opt):
    """
    From the system, get the ode function and args
    """
    H_func = HS.H

    def _HO_OH(t, mat):
        H = H_func(t, args).full()
        op = mat.reshape((n, n)).T
        return op @ H - H @ op

    def _HO_OH_state(t, mat):
        H = H_func(t, mat, args).full()
        op = mat.reshape((n, n)).T
        return op @ H - H @ op

    return _HO_OH_state if opt.rhs_with_state else _HO_OH


# -----------------------------------------------------------------------------
# evaluate dU(t)/dt according to the schrodinger equation
#
def _ode_oper_func_td(t, y, H_func, args):
    H = H_func(t, args).data * -1j
    ym = vec2mat(y)
    return (H * ym).ravel("F")

def _ode_oper_func_td_with_state(t, y, H_func, args):
    H = H_func(t, y, args).data * -1j
    ym = vec2mat(y)
    return (H * ym).ravel("F")


# -----------------------------------------------------------------------------
# Solve an ODE for func.
# Calculate the required expectation values or invoke callback
# function at each time step.
def _se_ode_solve(func, ode_args, psi0, tlist, e_ops, opt,
                       progress_bar, dims=None):
    """
    Internal function for solving ODEs.
    """
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This function is made similar to mesolve's one for futur merging in a
    # solver class
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # prepare output array
    n_tsteps = len(tlist)
    output = Result()
    output.solver = "sesolve"
    output.times = tlist

    if psi0.isunitary:
        initial_vector = psi0.full().ravel('F')
        oper_evo = True
        size = psi0.shape[0]
        # oper_n = dims[0][0]
        # norm_dim_factor = np.sqrt(oper_n)
    elif psi0.isket:
        initial_vector = psi0.full().ravel()
        oper_evo = False
        # norm_dim_factor = 1.0

    r = scipy.integrate.ode(func)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    if ode_args:
        r.set_f_params(*ode_args)
    r.set_initial_value(initial_vector, tlist[0])

    e_ops_data = []
    output.expect = []
    if callable(e_ops):
        n_expt_op = 0
        expt_callback = True
        output.num_expect = 1
    elif isinstance(e_ops, list):
        n_expt_op = len(e_ops)
        expt_callback = False
        output.num_expect = n_expt_op
        if n_expt_op == 0:
            # fallback on storing states
            opt.store_states = True
        else:
            for op in e_ops:
                if op.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
        if oper_evo:
            for e in e_ops:
                e_ops_data.append(e.dag().data)
        else:
            for e in e_ops:
                e_ops_data.append(e.data)
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    if opt.store_states:
        output.states = []

    if oper_evo:
        def get_curr_state_data(r):
            return vec2mat(r.y)
    else:
        def get_curr_state_data(r):
            return r.y

    #
    # start evolution
    #
    dt = np.diff(tlist)
    cdata = None
    progress_bar.start(n_tsteps)
    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)
        if not r.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")
        # get the current state / oper data if needed
        if opt.store_states or opt.normalize_output or n_expt_op > 0 or expt_callback:
            cdata = get_curr_state_data(r)

        if opt.normalize_output:
            # normalize per column
            if oper_evo:
                cdata /= la_norm(cdata, axis=0)
                #cdata *= norm_dim_factor / la_norm(cdata)
                r.set_initial_value(cdata.ravel('F'), r.t)
            else:
                #cdata /= la_norm(cdata)
                norm = normalize_inplace(cdata)
                if norm > 1e-12:
                    # only reset the solver if state changed
                    r.set_initial_value(cdata, r.t)
                else:
                    r._y = cdata

        if opt.store_states:
            if oper_evo:
                fdata = dense2D_to_fastcsr_fmode(cdata, size, size)
                output.states.append(Qobj(fdata, dims=dims))
            else:
                fdata = dense1D_to_fastcsr_ket(cdata)
                output.states.append(Qobj(fdata, dims=dims, fast='mc'))

        if expt_callback:
            # use callback method
            output.expect.append(e_ops(t, Qobj(cdata, dims=dims)))

        if oper_evo:
            for m in range(n_expt_op):
                output.expect[m][t_idx] = (e_ops_data[m] * cdata).trace()
        else:
            for m in range(n_expt_op):
                output.expect[m][t_idx] = cy_expect_psi(e_ops_data[m], cdata,
                                                        e_ops[m].isherm)

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if opt.store_final_state:
        cdata = get_curr_state_data(r)
        if opt.normalize_output:
            cdata /= la_norm(cdata, axis=0)
            # cdata *= norm_dim_factor / la_norm(cdata)
        output.final_state = Qobj(cdata, dims=dims)

    return output
