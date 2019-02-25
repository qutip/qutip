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
from functools import partial
import numpy as np
import scipy.integrate
from scipy.linalg import norm as la_norm
import qutip.settings as qset
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.solver import (Result, Options, config,
                          _solver_safety_check, solver_safe, SolverSystem)
from qutip.superoperator import vec2mat
from qutip.settings import debug
from qutip.ui.progressbar import (BaseProgressBar, TextProgressBar)
from qutip.cy.openmp.utilities import check_use_openmp, openmp_components

if qset.has_openmp:
    from qutip.cy.openmp.parfuncs import cy_ode_rhs_openmp

if debug:
    import inspect



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

    #check if should use OPENMP
    check_use_openmp(options)

    if isinstance(H, (list, Qobj, QobjEvo, SolverSystem)) or options.rhs_reuse:
        res = _sesolve_QobjEvo(H, psi0, tlist, e_ops, args, options,
                               progress_bar, _safe_mode)
    elif iscallable(H):
        res = _sesolve_func_td(H, psi0, tlist, e_ops, args, options,
                               progress_bar, _safe_mode)
    else:
        raise Exception("Invalid H type")

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res

# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#
def _sesolve_QobjEvo(H, psi0, tlist, e_ops, args, opt,
                     progress_bar, _safe_mode=False):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """
    if debug:
        print(inspect.stack()[0][3])

    if isinstance(H, SolverSystem):
        H_td = H.H
        H_td.arguments(args)
    elif opt.rhs_reuse:
        # TODO: to deprecate?
        H_td = solver_safe["sesolve"]
        H_td.arguments(args)
    else:
        H_td = -1.0j * td_Qobj(H, args, tlist)
        if opt.rhs_with_state:
            H._check_old_with_state()
        H_td.compile()
        ss = SolverSystem()
        ss.type = "QobjEvo"
        ss.H = H_td
        solver_safe["sesolve"] = ss

    # check initial state or oper and setup integrator
    if psi0.isunitary:
        func = H_td.compiled_Qobj.mul_mat
        if _safe_mode:
            H_td.mul_mat(0, psi0)
    elif psi0.isket:
        func = H_td.compiled_Qobj.mul_vec
        if _safe_mode:
            H_td.mul_vec(0, psi0)
    else:
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")

    # call generic ODE code
    return _generic_ode_solve(r, psi0, tlist, e_ops, opt, progress_bar,
                              dims=psi0.dims)


# -----------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians.
#
def _sesolve_func_td(H_func, psi0, tlist, e_ops, args, opt, progress_bar,
                     _safe_mode=False):
    """!
    Evolve the wave function using an ODE solver with time-dependent
    Hamiltonian.
    """
    if _safe_mode:
        _solver_safety_check(H, psi0, c_ops=[], e_ops=e_ops, args=args)

    if debug:
        print(inspect.stack()[0][3])

    if psi0.isunitary:
        if not opt.rhs_with_state:
            func = _ode_oper_func_td
        else:
            func = _ode_oper_func_td_with_state

    else:
        if not opt.rhs_with_state:
            func = cy_ode_psi_func_td
        else:
            func = cy_ode_psi_func_td_with_state

    #
    # call generic ODE code
    #
    return _generic_ode_solve(func, psi0, tlist, e_ops, opt, progress_bar,
                              dims=psi0.dims, ode_args=(H_func, args))

#
# evaluate dU(t)/dt according to the master equation using the
#
# TODO cythonize these?
def _ode_oper_func_td(t, y, H_func, args):
    H = H_func(t, args)
    return -1j * _ode_oper_func(t, y, H.data)

def _ode_oper_func_td_with_state(t, y, H_func, args):
    H = H_func(t, y, args)
    return -1j * _ode_oper_func(t, y, H.data)

# -----------------------------------------------------------------------------
# Solve an ODE which solver parameters already setup (r). Calculate the
# required expectation values or invoke callback function at each time step.
#
def _generic_ode_solve(func, psi0, tlist, e_ops, opt, progress_bar, dims=None, ode_args=()):
    """
    Internal function for solving ODEs.
    """
    # prepare output array
    n_tsteps = len(tlist)
    output = Result()
    output.solver = "sesolve"
    output.times = tlist

    r = scipy.integrate.ode(func)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_f_params(ode_args)

    if psi0.isunitary:
        initial_vector = psi0.full().ravel('F')
        oper_evo = True
        oper_n = dims[0][0]
        norm_dim_factor = np.sqrt(oper_n)
    elif psi0.isket:
        initial_vector = psi0.full().ravel()
        oper_evo = False
        norm_dim_factor = 1.0
    else:
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")
    r.set_initial_value(initial_vector, tlist[0])

    output.expect = []
    if isinstance(e_ops, types.FunctionType):
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
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    if opt.store_states:
        output.states = []

    if oper_evo:
        def get_curr_state_data():
            return vec2mat(r.y)
    else:
        def get_curr_state_data():
            return r.y

    #
    # start evolution
    #
    progress_bar.start(n_tsteps)

    dt = np.diff(tlist)
    cdata = None
    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)

        if not r.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")
        # get the current state / oper data if needed
        if opt.store_states or opt.normalize_output or n_expt_op > 0 or expt_callback:
            cdata = get_curr_state_data()

        if opt.normalize_output:
            # normalize per column
            cdata /= la_norm(cdata, axis=0)
            #cdata *= norm_dim_factor / la_norm(cdata)
            if oper_evo:
                r.set_initial_value(cdata.ravel('F'), r.t)
            else:
                r.set_initial_value(cdata, r.t)

        if opt.store_states:
            output.states.append(Qobj(cdata, dims=dims))

        if expt_callback:
            # use callback method
            output.expect.append(e_ops(t, Qobj(cdata, dims=dims)))

        if oper_evo:
            for m in range(n_expt_op):
                output.expect[m][t_idx] = (e_ops[m].dag().data * cdata).trace()
        else:
            for m in range(n_expt_op):
                output.expect[m][t_idx] = cy_expect_psi(e_ops[m].data,
                                                        cdata, e_ops[m].isherm)

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% to remove
    if not opt.rhs_reuse and config.tdname is not None:
        try:
            os.remove(config.tdname + ".pyx")
        except:
            pass
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% to remove

    if opt.store_final_state:
        cdata = get_curr_state_data()
        if opt.normalize_output:
            cdata *= norm_dim_factor / la_norm(cdata)
        output.final_state = Qobj(cdata, dims=dims)

    return output
