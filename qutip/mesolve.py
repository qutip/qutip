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
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

__all__ = ['mesolve']

import numpy as np
import scipy.integrate
import warnings
from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.superoperator import spre, spost, liouvillian, mat2vec, vec2mat, lindblad_dissipator
from qutip.expect import expect_rho_vec
from qutip.solver import Options, Result, config, solver_safe, SolverSystem
from qutip.cy.spmatfuncs import spmv
from qutip.cy.spconvert import dense2D_to_fastcsr_cmode, dense2D_to_fastcsr_fmode
from qutip.states import ket2dm
from qutip.settings import debug
from qutip.sesolve import sesolve
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.qobjevo import QobjEvo

from qutip.cy.openmp.utilities import check_use_openmp, openmp_components

# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
def mesolve(H, rho0, tlist, c_ops=[], e_ops=[], args={}, options=Options(),
            progress_bar=BaseProgressBar(), _safe_mode=True):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian (`H`) and an [optional] set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system. In the absence of collapse operators the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values.

    If either `H` or the Qobj elements in `c_ops` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows to solve master equations that are not on standard Lindblad
    form by passing a custom Liouvillian in place of either the `H` or `c_ops`
    elements.

    **Time-dependent operators**

    For time-dependent problems, `H` and `c_ops` can be callback
    functions that takes two arguments, time and `args`, and returns the
    Hamiltonian or Liouvillian for the system at that point in time
    (*callback format*).

    Alternatively, `H` and `c_ops` can be a specified in a nested-list format
    where each element in the list is a list of length 2, containing an
    operator (:class:`qutip.qobj`) at the first element and where the
    second element is either a string (*list string format*), a callback
    function (*list callback format*) that evaluates to the time-dependent
    coefficient for the corresponding operator, or a NumPy array (*list
    array format*) which specifies the value of the coefficient to the
    corresponding operator for each value of t in tlist.

    *Examples*

        H = [[H0, 'sin(w*t)'], [H1, 'sin(2*w*t)']]

        H = [[H0, f0_t], [H1, f1_t]]

        where f0_t and f1_t are python functions with signature f_t(t, args).

        H = [[H0, np.sin(w*tlist)], [H1, np.sin(2*w*tlist)]]

    In the *list string format* and *list callback format*, the string
    expression and the callback function must evaluate to a real or complex
    number (coefficient for the corresponding operator).

    In all cases of time-dependent operators, `args` is a dictionary of
    parameters that is used when evaluating operators. It is passed to the
    callback functions as second argument.

    **Additional options**

    Additional options to mesolve can be set via the `options` argument, which
    should be an instance of :class:`qutip.solver.Options`. Many ODE
    integration options can be set this way, and the `store_states` and
    `store_final_state` options can be used to store states even though
    expectation values are requested via the `e_ops` argument.

    .. note::

        If an element in the list-specification of the Hamiltonian or
        the list of collapse operators are in superoperator form it will be
        added to the total Liouvillian of the problem with out further
        transformation. This allows for using mesolve for solving master
        equations that are not on standard Lindblad form.

    .. note::

        On using callback function: mesolve transforms all :class:`qutip.qobj`
        objects to sparse matrices before handing the problem to the integrator
        function. In order for your callback function to work correctly, pass
        all :class:`qutip.qobj` objects that are used in constructing the
        Hamiltonian via args. mesolve will check for :class:`qutip.qobj` in
        `args` and handle the conversion to sparse matrices. All other
        :class:`qutip.qobj` objects that are not passed via `args` will be
        passed on to the integrator in scipy which will raise an NotImplemented
        exception.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian, or a callback function for time-dependent
        Hamiltonians, or alternatively a system Liouvillian.

    rho0 : :class:`qutip.Qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of :class:`qutip.Qobj`
        single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.Options`
        with options for the solver.

    progress_bar : BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        either an *array* `result.expect` of expectation values for the times
        specified by `tlist`, or an *array* `result.states` of state vectors or
        density matrices corresponding to the times in `tlist` [if `e_ops` is
        an empty list], or nothing if a callback function was given in place of
        operators for which to calculate the expectation values.

    """
    # check whether c_ops or e_ops is is a single operator
    # if so convert it to a list containing only that operator
    if isinstance(c_ops, (Qobj, QobjEvo)):
        c_ops = [c_ops]

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if progress_bar is True:
        progress_bar = TextProgressBar()

    # check if rho0 is a superoperator, in which case e_ops argument should
    # be empty, i.e., e_ops = []
    # TODO: e_ops for superoperator
    if issuper(rho0) and not e_ops == []:
        raise TypeError("Must have e_ops = [] when initial condition rho0 is" +
                " a superoperator.")

    if options.rhs_reuse and not isinstance(H, SolverSystem):
        # TODO: deprecate when going to class based solver.
        if "mesolve" in solver_safe:
            # print(" ")
            H = solver_safe["mesolve"]
        else:
            pass
            # raise Exception("Could not find the Hamiltonian to reuse.")

    #check if should use OPENMP
    check_use_openmp(options)

    use_mesolve = ((c_ops and len(c_ops) > 0)
                   or (not isket(rho0))
                   or (isinstance(H, Qobj) and issuper(H))
                   or (isinstance(H, QobjEvo) and issuper(H.cte))
                   or (isinstance(H, list) and isinstance(H[0], Qobj) and
                            issuper(H[0]))
                   or (not isinstance(H, (Qobj, QobjEvo)) and callable(H) and
                            not options.rhs_with_state and issuper(H(0., args)))
                   or (not isinstance(H, (Qobj, QobjEvo)) and callable(H) and
                            options.rhs_with_state))

    if not use_mesolve:
        print("no collapse operator, using sesolve")
        return sesolve(H, rho0, tlist, e_ops=e_ops, args=args, options=options,
                    progress_bar=progress_bar, _safe_mode=_safe_mode)


    if isinstance(H, SolverSystem):
        ss = H
    elif isinstance(H, (list, Qobj, QobjEvo)):
        ss = _mesolve_QobjEvo(H, c_ops, tlist, args, options)
    elif callable(H):
        ss = _mesolve_func_td(H, c_ops, rho0, tlist, args, options)
    else:
        raise Exception("Invalid H type")

    func, ode_args = ss.makefunc(ss, rho0, args, options)
    if isket(rho0):
        rho0 = ket2dm(rho0)

    if _safe_mode:
        v = rho0.full().ravel('F')
        func(0., v, *ode_args) + v

    res = _generic_ode_solve(func, ode_args, rho0, tlist, e_ops, options,
                             progress_bar, dims=rho0.dims)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#_mesolve_QobjEvo(H, c_ops, tlist, args, options)
def _mesolve_QobjEvo(H, c_ops, tlist, args, opt):
    """
    Prepare the system for the solver, H can be an QobjEvo.
    """
    H_td = QobjEvo(H, args, tlist)
    if not issuper(H_td.cte):
        L_td = liouvillian(H_td)
    else:
        L_td = H_td
    for op in c_ops:
        op_td = QobjEvo(op, args, tlist)
        if not issuper(op_td.cte):
            op_td = lindblad_dissipator(op_td)
        L_td += op_td

    if opt.rhs_with_state:
        L_td._check_old_with_state()

    nthread = opt.openmp_threads if opt.use_openmp else 0
    L_td.compile(omp=nthread)

    ss = SolverSystem()
    ss.H = L_td
    ss.makefunc = _qobjevo_set
    solver_safe["mesolve"] = ss
    return ss

def _qobjevo_set(HS, rho0, args, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.arguments(args)
    if issuper(rho0):
        func = H_td.compiled_qobjevo.ode_mul_mat_f_vec
    elif rho0.isket or rho0.isoper:
        func = H_td.compiled_qobjevo.mul_vec
    else:
        raise TypeError("The unitary solver requires rho0 to be"
                        " a ket or dm as initial state"
                        " or a super operator as initial state.")
    return func, ()

# -----------------------------------------------------------------------------
# Master equation solver for python-function time-dependence.
#
class _LiouvillianFromFunc:
    def __init__(self, func, c_ops):
        self.f = func
        self.c_ops = c_ops

    def H2L(self, t, rho, args):
        Ht = self.f(t, args)
        Lt = -1.0j * (spre(Ht) - spost(Ht)).data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt

    def H2L_with_state(self, t, rho, args):
        Ht = self.f(t, rho, args)
        Lt = -1.0j * (spre(Ht) - spost(Ht)).data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt

    def L(self, t, rho, args):
        Lt = self.f(t, args).data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt

    def L_with_state(self, t, rho, args):
        Lt = self.f(t, rho, args).data
        for op in self.c_ops:
            Lt += op(t).data
        return Lt


def _mesolve_func_td(L_func, c_op_list, rho0, tlist, args, opt):
    """
    Evolve the density matrix using an ODE solver with time dependent
    Hamiltonian.
    """
    c_ops = []
    for op in c_op_list:
        op_td = QobjEvo(op, args, tlist, copy=False)
        if not issuper(op_td.cte):
            c_ops += [lindblad_dissipator(op_td)]
        else:
            c_ops += [op_td]
    if c_op_list:
        c_ops_ = [sum(c_ops)]
    else:
        c_ops_ = []

    if opt.rhs_with_state:
        state0 = rho0.full().ravel("F")
        obj = L_func(0., state0, args)
        if not issuper(obj):
            L_func = _LiouvillianFromFunc(L_func, c_ops_).H2L_with_state
        else:
            L_func = _LiouvillianFromFunc(L_func, c_ops_).L_with_state
    else:
        obj = L_func(0., args)
        if not issuper(obj):
            L_func = _LiouvillianFromFunc(L_func, c_ops_).H2L
        else:
            L_func = _LiouvillianFromFunc(L_func, c_ops_).L

    ss = SolverSystem()
    ss.L = L_func
    ss.makefunc = _Lfunc_set
    solver_safe["mesolve"] = ss
    return ss


def _Lfunc_set(HS, rho0, args, opt):
    """
    From the system, get the ode function and args
    """
    L_func = HS.L
    if issuper(rho0):
        func = _ode_super_func_td
    else:
        func = _ode_rho_func_td

    return func, (L_func, args)

def _ode_rho_func_td(t, y, L_func, args):
    L = L_func(t, y, args)
    return spmv(L, y)

def _ode_super_func_td(t, y, L_func, args):
    L = L_func(t, y, args)
    ym = vec2mat(y)
    return (L*ym).ravel('F')

# -----------------------------------------------------------------------------
# Generic ODE solver: shared code among the various ODE solver
# -----------------------------------------------------------------------------

def _generic_ode_solve(func, ode_args, rho0, tlist, e_ops, opt,
                       progress_bar, dims=None):
    """
    Internal function for solving ME.
    Calculate the required expectation values or invoke
    callback function at each time step.
    """
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This function is made similar to sesolve's one for futur merging in a
    # solver class
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # prepare output array
    n_tsteps = len(tlist)
    output = Result()
    output.solver = "mesolve"
    output.times = tlist
    size = rho0.shape[0]

    initial_vector = rho0.full().ravel('F')

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
            # fall back on storing states
            opt.store_states = True
        else:
            for op in e_ops:
                e_ops_data.append(spre(op).data)
                if op.isherm and rho0.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    if opt.store_states:
        output.states = []

    def get_curr_state_data(r):
        return vec2mat(r.y)

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

        if opt.store_states or expt_callback:
            cdata = get_curr_state_data(r)

        if opt.store_states:
            if issuper(rho0):
                fdata = dense2D_to_fastcsr_fmode(cdata, size, size)
                output.states.append(Qobj(fdata, dims=dims))
            else:
                fdata = dense2D_to_fastcsr_fmode(cdata, size, size)
                output.states.append(Qobj(fdata, dims=dims, fast="mc-dm"))

        if expt_callback:
            # use callback method
            output.expect.append(e_ops(t, Qobj(cdata, dims=dims)))

        for m in range(n_expt_op):
            output.expect[m][t_idx] = expect_rho_vec(e_ops_data[m], r.y,
                                                     e_ops[m].isherm)

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if opt.store_final_state:
        cdata = get_curr_state_data(r)
        output.final_state = Qobj(cdata, dims=dims, isherm=True)

    return output
