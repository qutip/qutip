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
from warnings import warn

from qutip.superoperator import (spre, spost, liouvillian, mat2vec,
                                 vec2mat, lindblad_dissipator)
from qutip.solver import (Result, Options, config, solver_safe,
                          Solver, ExpectOps)
from qutip.cy.spmatfuncs import spmv
from qutip.cy.spconvert import (dense2D_to_fastcsr_cmode,
                                dense2D_to_fastcsr_fmode)

from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.parallel import parallel_map, serial_map
from qutip.states import ket2dm
from qutip.sesolve import sesolve
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.solverode import OdeScipyZvode, OdeScipyDop853, OdeScipyIVP

from qutip.qobjevo import QobjEvo
from qutip.qobjevo_maker import qobjevo_maker
from qutip.cy.openmp.utilities import check_use_openmp


class MeSolver(Solver):
    def __init__(self, H, c_ops, args=None,
                 rho0=None, tlist=[], e_ops=None,
                 options=None, progress_bar=None):
        self.e_ops = e_ops
        self.args = args
        self.progress_bar = progress_bar
        self.options = options
        #if options is None:
        self.options.normalize_output = False
        check_use_openmp(self.options)

        self.H = qobjevo_maker(H, self.args, tlist=tlist,
                               e_ops=e_ops, state=rho0)
        self.c_ops = [qobjevo_maker(op, self.args, tlist=tlist,
                                    e_ops=e_ops, state=rho0) for op in c_ops]

        self.with_state = bool(self.H.dynamics_args)
        self.cte = self.H.const
        self.shape = self.H.cte.shape
        self.dims = self.H.cte.dims
        self.rho0 = None
        self.rho = None
        self.solver = None

        if rho0 is not None:
            self._set_rho(rho0)

    def _get_solver(self):
        solver = self.options.solver
        if self.solver and self.solver.name == solver:
            self.solver.update_args(self.args)
            return self.solver
        L = liouvillian(self.H, self.c_ops)
        L.compile(omp=self.options.openmp_threads
                  if self.options.use_openmp else 0)
        if solver == "scipy_ivp":
            return OdeScipyIVP(L, self.options, self.progress_bar)
        elif solver == "scipy_zvode":
            return OdeScipyZvode(L, self.options, self.progress_bar)
        elif solver == "scipy_dop853":
            return OdeScipyDop853(L, self.options, self.progress_bar)
        else:
            raise ValueError("Invalid options.solver", solver)

    def _set_rho(self, rho0):
        if rho0.isket:
            rho0 = ket2dm(rho0)
        self.rho0 = rho0
        self.state_dims = rho0.dims
        self.state_shape = rho0.shape
        self.rho = rho0.full().ravel("F")
        self.solver = self._get_solver()

    def _check(self, rho0):
        dims_H = self.dims if not issuper(self.H.cte) else self.dims[1]
        ket_ok = rho0.isket and (dims_H[1] == rho0.dims[0])
        dm_ok = rho0.isoper and (dims_H == rho0.dims)
        sp_ok = rho0.issuper and (dims_H == rho0.dims[0])
        if not (ket_ok or dm_ok or sp_ok):
            raise ValueError("The dimension of rho0 does not "
                             "fit the Hamiltonian")

    def run(self, tlist, rho0=None, args=None, e_ops=None,
            outtype=Qobj, _safe_mode=False):
        if args is not None:
            self.args = args
        self.set(rho0, tlist[0])
        opt = self.options
        if _safe_mode:
            self._check(self.rho0)
        old_ss = opt.store_states
        e_ops = ExpectOps(e_ops) if e_ops is not None else self.e_ops
        if not e_ops:
            opt.store_states = True

        output = Result()
        output.solver = "mesolve"
        output.times = tlist
        states, expect = self.solver.run(self.rho0, tlist, {}, e_ops)
        output.expect = expect
        output.num_expect = len(e_ops)
        if opt.store_final_state:
            output.final_state = self.transform(states[-1], outtype)
        if opt.store_states:
            output.states = [self.transform(rho,  outtype)
                             for rho in states]
        opt.store_states = old_ss
        return output

    def step(self, t, args=None, outtype=Qobj, e_ops=[]):
        if args is not None:
            self.solver.update_args(args)
            changed=True
        else:
            changed=False
        state = self.solver.step(self.rho, t, changed=changed)
        self.t = t
        self.rho = state
        if e_ops:
            return [expect(op, state) for op in e_ops]
        return self.transform(states, outtype)

    def set(self, rho0=None, t0=0):
        self.t = t0
        rho0 = rho0 if rho0 is not None else self.rho0
        self._set_rho(rho0)


def mesolve(H, rho0, tlist, c_ops=None, e_ops=None, args=None, options=None,
            progress_bar=None, _safe_mode=True):
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

    c_ops : None / list of :class:`qutip.Qobj`
        single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators.

    e_ops : None / list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : None / :class:`qutip.Options`
        with options for the solver.

    progress_bar : None / BaseProgressBar
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
    args = args if args is not None else {}
    e_ops = e_ops if e_ops is not None else []
    H_evo = qobjevo_maker(H, args=args, e_ops=e_ops, tlist=tlist)
    H_issuper = H_evo.issuper
    H_feedback = H_evo.feedback
    if not c_ops and isket(rho0) and not H_issuper and not H_feedback:
        warn("mesolve will no longer return ket for closed systems from v5")
        return sesolve(H, rho0, tlist, e_ops, args,
                       options, progress_bar, _safe_mode)
    if options is not None and options.rhs_reuse:
        raise DeprecationWarning
        warn("'rhs_reuse' of Options will be deprecated. "
             "Use the object interface of instead: 'MeSolver'")
        if "mesolve" in solver_safe:
            solver = solver_safe["mesolve"]
            if e_ops: solver.e_ops = e_ops
            if options is not None: solver.options = options
            solver.progress_bar = progress_bar
        else:
            c_ops = c_ops if c_ops is not None else []
            solver = MeSolver(H, c_ops, args, rho0, tlist, e_ops,
                              options, progress_bar)
            solver_safe["mesolve"] = solver
    else:
        c_ops = c_ops if c_ops is not None else []
        solver = MeSolver(H, c_ops, args, rho0, tlist, e_ops,
                          options, progress_bar)
        # solver_safe["mesolve"] = solver
    return solver.run(tlist, rho0, args, _safe_mode=_safe_mode)


class Splited_Liouvillian:
    """apply liouvillian without using super operator.
    Could be faster in some case, to test.

    Expectation:
        better for denser operators if compiled with dense=True
        better for small matrix
        better for constant system
        Need compiled left product in QobjEvo and return dense state.
    """
    def __init__(self, H, c_ops, num_cpus=0, dense=False):
        self.H = H
        self.c_ops = c_ops
        self.c_dag = [c.dag() for c in c_ops]
        self.cdc = [c._cdc() for c in c_ops]
        [c.compile(dense=dense, omp=num_cpus) for c in self.c_ops]
        [c.compile(dense=dense, omp=num_cpus) for c in self.c_dag]
        [c.compile(dense=dense, omp=num_cpus) for c in self.cdc]

    def arguments(self, args):
        self.H.arguments(args)
        [c.arguments(args) for c in self.c_ops]
        [c.arguments(args) for c in self.c_dag]
        [c.arguments(args) for c in self.cdc]

    def _get_mul(self, state):
        self.shape = state.shape
        return self.__call__

    def __call__(self, state):
        dm = state.resize(self.shape).T
        H = self.H(t, state=state, data=True)
        ddm = -1j * (H @ dm - dm @ H)
        for i, cdc_op in enumerate(self.cdc):
            cdc = cdc_op(t, state=state, data=True)
            c = self._c_ops[i](t, state=state, data=True)
            cdag = self.c_dag[i](t, state=state, data=True)
            ddm += -0.5 * (cdc @ dm + dm @ cdc)
            ddm += c @ dm @ cdag
        return ddm
