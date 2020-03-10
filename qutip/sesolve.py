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

__all__ = ['sesolve', 'SeSolver']

import numpy as np
import scipy.integrate
from warnings import warn
from itertools import product

import qutip.settings as qset
from qutip.qobj import Qobj
#from qutip.operators import qeye
from qutip.qobjevo import QobjEvo
from qutip.qobjevo_maker import qobjevo_maker
from scipy.linalg import norm as la_norm
#from qutip.parallel import parallel_map, serial_map
from qutip.solver import (Result, Options, config, solver_safe,
                          Solver, ExpectOps)
from qutip.superoperator import vec2mat

from qutip.ui.progressbar import (BaseProgressBar, TextProgressBar)
from qutip.solverode import OdeScipyZvode, OdeScipyDop853, OdeScipyIVP
from qutip.cy.openmp.utilities import check_use_openmp, openmp_components
from qutip.cy.spconvert import dense1D_to_fastcsr_ket, dense2D_to_fastcsr_fmode
from qutip.cy.spmatfuncs import (cy_expect_psi, cy_ode_psi_func_td,
                                 cy_ode_psi_func_td_with_state,
                                 normalize_inplace, normalize_op_inplace,
                                 normalize_mixed)


def sesolve(H, psi0, tlist, e_ops=None, args=None, options=None,
            progress_bar=None, _safe_mode=True):
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

    e_ops : None / list of :class:`qutip.qobj` / callback function
        single operator or list of operators for which to evaluate
        expectation values.
        For list operator evolution, the overlapse is computed:
            tr(e_ops[i].dag()*op(t))

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : None / :class:`qutip.Qdeoptions`
        with options for the ODE solver.

    progress_bar : None / BaseProgressBar
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
    if options is not None and options.rhs_reuse:
        raise DeprecationWarning
        warn("'rhs_reuse' of Options will be deprecated. "
             "Use the object interface of instead: 'SeSolver'")
        if "sesolve" in solver_safe:
            solver = solver_safe["sesolve"]
            if e_ops: solver.e_ops = e_ops
            if options is not None: solver.options = options
            solver.progress_bar = progress_bar
        else:
            solver = SeSolver(H, args, psi0, tlist, e_ops,
                              options, progress_bar)
            solver_safe["sesolve"] = solver
    else:
        solver = SeSolver(H, args, psi0, tlist, e_ops,
                          options, progress_bar)
        # solver_safe["sesolve"] = solver
    return solver.run(psi0, tlist, args=args, _safe_mode=_safe_mode)


class SeSolver(Solver):
    """Schrodinger equation solver for a given Hamiltonian.

    Evolve the state vector using a given Hamiltonian (`H`), by integrating
    the set of ordinary differential equations that define the system.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments.

    **Time-dependent operators**

    For time-dependent problems, `H` can be a callback function that takes
    two arguments, time and `args`, and returns the Hamiltonian  for the
    system at that point in time (*callback format*).

    Alternatively, `H` can be specified in a nested-list format where each
    element in the list is a list of length 2, containing an operator
    (:class:`qutip.qobj`) at the first element and where the
    second element is either a string (*list string format*), a callback
    function (*list callback format*) that evaluates to the time-dependent
    coefficient for the corresponding operator, or a NumPy array (*list
    array format*) which specifies the value of the coefficient to the
    corresponding operator for each value of t in tlist.

    As an example of a time-dependent problem, consider a Hamiltonian with two
    terms ``H0`` and ``H1``, where ``H1`` is time-dependent with coefficient
    ``sin(w*t)``.  Here, w is a constant arguments with values ``W``.

    Using the Python function time-dependent format requires a Python
    functions. Therefore, this problem could be expressed as::

        def H1_coeff(t,args):
            return sin(args['w']*t)

        H = [H0, [H1, H1_coeff]]

        args={'a': A}

    or in String (Cython) format we could write::

        H = [H0, [H1, 'sin(w*t)']]

        args={'a': A}

    Constant terms are preferably placed first in the Hamiltonian list.

    The Hamiltonian and collapse operators can also be represented a function,
    but this is usually slower than the list format::
        def H(t, args):
            return H0 + H1 * sin(args['w'] * t)

    Parameters
    ----------
    H : :class:`qutip.Qobj`, ``list``
        System Hamiltonian.

    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    psi0 : :class:`qutip.Qobj`
        Initial state

    tlist : array_like
        Times at array based coefficients of time-dependent systems are taken.

    e_ops : :class:`qutip.Qobj`, ``list``
        single operator as Qobj or ``list`` or equivalent of Qobj operators
        for calculating expectation values for feedback. Use the 'e_ops' of the
        'run' method to obtain the expectation values of the evolutions.

    options : Options
        Instance of ODE solver options.

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation. Set to None to disable the
        progress bar.

    outtype: ["Qobj", Qobj, "dense", "sparse", scipy.sparse.spmatrix]
        Type of output states.

    Attributes
    ----------
    options: :class:`qutip.Options`
        options for the evolution, some to pass to scipy's solver, some for
        the output control.

    progress_bar: :class:`qutip.BaseProgressBar`
        How to show the evolution's progress.

    args: dict
        arguments for the time-dependent system.

    outtype: ["Qobj", Qobj, "dense", "sparse", scipy.sparse.spmatrix]
        type of the states being returned.

    Methods
    -------
    run(psi0, tlist, e_ops=None, args=None)
        Compute num_traj trajectories returning a Result object containing.
        the expectation values if e_ops is defined. If not defined or
        options.store_states is set, states are also included in results.

    set(psi0, t0=0)
        Initiate a step by step evolution.

    step(t, args=None, e_ops=[])
        Evolve to t and return the expectation values if e_ops is given and the
        state otherwise. If given, new args are used for the evolution. (Might
        results in a slower step, does not check if the args changed, only that
        they are given.)

    """
    def __init__(self, H, args=None, psi0=None, tlist=[], e_ops=None,
                 options=None, progress_bar=None, outtype=Qobj):
        self.e_ops = e_ops
        self.args = args
        self.progress_bar = progress_bar
        self.options = options
        self.outtype = outtype
        check_use_openmp(self.options)

        self.H = -1j* qobjevo_maker(H, self.args, tlist=tlist,
                                    e_ops=self.e_ops, state=psi0)

        self.with_state = bool(self.H.dynamics_args)
        self.cte = self.H.const
        self.shape = self.H.cte.shape
        self.dims = self.H.cte
        self.psi0 = None
        self.psi = None
        self.solver = None

        if psi0 is not None:
            self._set_psi(psi0)

    def _get_solver(self):
        solver = self.options.solver
        if self.solver and self.solver.name == solver:
            self.solver.update_args(self.args)
            return self.solver

        self.H.compile(omp=self.options.openmp_threads
                       if self.options.use_openmp else 0)
        if solver == "scipy_ivp":
            return OdeScipyIVP(self.H, self.options, self.progress_bar)
        elif solver == "scipy_zvode":
            return OdeScipyZvode(self.H, self.options, self.progress_bar)
        elif solver == "scipy_dop853":
            return OdeScipyDop853(self.H, self.options, self.progress_bar)

    def _set_psi(self, psi0):
        self.psi0 = psi0
        self.state_dims = psi0.dims
        self.state_shape = psi0.shape
        self.psi = psi0.full().ravel("F")
        self.solver = self._get_solver()

    def _check(self, psi0):
        if not (psi0.isket or psi0.isunitary):
            raise TypeError("The unitary solver requires psi0 to be"
                            " a ket as initial state"
                            " or a unitary as initial operator.")
        if psi0.dims[0] != self.H.dims[1]:
            raise ValueError("The dimension of psi0 does not "
                             "fit the Hamiltonian")

    def run(self, psi0, tlist, e_ops=None, args=None, _safe_mode=True):
        if args is not None:
            self.args = args
            self.H.arguments(args)
        opt = self.options
        old_ss = opt.store_states
        e_ops = ExpectOps(e_ops if e_ops is not None else self.e_ops)
        if not self.e_ops:
            opt.store_states = True

        #self.set(psi0, tlist[0])
        self._set_psi(psi0)
        if _safe_mode:
            self._check(self.psi0)
        output = Result()
        output.solver = "sesolve"
        output.times = tlist

        states, expect = self.solver.run(self.psi0, tlist, {}, self.e_ops)

        output.expect = expect
        output.num_expect = len(self.e_ops)
        if opt.store_final_state:
            output.final_state = self.transform(states[-1])
        if opt.store_states:
            output.states = [self.transform(psi)
                             for psi in states]
        opt.store_states = old_ss
        return output

    def step(self, t, args=None, e_ops=[]):
        if args is not None:
            self.solver.update_args(args)
            changed=True
        else:
            changed=False
        state = self.solver.step(t, changed=changed)
        self.t = t
        self.psi = state
        if e_ops:
            return [cy_expect_psi(op.data, state, op.isherm) for op in e_ops]
        return self.transform(state)

    def set(self, psi0=None, t0=0):
        self.t = t0
        psi0 = psi0 if psi0 is not None else self.psi0
        self._set_psi(psi0)
        self.solver.set(self.psi, self.t)
