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
from time import time
from .. import Qobj, QobjEvo
from ..core.qobjevofunc import QobjEvoFunc
from .solver_base import Solver, _to_qevo
from .options import SolverOptions


def sesolve(H, psi0, tlist, e_ops=None,
            args=None, feedback_args=None,
            options=None, _safe_mode=True):
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
        For list operator evolution, the overlap is computed:
            tr(e_ops[i].dag()*op(t))

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    feedback_args : None / *dictionary*
        dictionary of args that dependent on the states.
        With `feedback_args = {key: Qobj}`
        args[key] will be updated to be the state as a Qobj at every use of
        the system.
        `feedback_args = {key: op}` will make args[key] == expect(op, state)

    options : None / :class:`qutip.SolverOptions`
        with options for the ODE solver.

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
    solver = SeSolver(H, e_ops, options, tlist,
                      args, feedback_args, _safe_mode)

    return solver.run(psi0, tlist, args)


class SeSolver(Solver):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Parameters
    ----------
    SeSolver(H, e_ops=None, options=None,
             times=None, args=None, feedback_args=None,
             _safe_mode=False)

    H : :class:`qutip.qobj`, :class:`qutip.qobjevo`, *list*, *callable*
        System Hamiltonian as a Qobj, list of Qobj and coefficient, QobjEvo,
        or a callback function for time-dependent Hamiltonians.
        list format and options can be found in QobjEvo's description.

    e_ops : None / list of :class:`qutip.qobj` or callback function
        single operator or list of operators for which to evaluate
        expectation values.
        For list operator evolution, the overlap is computed:
            tr(e_ops[i].dag()*op(t))

    options : SolverOptions
        Options for the solver

    times : array_like
        List of times at which the numpy-array coefficients are applied.
        Does not need to be the same times as those used for the evolution.

    args : dict
        dictionary that contain the arguments for the coeffients

    feedback_args : None / *dictionary*
        dictionary of args that dependent on the states.
        With `feedback_args = {key: Qobj}`
        args[key] will be updated to be the state as a Qobj at every use of
        the system.
        `feedback_args = {key: op}` will make args[key] == expect(op, state)

    methods
    -------
    run(state, tlist, args)
        Evolve the state vector (`psi0`) using a given
        Hamiltonian (`H`), by integrating the set of ordinary differential
        equations that define the system. Alternatively evolve a unitary
        matrix in solving the Schrodinger operator equation.
        return a Result object

    start(state0, t0):
        Set the initial values for an evolution by steps

    step(t, args={}):
        Evolve to `t`. The system arguments for this step can be updated
        with `args`.
        return the state at t (Qobj), does not compute the expectation values.

    attributes
    ----------
    options : SolverOptions
        Options for the solver
        options can be changed between evolution (before `run` and `start`),
        but not between `step`.

    e_ops : list
        list of Qobj or QobjEvo to compute the expectation values.
        Alternatively, function[s] with the signature f(t, state) -> expect
        can be used.

    """
    def __init__(self, H, e_ops=None, options=None,
                 times=None, args=None, feedback_args=None,
                 _safe_mode=False):
        _time_start = time()
        self.stats = {}
        if e_ops is None:
            e_ops = []
        if options is None:
            options = SolverOptions()
        elif not isinstance(options, SolverOptions):
            raise ValueError("options must be an instance of "
                             "qutip.solver.SolverOptions")
        if args is None:
            args = {}
        if feedback_args is None:
            feedback_args = {}

        self._safe_mode = _safe_mode
        self._super = False
        self._state = None
        self._t = 0

        self.options = options
        self.e_ops = e_ops

        self._system = -1j * _to_qevo(H, args, times)

        self._evolver = self._get_evolver(options, args, feedback_args)
        self.stats["preparation time"] = time() - _time_start
        self.stats['solver'] = "Stochastic Evolution"

    def _safety_check(self, state):
        self._system.mul(0, state)

        if not (state.isket or state.isunitary):
            raise TypeError("The unitary solver requires psi0 to be"
                            " a ket as initial state"
                            " or a unitary as initial operator.")
