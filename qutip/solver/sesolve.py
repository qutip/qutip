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
from .solver_base import Solver
from .options import SolverOptions


def sesolve(H, psi0, tlist, e_ops=None, args=None, options=None):
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
    H : :class:`Qobj`, :class:`QobjEvo`
        System Hamiltonian as a Qobj or QobjEvo for time-dependent Hamiltonians.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : :class:`qutip.qobj`, callable, or list.
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : None / :class:`qutip.SolverOptions`
        with options for the ODE solver.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        a *list of array* `result.expect` of expectation values for the times
        specified by `tlist`, and/or a *list* `result.states` of state vectors
        or density matrices corresponding to the times in `tlist` [if `e_ops`
        is an empty list of `store_states=True` in options].
    """
    solver = SeSolver(H, e_ops, options, tlist=tlist, args=args)
    return solver.run(psi0, tlist, args)


class SeSolver(Solver):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`
        System Hamiltonian as a Qobj or QobjEvo for time-dependent Hamiltonians.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    e_ops : :class:`qutip.qobj`, callable, or list.
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    options : :class:`SolverOptions`
        Options for the solver

    **kwargs :
        Extra parameters to pass to the QobjEvo creation, such as ``args``.
        See :class:`QobjEvo` for more information.

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

    stats: dict
        Diverse statistics of the evolution.

    """
    name = "sesolve"
    _avail_integrators = {}
    _avail_rhs = {}

    def __init__(self, H, e_ops=None, options=None, **kwargs):
        _time_start = time()
        self.e_ops = e_ops
        self.options = options
        self._system = -1j * QobjEvo(H, **kwargs)
        if not self._system.isoper:
            raise ValueError("The hamiltonian must be an operator")

        self.stats = {}
        self.stats['solver'] = "Schrodinger Evolution"
        self.stats["preparation time"] = time() - _time_start
        self.stats["run time"] = 0

    def _prepare_state(self, state):
        if not (state.isket or state.isunitary):
            raise TypeError("The unitary solver requires psi0 to be"
                            " a ket as initial state"
                            " or a unitary as initial operator.")

        if self._system.dims[1] != state.dims[0]:
            raise TypeError("".join([
                            "incompatible dimensions ",
                            repr(self._system.dims),
                            " and ",
                            repr(state.dims),])
                           )

        if self.options.ode["State_data_type"]:
            state = state.to(self.options.ode["State_data_type"])
        return state.data, {'dims': state.dims, 'type': state.type}

    def _restore_state(self, state, state_metadata, copy=True):
        return Qobj(state, **state_metadata, copy=copy)
