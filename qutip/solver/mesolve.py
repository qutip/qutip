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

__all__ = ['mesolve', 'MeSolver']

import numpy as np
from time import time
from .. import (Qobj, QobjEvo, isket, liouvillian, ket2dm, lindblad_dissipator)
from ..core import stack_columns, unstack_columns
from ..core.data import to
from .solver_base import Solver
from .options import SolverOptions
from .sesolve import sesolve


# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
def mesolve(H, rho0, tlist, c_ops=None, e_ops=None, args=None, options=None):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian or Liouvillian (`H`) and an optional set of collapse operators
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
    This allows the solution of master equations that are not in standard
    Lindblad form.

    **Time-dependent operators**

    For time-dependent problems, `H` and `c_ops` can be a QobjEvo

    **Additional options**

    Additional options to mesolve can be set via the `options` argument, which
    should be an instance of :class:`qutip.solver.SolverOptions`. Many ODE
    integration options can be set this way, and the `store_states` and
    `store_final_state` options can be used to store states even though
    expectation values are requested via the `e_ops` argument.

    .. note::

        If an element in the list-specification of the Hamiltonian or
        the list of collapse operators are in superoperator form it will be
        added to the total Liouvillian of the problem without further
        transformation. This allows for using mesolve for solving master
        equations that are not in standard Lindblad form.

    Parameters
    ----------

    H : :class:`Qobj`, :class:`QobjEvo`
        System Hamiltonian as a Qobj or QobjEvo for time-dependent Hamiltonians.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    rho0 : :class:`qutip.Qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of :class:`qutip.Qobj`, :class:`QobjEvo`
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. If none are needed, use an empty list.

    e_ops : list of :class:`qutip.Qobj` / callback function
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : None / :class:`qutip.SolverOptions`
        with options for the solver.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        a *list of array* `result.expect` of expectation values for the times
        specified by `tlist`, and/or a *list* `result.states` of state vectors
        or density matrices corresponding to the times in `tlist` [if `e_ops`
        is an empty list of `store_states=True` in options].

    """
    c_ops = c_ops if c_ops is not None else []
    if not isinstance(c_ops, list):
        c_ops = [c_ops]
    H = QobjEvo(H, args=args, tlist=tlist)

    use_mesolve = len(c_ops) > 0 or (not rho0.isket) or H.issuper

    if not use_mesolve:
        return sesolve(H, rho0, tlist, e_ops=e_ops, args=args, options=options)

    solver = MeSolver(H, c_ops, e_ops, options, tlist, args)

    return solver.run(rho0, tlist, args)


class MeSolver(Solver):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the density matrix (`rho0`) using a given
    Hamiltonian or Liouvillian (`H`) and an optional set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system.

    If either `H` or the Qobj elements in `c_ops` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows the solution of master equations that are not in standard
    Lindblad form.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`
        System Hamiltonian as a Qobj or QobjEvo for time-dependent Hamiltonians.
        list of [:class:`Qobj`, :class:`Coefficient`] or callable that can be
        made into :class:`QobjEvo` are also accepted.

    c_ops : list of :class:`qutip.Qobj`, :class:`QobjEvo`
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. If none are needed, use an empty list.

    e_ops : :class:`qutip.qobj`, callable, or list.
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    options : SolverOptions
        Options for the solver

    times : array_like
        List of times at which the numpy-array coefficients are applied.
        Used when the hamiltonian is passed as a list with array for coeffients.

    args : dict
        dictionary that contain the arguments for the coeffients
        Used when the hamiltonian is passed as a list or callable.

    methods
    -------
    run(state, tlist, args)
        Evolve the density matrix (`rho0`) using a given
        Hamiltonian (`H`) or Liouvillian (`H`), Alternatively evolve a unitary
        matrix.
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
    name = "mesolve"

    def __init__(self, H, c_ops, e_ops=None, options=None,
                 times=None, args=None):
        _time_start = time()
        self.e_ops = e_ops
        self.options = options

        H = QobjEvo(H, args=args, tlist=times)
        c_evos = [QobjEvo(op, args=args, tlist=times) for op in c_ops]

        self._system = H if H.issuper else liouvillian(H)
        self._system += sum(c_op if c_op.issuper else lindblad_dissipator(c_op)
                            for c_op in c_evos )

        self.stats = {}
        self.stats['solver'] = "Master Equation Evolution"
        self.stats['num_collapse'] = len(c_ops)
        self.stats["preparation time"] = time() - _time_start

    def _prepare_state(self, state):
        if isket(state):
            state = ket2dm(state)

        if self.options.ode["State_data_type"]:
            state = state.to(self.options.ode["State_data_type"])
        info = state.dims, state.type, state.isherm

        if self._system.dims[1] == state.dims:
            return stack_columns(state.data), info
        elif self._system.dims[1] == state.dims[0]:
            return state.data, info
        else:
            raise TypeError("".join([
                            "incompatible dimensions ",
                            repr(self._system.dims),
                            " and ",
                            repr(state.dims),])
                           )

    def _restore_state(self, state, info, copy=True):
        dims, type, herm = info
        if state.shape[1] == 1:
            return Qobj(unstack_columns(state),
                        dims=dims, type=type, isherm=herm, copy=False)
        else:
            return Qobj(state, dims=dims, type=type, copy=copy)
