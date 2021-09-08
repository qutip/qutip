# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson,
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
from __future__ import print_function

__all__ = ['Solver']

# import numpy as np
# from ..core import data as _data

from .. import Qobj, QobjEvo
from .options import SolverOptions
from .result import Result
from .integrator import integrator_collection
from ..ui.progressbar import get_progess_bar
from ..core.data import to
from time import time


class Solver:
    """
    Main class of the solvers.
    Do the loop over each times in tlist and does the interface between the
    evolver which deal in data and the Result which use Qobj.
    It's children (SeSolver, McSolver) are responsible with building the system
    (-1j*H).

    methods
    -------
    run(state0, tlist, args={}):
        Do an evolution starting with `state0` at `tlist[0]` and obtain a
        result for each time in `tlist`.
        The system's arguments can be changed with `args`.

    start(state0, t0):
        Set the initial values for an evolution by steps

    step(t, args={}):
        Do a step to `t`. The system arguments for this step can be updated
        with `args`

    attributes
    ----------
    options : SolverOptions
        Options for the solver

    e_ops : list
        list of Qobj or QobjEvo to compute the expectation values.
        Alternatively, function[s] with the signature f(t, state) -> expect
        can be used.

    stats: dict
        Diverse statistics of the evolution.

    """
    # sesolve, mesolve, etc. used when choosing the
    name = ""

    # State, time and Integrator of the stepper functionnality
    _t = 0
    _state = None
    _integrator = False

    # Class of option used by the solver
    optionsclass = SolverOptions

    def __init__(self, e_ops, args, feedback_args):
        raise NotImplementedError

    def _prepare_state(self, state):
        # Do the dims checks
        # prepare the data from the Qobj (reshape, update type)
        # return state.data, info
        raise NotImplementedError

    def _restore_state(self, state, info, copy=True):
        # rebuild the Qobj from the state's data
        # info pass dims, type, etc., from _prepare_state to _restore_state
        raise NotImplementedError

    def run(self, state0, tlist, args={}):
        _data0, info = self._prepare_state(state0)
        _integrator = self._get_integrator()
        if args:
            _integrator.update_args(args)
        _time_start = time()
        _integrator.set_state(tlist[0], _data0)
        self.stats["preparation time"] += time() - _time_start
        res = Result(self.e_ops, self.options.results, state0)
        res.add(tlist[0], state0)

        progress_bar = get_progess_bar(self.options['progress_bar'])
        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in _integrator.run(tlist):
            progress_bar.update()
            res.add(t, self._restore_state(state, info, False))
        progress_bar.finished()

        self.stats['run time'] = progress_bar.total_time()
        self.stats.update(_integrator.stats)
        self.stats["method"] = _integrator.name
        res.stats = self.stats.copy()
        return res

    def start(self, state0, t0):
        _time_start = time()
        self._state, self.info = self._prepare_state(state0)
        self._t = t0
        self._integrator = self._get_integrator()
        self._integrator.set_state(self._t, self._state)
        self.stats["preparation time"] += time() - _time_start

    def step(self, t, args={}):
        if not self._integrator:
            raise RuntimeError("The `start` method must called first")
        if args:
            self._integrator.update_args(args)
            self._integrator.set_state(self._t, self._state)
        self._t, self._state = self._integrator.step(t, copy=False)
        return self._restore_state(self._state, self.info)

    def _get_integrator(self):
        method = self.options.ode["method"]
        rhs = self.options.ode["rhs"]
        td_system = not self._system.isconstant or None
        op_type = self.options.ode["Operator_data_type"]
        # TODO: with #1420, it should be changed to `in to._str2type`
        if op_type in to.dtypes:
            self._system = self._system.to(op_type)
        integrator = integrator_collection[method, rhs]
        # Check if the solver is supported by the integrator
        if not integrator_collection.check_condition(
            method, "", solver=self.name, time_dependent=td_system
        ):
            raise ValueError(f"ODE integrator method {method} not supported "
                f"by {self.name}" +
                ("for time dependent system" if td_system else "")
            )
        if not integrator_collection.check_condition(
            "", rhs, solver=self.name, time_dependent=td_system
        ):
            raise ValueError(f"ODE integrator rhs {rhs} not supported by " +
                f"{self.name}" +
                ("for time dependent system" if td_system else "")
            )
        return integrator(self._system, self.options)

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new):
        if new is None:
            new = self.optionsclass()
        if not isinstance(new, self.optionsclass):
            raise TypeError("options must be an instance of",
                            self.optionsclass)
        self._options = new
