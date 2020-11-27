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
This module provides solvers for
"""
__all__ = ['Evolver', 'EvolverScipyDop853',
           'EvolverVern', 'EvolverDiag', 'get_evolver']


import numpy as np
from numpy.linalg import norm as la_norm
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from ..core import data as _data
from ._solverqevo import SolverQEvo
import warnings

all_ode_method = ['adams', 'bdf', 'dop853', 'vern7', 'vern9']

class qutip_zvode(zvode):
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r

def get_evolver(system, options, args, feedback_args):
    if options.ode['method'] in ['adams','bdf']:
        return EvolverScipyZvode(system, options, args, feedback_args)
    elif options.ode['method'] in ['dop853']:
        return EvolverScipyDop853(system, options, args, feedback_args)
    elif options.ode['method'] in ['vern7', 'vern9']:
        return EvolverVern(system, options, args, feedback_args)
    elif options.ode['method'] in ['diagonalized', 'diag']:
        return EvolverDiag(system, options, args, feedback_args)
    raise ValueError("method options not recognized")

class _EvolverCollection:
    def __init__(self):
        self.method2evolver = {}
        self.rhs2evolver = {}
        self.evolver_data = {}

    def add(self, evolver, methods=[], rhs=[], limits={}, _test=True):
        if not isinstance(methods, list):
            methods = [methods]
        if not isinstance(rhs, list):
            rhs = [rhs]
        if not (methods or rhs):
            raise ValueError("Most have a method or rhs associated")
        if methods and rhs:
            raise ValueError("Cannot be both a method and rhs")
        for method in methods:
            if (
                method in self.method2evolver and
                evolver.__name__ not in self.evolver_data
            ):
                raise ValueError("method '{}' already used")
        for rhs_ in rhs:
            if (
                rhs_ in self.rhs2evolver and
                evolver.__name__ not in self.evolver_data
            ):
                raise ValueError("rhs keyword '{}' already used")
        evolver_data = {
            "description": evolver.description,
            "options": evolver.options,
            "evolver": evolver,
            "base": None,
            "backstep": None,
            "update_args": None,
            "feedback": None,
            "cte": None,
        }
        evolver_data.update(limits)
        if _test and not self._simple_test(evolver):
            raise ValueError("Could not use given evolver")
        if methods:
            evolver_data["methods"] = methods
            for method in methods:
                self.method2evolver[method] = evolver
        if rhs:
            evolver_data["rhs"] = rhs
            for rhs_ in rhs:
                self.rhs2evolver[rhs_] = evolver
        self._complete_data(evolver, evolver_data)
        self.evolver_data[evolver.__name__] = evolver_data

    def _simple_test(self, evolver):
        return True

    def _complete_data(evolver, evolver_data):
        pass

    def __getitem__[self, key]:
        method, rhs = *key
        try:
            evolver = self.method2evolver[method]
        except KeyError:
            raise KeyError("ode method not found")
        if rhs:
            try:
                composite_evolver = self.rhs2evolver[rhs]
            except KeyError:
                raise KeyError("ode rhs not found")
            if not self.evolver_data[evolver.__name__]['base']:
                # Give the composite_evolver base evolver_data and let it fail?
                raise KeyError("ode method cannot be used with rhs")
            evolver = composite_evolver(evolver)
        return evolver

    def _none2list(self, var):
        var = var or []
        if not isinstance(var, list):
            var = [var]
        return var

    def list_keys(self, etype="methods", limits={}):
        if etype not in ["rhs", "methods"]:
            raise ValueError
        if etype == "rhs":
            names = [val.__name__ for val in self.rhs2evolver.values]
        else:
            names = [val.__name__ for val in self.method2evolver.values]

        names = (name for name in names
                 if all([self.evolver_data[name][key] == val
                             for key, val in limits.items()]))
        return [item
                for item in self.evolver_data[name][etype]
                for name in names]

    def explain_evolvers(self, method=None, rhs=None, names=None,
                          limits={}, full=None):
        method = self._none2list(method)
        rhs = self._none2list(rhs)
        names = self._none2list(names)
        if rhs:
            names.append(self.rhs2evolver[rhs].__name__)
        if method:
            names.append(self.method2evolver[method].__name__)
        if not method and not rhs None and not names:
            names = self.evolver_data.keys()
        if limits:
            names = (name for name in names
                     if all([self.evolver_data[name][key] == val
                             for key, val in limits.items()]))
        names = set(names)
        datas = [self.evolver_data[name] for name in names]
        if (full is None and len(data)<3) or full:
            return "\n\n".join([self._full_print(data) for data in datas])
        else:
            return "\n".join([self._short_print(data) for data in datas])

    def _full_print(self, data):
        out = data['evolver'].__name__ + "\n"
        out += data['description'] + "\n"
        if "methods" in data:
            out = "methods: '" + "' ,'".join(data['methods']) +"'\n"
        else:
            out = "rhs: '" + "' ,'".join(data['rhs']) + "'\n"
        out += "used options: " + str(data['options']) + "'\n"
        support = []
        if not data['cte']:
            support += "time-dependent system supported"
        if data["feedback"]:
            support += "feedback supported"
        if data["backstep"]:
            support += "mcsolve supported"
        out += ", ".join(support)
        return out

    def _short_print(self, data):
        if "methods" in data:
            out = "methods: '" + "' ,'".join(data['methods']) +"': "
        else:
            out = "rhs: '" + "' ,'".join(data['rhs']) +"': "
        out += data['description']
        return out

evolver_collection = _EvolverCollection()


class Evolver:
    """ A wrapper around ODE solvers.
    Ensure a common interface for Solver usage.
    Take and return states as :class:`qutip.core.data.Data`.

    Methods
    -------
    set(state, t0, options)
        Prepare the ODE solver.

    step(t)
        Evolve to t, must be `set` before.

    run(state, tlist)
        Yield (t, state(t)) for t in tlist, must be `set` before.

    update_args(args)
        Change the argument of the active system

    get_state()
        Optain the state of the solver.

    set_state(state, t)
        Set the state of an existing ODE solver.

    """
    used_options = []
    description = ""

    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvo(system, options, args, feedback_args)
        self.stats = self.system.stats
        self.options = options
        self.prepare()

    def prepare(self):
        raise NotImplementedError

    def step(self, t):
        raise NotImplementedError

    def get_state(self, copy=False):
        raise NotImplementedError

    def set_state(self, state0, t):
        raise NotImplementedError

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        for t in tlist[1:]:
            self.step(t)
            yield self.get_state()

    def one_step(self, t):
        self.back = self.get_state(True)
        return self.step(t)

    def backstep(self, t):
        self.set_state(*self.back)
        return self.step(t)

    def update_args(self, args):
        self.system.arguments(args)

    @property
    def stats(self):
        return self.stats
    # "calls": self.system.func_call - self._previous_call


class EvolverScipyZvode(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with zvode solver
    #
    _error_msg = ("ODE integration error: Try to increase "
                  "the allowed number of substeps by increasing "
                  "the nsteps parameter in the Options class.")
    used_options = ['atol', 'rtol', 'nsteps', 'method', 'order',
                    'first_step', 'max_step', 'min_step']
    description = "scipy.integrate.ode using zvode integrator"

    def prepare(self):
        self._ode_solver = ode(self.system.mul_np_vec)
        opt = {key: self.options[key]
               for key in self.used_options
               if key in self.options}
        self._ode_solver.set_integrator('zvode', **opt)
        self.name = "scipy zvode " + opt["method"]

    def get_state(self, copy=False):
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._ode_solver._y),
                self._size,
                inplace=True)
        else:
            state = _data.dense.fast_from_numpy(self._ode_solver._y)
        return t, state.copy() if copy else state

    def set_state(self, state0, t):
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel(),
            t
        )

    def step(self, t):
        if self._ode_solver.t != t:
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()

    def one_step(self, t):
        # integrate(t, step=True) ignore the time and advance one step.
        # Here we want to advance up to t doing maximum one step.
        # So we check if a new step is really needed.
        self.back = self.get_state(copy=True)
        t_front = self._ode_solver._integrator.rwork[12]
        t_ode = self._ode_solver.t
        if t > t_front and t_front <= t_ode:
            self._ode_solver.integrate(t, step=True)
            t_front = self._ode_solver._integrator.rwork[12]
        elif t > t_front:
            t = t_front
        if t_front >= t:
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()

    def backstep(self, t):
        # zvode can step with time lower than the most recent but not all the
        # step interval. (About the last 90%)
        """ Evolve to t, must be `set` before. """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            self.set_state(*self.back)
            self._ode_solver.integrate(t)
        return self.get_state()


class EvolverScipyDop853(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with dop853 solver
    #
    _error_msg = ("ODE integration error: Try to increase "
                  "the allowed number of substeps by increasing "
                  "the nsteps parameter in the Options class.")
    description = "scipy.integrate.ode using dopri5 integrator"
    used_options = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                    'ifactor', 'dfactor', 'beta']

    def prepare(self, options=None):
        self._ode_solver = ode(self.system.mul_np_double_vec)
        opt = {key: self.options[key]
               for key in self.used_options
               if key in self.options}
        self._ode_solver.set_integrator('dop853', **opt)
        self.name = "scipy ode dop853"

    def step(self, t):
        if self._ode_solver.t != t:
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()

    def get_state(self, copy=False):
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._ode_solver.
                                            _y.view(np.complex)),
                self._size,
                inplace=True)
        else:
            state = _data.dense.fast_from_numpy(self._ode_solver.
                                                _y.view(np.complex))
        return t, state.copy() if copy else state

    def set_state(self, state0, t):
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel().view(np.float64),
            t
        )

    def step(self, t):
        """ Evolve to t, must be `set` before. """
        self._ode_solver.integrate(t)
        return self.get_state()

    def one_step(self, t):
        """ Evolve to t, must be `set` before. """
        # dop853 integrator does not support one step and is ineficient
        # when changing direction on integration.
        dt_max = self._ode_solver._integrator.work[6] # allowed max timestep
        dt = t - self._ode_solver.t
        if dt_max * dt < 0: # chande in direction
            self._ode_solver._integrator.reset(len(self._ode_solver._y), False)
            dt_max = -dt_max
        elif dt_max == 0:
            dt_max = 0.01 * dt
        # Will probably do more work than strickly one step if cought in
        # one of the previous conditions, making collapse finding for
        # mcsolve not ideal.
        t = self._ode_solver.t + min(dt_max, dt) if dt > 0 else max(dt_max, dt)
        self._ode_solver.integrate(t)
        return self.get_state()

    def backstep(self, t):
        """ Evolve to t, must be `set` before. """
        self._ode_solver._integrator.reset(len(self._ode_solver._y), False)
        self._ode_solver.integrate(t)
        return self.get_state()


class EvolverScipylsoda(EvolverScipyDop853):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with zvode solver
    #
    _error_msg = ("ODE integration error: Try to increase "
                  "the allowed number of substeps by increasing "
                  "the nsteps parameter in the Options class.")
    used_options = ['atol', 'rtol', 'nsteps', 'max_order_ns', 'max_order_s',
                    'first_step', 'max_step', 'min_step']
    description = "scipy.integrate.ode using lsoda integrator"

    def prepare(self):
        self._ode_solver = ode(self.system.mul_np_double_vec)
        opt = {key: self.options[key]
               for key in self.used_options
               if key in self.options}
        self._ode_solver.set_integrator('lsoda', **opt)
        self.name = "scipy lsoda"

    def one_step(self, t):
        # integrate(t, step=True) ignore the time and advance one step.
        # Here we want to advance up to t doing maximum one step.
        # So we check if a new step is really needed.
        t_front = self._ode_solver._integrator.rwork[12]
        t_back = t_front - self._ode_solver._integrator.rwork[11]
        t_ode = self._ode_solver.t

        if t >= t_back and t <= t_front:
            self._ode_solver.integrate(t)
        elif t_ode < t_front:
            self._ode_solver.integrate(t)
        else:
            self._ode_solver.integrate(t, step=True)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()

    def backstep(self, t):
        """ Evolve to t, must be `set` before. """
        t_front = self._ode_solver._integrator.rwork[12]
        t_back = t_front - self._ode_solver._integrator.rwork[11]
        if t < t_back:
            # Use set_initial_value to reset the direction front the back.
            # Do the same to reset direction to "+" at the end.
            self._ode_solver.integrate(t_back)
            self._ode_solver.set_initial_value(self._ode_solver.y,
                                               self._ode_solver.t)
            self._ode_solver.integrate(t)
            t_front = self._ode_solver._integrator.rwork[12]
            t_back = t_front - self._ode_solver._integrator.rwork[11]
            self._ode_solver.integrate(t_back)
            self._ode_solver.set_initial_value(self._ode_solver.y,
                                               self._ode_solver.t)
        else:
            self._ode_solver.integrate(t)
        return self.get_state()


limits = {
    "base": True,
    "backstep": True,
    "update_args": True,
    "feedback": True,
    "cte": False,
}

evolver_collection.add(EvolverScipyZvode, methods=['adams', 'bdf'],
                       limits=limits, _test=False)

evolver_collection.add(EvolverScipyDop853, methods=['dop853'],
                       limits=limits, _test=False)

evolver_collection.add(EvolverScipylsoda, methods=['lsoda'],
                       limits=limits, _test=False)
