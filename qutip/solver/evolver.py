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
__all__ = ['Evolver',  'evolver_collection',
           'EvolverScipyZvode', 'EvolverScipyDop853', 'EvolverScipylsoda']


import numpy as np
from numpy.linalg import norm as la_norm
from itertools import product
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from ..core import data as _data
from ._solverqevo import SolverQEvo
from .options import SolverOptions, SolverOdeOptions
from ..core import QobjEvo, qeye, basis
import warnings


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

        if _test and not self._simple_test(evolver, bool(methods)):
            raise ValueError("Could not use given evolver")
        evolver_data = self._complete_data(evolver, limits, bool(methods))
        if methods:
            evolver_data["methods"] = methods
            for method in methods:
                self.method2evolver[method] = evolver
        if rhs:
            evolver_data["rhs"] = rhs
            for rhs_ in rhs:
                self.rhs2evolver[rhs_] = evolver
        self.evolver_data[evolver.__name__] = evolver_data

    def _simple_test(self, evolver, method):
        system = QobjEvo(qeye(1))
        try:
            if method:
                evo = evolver(system, SolverOptions(), {}, {})
            else:
                evo = evolver(self["dop853", ""])(system, {}, {}, {})
            evo.set_state(0., basis(1,0).data)
            assert np.all_close(evo.step(1)[1].to_array()[0,0], 2., 1e-5)
        except Exception:
            return False
        return True

    def _complete_data(self, evolver, limits, method):
        evolver_data = {
            "description": "",
            "options": [],
            "evolver": evolver,
            "base": None,
            "backstep": None,
            "update_args": None,
            "feedback": None,
            "time_dependent": None,
        }
        evolver_data.update(limits)
        if hasattr(evolver, 'description'):
            evolver_data['description'] = evolver.description
        if hasattr(evolver, 'used_options'):
            evolver_data['options'] = evolver.used_options
            [SolverOdeOptions.extra_options.add(opt)
             for opt in evolver.used_options]

        if not method:
            evol = evolver(self["dop853", ""])
            evolver_data['base'] = False

        if evolver_data['time_dependent'] is None:
            try:
                system = QobjEvo([qeye(1), lambda t, args: t])
                evo = evol(system, SolverOptions(), {}, {})
                evo.set_state(0, basis(1,0).data)
                assert np.all_close(
                    evo.step(1)[1].to_array()[0,0], 1.5, atol=1e-5
                    )
                evolver_data['time_dependent'] = True
            except Exception:
                evolver_data['time_dependent'] = False
                evolver_data['update_args'] = False
                evolver_data['feedback'] = False
                evolver_data['base'] = False

        if evolver_data['update_args'] is None:
            try:
                system = QobjEvo([qeye(1), lambda t, args: args['cte']])
                evo = evol(system, SolverOptions(), {"cte":0}, {})
                evo.set_state(0, basis(1,0).data)
                evo.step(0.5)
                evo.update_args({"cte":1})
                assert np.all_close(
                    evo.step(1)[1].to_array()[0,0], 0.5, atol=1e-5
                    )
                evolver_data['update_args'] = True
            except Exception:
                evolver_data['update_args'] = False
                evolver_data['base'] = False

        if evolver_data['feedback'] is None:
            try:
                system = QobjEvo([qeye(1), lambda t, args: args['obj'].full()[0,0]])
                evo = evol(system, SolverOptions(), {"obj":basis(2,0)}, {})
                evo.set_state(0, basis(1,0).data)
                assert np.all_close(
                    evo.step(1)[1].to_array()[0,0], np.exp(1), atol=1e-3
                    )
                evolver_data['feedback'] = True
            except Exception:
                evolver_data['feedback'] = False
                evolver_data['base'] = False

        if evolver_data['backstep'] is None:
            try:
                system = QobjEvo([qeye(1), lambda t, args: t])
                evo = evol(system, SolverOptions(), {}, {})
                evo.set_state(0, basis(1,0).data)
                t, _ = evo.step(0.5)
                for t_target in [0.6, 1.0]:
                    t_old = t
                    while t < t_target:
                        t_old = t
                        t, state = evo.stepping(t_target)
                    for t_backstep in np.linspace(t_old, t, 21):
                        # since scipy zvode naturally backstep on most but not
                        # all the range, we test on the full range.
                        _, state = evo.backstep(t_backstep)
                        assert np.all_close(state.to_array()[0,0],
                                            1 + t_backstep, atol=1e-5)
                evolver_data['backstep'] = True
            except Exception:
                evolver_data['backstep'] = False
                evolver_data['base'] = False

        if evolver_data['base'] is None:
            if not isinstance(evo.system, SolverQEvo):
                evolver_data['base'] = False
            else:
                try:
                    system = QobjEvo(qeye(1))
                    evo = evol(system, SolverOptions(), {}, {})
                    evo.system = SolverQEvo(QobjEvo(qeye(1)*-1),
                                            SolverOptions(), {}, {})
                    evo.set_state(0, basis(1,0).to(_data.Dense).data)
                    assert  np.all_close(
                        evo.step(1)[1].to_array()[0,0], 0., atol=1e-5
                        )
                    evolver_data['base'] = True
                except Exception:
                    evolver_data['base'] = False
        return evolver_data

    def __getitem__(self, key):
        method, rhs = key
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

    def list_keys(self, etype="methods", **limits):
        if etype not in ["rhs", "methods", "pairs"]:
            raise ValueError
        if etype in ["rhs", 'pairs']:
            names = [val.__name__ for val in self.rhs2evolver.values()]
        else:
            names = [val.__name__ for val in self.method2evolver.values()]

        for key in limits:
            if key not in [
                'base',
                "backstep",
                "update_args",
                "feedback",
                "time_dependent",
            ]:
                raise ValueError("key " + key + "not available.")

        names = (name for name in names
                 if all([self.evolver_data[name][key] == val
                             for key, val in limits.items()]))
        keys = set([item
                    for name in names
                    for item in self.evolver_data[name][etype]
                   ])
        if etype == 'pairs':
            methods = {(key, "") for key in self.list_keys('methods', **limits)}
            bases = self.list_keys('methods', base=True)
            return list(methods.union({pair for pair in product(bases, keys)}))
        else:
            return list(keys)

    def explain_evolvers(self, method=None, rhs=None, names=None,
                         limits={}, full=None):
        method = self._none2list(method)
        rhs = self._none2list(rhs)
        names = self._none2list(names)
        if rhs:
            names.append(self.rhs2evolver[rhs].__name__)
        if method:
            names.append(self.method2evolver[method].__name__)
        if not method and not rhs and not names:
            names = self.evolver_data.keys()
        if limits:
            names = (name for name in names
                     if all([self.evolver_data[name][key] == val
                             for key, val in limits.items()]))
        names = set(names)
        datas = [self.evolver_data[name] for name in names]
        if (full is None and len(datas)<3) or full:
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
        if data['time_dependent']:
            support += ["time-dependent system supported"]
        if data["feedback"]:
            support += ["feedback supported"]
        if data["backstep"]:
            support += ["mcsolve supported"]
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

    set_state(t, state)
        Set the state of an existing ODE solver.

    """
    used_options = []
    description = ""

    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvo(system, options, args, feedback_args)
        self._stats = {}
        self.options = options
        self.prepare()

    def prepare(self):
        raise NotImplementedError

    def step(self, t):
        raise NotImplementedError

    def get_state(self, copy=False):
        raise NotImplementedError

    def set_state(self, t, state0):
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
        try:
            self._stats.update(self.system.stats)
        except:
            pass
        return self._stats
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
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
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

    def set_state(self, t, state0):
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

    def prepare(self):
        self._ode_solver = ode(self.system.mul_np_double_vec)
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
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

    def set_state(self, t, state0):
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel().view(np.float64),
            t
        )

    def one_step(self, t):
        self.back = self.get_state(True)
        return self._safe_step(t)

    def backstep(self, t):
        self.set_state(*self.back)
        return self._safe_step(t)

    def _safe_step(self, t):
        """step but safe when changing direction"""
        dt_max = self._ode_solver._integrator.work[6]
        dt = t - self._ode_solver.t
        if dt == 0:
            return self.get_state()
        if dt * dt_max < 0:
            self.set_state(*self.get_state())
        out = self.step(t)
        if self._ode_solver._integrator.work[6] < 0:
            self.set_state(*self.get_state())
        return out


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
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver.set_integrator('lsoda', **opt)
        self.name = "scipy lsoda"

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


limits = {
    "base": True,
    "backstep": True,
    "update_args": True,
    "feedback": True,
    "time_dependent": True,
}

evolver_collection.add(EvolverScipyZvode, methods=['adams', 'bdf'],
                       limits=limits, _test=False)

evolver_collection.add(EvolverScipyDop853, methods=['dop853'],
                       limits=limits, _test=False)

evolver_collection.add(EvolverScipylsoda, methods=['lsoda'],
                       limits=limits, _test=False)
