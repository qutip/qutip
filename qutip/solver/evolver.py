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
...
"""
#TODO: DOC

__all__ = ['Evolver',  'evolver_collection', 'EvolverException',
           'EvolverScipyZvode', 'EvolverScipyDop853', 'EvolverScipylsoda']


import numpy as np
from numpy.linalg import norm as la_norm
from itertools import product
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from qutip.core import data as _data
from ._solverqevo import SolverQEvo
from .options import SolverOptions, SolverOdeOptions
from qutip import QobjEvo, qeye, basis
import warnings


class EvolverException(Exception):
    pass


class _EvolverCollection:
    """
    Set of Evolver available to Solver.
    """
    def __init__(self):
        self.method2evolver = {}
        self.rhs2evolver = {}
        self.evolver_data = {}

    def add(self, evolver, methods=[], rhs=[], limits={}, _test=True):
        """
        Add a new evolver to the set available to Solver.
        """
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
        """
        Obtain the evolver corresponding to the key
        """
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
        """ list Evolver available corresponding to the conditions given
        """
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
        """ Describe the Evolver, can choose the evolver from key,
        and/or capacities.
        """
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
    prepare()
        Prepare the ODE solver.

    step(t, copy)
        Evolve to t, must be `prepare` before.
        return the pair t, state.

    get_state()
        Obtain the state of the solver as a pair t, state

    set_state(t, state)
        Set the state of the ODE solver.

    run(state, tlist)
        Yield (t, state(t)) for t in tlist, must be `set` before.

    update_args(args)
        Change the argument of the active system.

    one_step(t, copy):
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        return the pair t, state.

    backstep(t, copy):
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.

    Parameters
    ----------
    sytem: qutip.QobjEvo_base
        Input system controling the evolution.

    options: qutip.SolverOptions
        Options for the solver.

    args: dict
        Arguments passed to the system.

    feedback_args: dict
        Arguments that dependent on the states.
    """
    used_options = []
    description = ""

    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvo(system, options, args, feedback_args)
        self._stats = {}
        self.options = options
        self.prepare()

    def prepare(self):
        """
        Initialize the solver
        """
        raise NotImplementedError

    def step(self, t, copy=True):
        """
        Evolve to t, must be `prepare` before.
        return the pair t, state.
        """
        raise NotImplementedError

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        raise NotImplementedError

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.
        """
        raise NotImplementedError

    def run(self, tlist):
        """
        Yield (t, state(t)) for t in tlist, must be `set` before.
        """
        for t in tlist[1:]:
            yield self.step(t, False)

    def one_step(self, t, copy=True):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        self.back = self.get_state(True)
        return self.step(t, copy)

    def backstep(self, t, copy=True):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        self.set_state(*self.back)
        return self.step(t, copy)

    def update_args(self, args):
        """
        Change the argument of the active system.
        """
        self.system.arguments(args)

    def update_feedback(self, what, data):
        if hasattr(self, "system") and isinstance(self.system, SolverQEvo):
            self.system.update_feedback(what, data)

    @property
    def stats(self):
        """
        Return statistic of the evolution as a dict
        """
        if not hasattr(self, "_stats"):
            self._stats = {}
        try:
            self._stats.update(self.system.stats)
        except:
            pass
        return self._stats


class EvolverScipyZvode(Evolver):
    """
    Evolver using Scipy `ode` interface for netlib zvode integrator.
    """
    used_options = ['atol', 'rtol', 'nsteps', 'method', 'order',
                    'first_step', 'max_step', 'min_step']
    description = "scipy.integrate.ode using zvode integrator"

    def prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self.system.mul_np_vec)
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver.set_integrator('zvode', **opt)
        self.name = "scipy zvode " + opt["method"]

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.Dense(self._ode_solver._y, copy=False),
                self._size,
                inplace=True)
        else:
            state = _data.dense.Dense(self._ode_solver._y, copy=False)
        return t, state.copy() if copy else state

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.
        """
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel(),
            t
        )

    def step(self, t, copy=True):
        """ Evolve to t, must be `set` before. """
        if self._ode_solver.t != t:
            self._ode_solver.integrate(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def one_step(self, t, copy=True):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
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
        self._check_failed_integration()
        return self.get_state(copy)

    def backstep(self, t, copy=True):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        # zvode can step with time lower than the most recent but not all the
        # step interval.
        # About 90% of the last step interval?
        # It return a warning, not an Error.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            self.set_state(*self.back)
            self._ode_solver.integrate(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'Excess work done on this call. Try to increasing '
                'the nsteps parameter in the Options class',
            -2: 'Excess accuracy requested. (Tolerances too small.)',
            -3: 'Illegal input detected.',
            -4: 'Repeated error test failures. (Check all input.)',
            -5: 'Repeated convergence failures. (Perhaps bad'
                ' Jacobian supplied or wrong choice of MF or tolerances.)',
            -6: 'Error weight became zero during problem. (Solution'
                ' component i vanished, and ATOL or ATOL(i) = 0.)'
        }
        raise EvolverException(messages[self._ode_solver._integrator.istate])


class EvolverScipyDop853(Evolver):
    """
    Evolver using Scipy `ode` interface with dop853 integrator.
    """
    description = "scipy.integrate.ode using dop853 integrator"
    used_options = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                    'ifactor', 'dfactor', 'beta']

    def prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self.system.mul_np_double_vec)
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver.set_integrator('dop853', **opt)
        self.name = "scipy ode dop853"

    def step(self, t, copy=True):
        """
        Evolve to t, must be `set` before.
        """
        if self._ode_solver.t != t:
            self._ode_solver.integrate(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.Dense(self._ode_solver._y.view(np.complex128),
                                  copy=False),
                self._size,
                inplace=True)
        else:
            state = _data.dense.Dense(self._ode_solver._y.view(np.complex128),
                                      copy=False)
        return t, state.copy() if copy else state

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.
        """
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel().view(np.float64),
            t
        )

    def one_step(self, t, copy=True):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        self.back = self.get_state(True)
        return self._safe_step(t, copy)

    def backstep(self, t, copy=True):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        self.set_state(*self.back)
        return self._safe_step(t)

    def _safe_step(self, t, copy=True):
        """
        DOP853 ODE does extra work when changing direction.
        This reset the state to save this extra work.
        """
        dt_max = self._ode_solver._integrator.work[6]
        dt = t - self._ode_solver.t
        if dt == 0:
            return self.get_state(copy)
        if dt * dt_max < 0:
            self.set_state(*self.get_state())
        out = self.step(t, copy)
        if self._ode_solver._integrator.work[6] < 0:
            self.set_state(*self.get_state())
        return out

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'input is not consistent',
            -2: 'larger nsteps is needed, Try to increase the nsteps '
                'parameter in the Options class.',
            -3: 'step size becomes too small. Try increasing tolerance',
            -4: 'problem is probably stiff (interrupted), try "bdf" '
                'method instead',
        }
        raise EvolverException(messages[self._ode_solver._integrator.istate])


class EvolverScipylsoda(EvolverScipyDop853):
    """
    Evolver using Scipy `ode` interface of lsoda integrator from netlib.
    """
    used_options = ['atol', 'rtol', 'nsteps', 'max_order_ns', 'max_order_s',
                    'first_step', 'max_step', 'min_step']
    description = "scipy.integrate.ode using lsoda integrator"

    def prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self.system.mul_np_double_vec)
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver.set_integrator('lsoda', **opt)
        self.name = "scipy lsoda"

    def one_step(self, t, copy=True):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
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
        self._check_failed_integration()
        return self.get_state(copy)

    def backstep(self, t, copy=True):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        # like zvode, lsoda can step with time lower than the most recent.
        # But not all the step interval. (About the last 90%)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            self.set_state(*self.back)
            self._ode_solver.integrate(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: "Excess work done on this call."
                "Try to increase the nsteps parameter in the Options class.",
            -2: "Excess accuracy requested (tolerances too small).",
            -3: "Illegal input detected (internal error).",
            -4: "Repeated error test failures (internal error).",
            -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
            -6: "Error weight became zero during problem.",
            -7: "Internal workspace insufficient to finish (internal error)."
        }
        raise EvolverException(messages[self._ode_solver._integrator.istate])


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
