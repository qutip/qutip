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
""" Define `Integrator`: ODE solver wrapper to use in qutip's Solver """


__all__ = ['Integrator', 'integrator_collection', 'IntegratorException']


from itertools import product
from .options import SolverOdeOptions
from qutip import QobjEvo, qeye, basis
from functools import partial


class IntegratorException(Exception):
    pass


class Integrator:
    """
    A wrapper around ODE solvers.
    It ensure a common interface for Solver usage.
    It takes and return states as :class:`qutip.core.data.Data`, it may return
    a different data-type than the input type.

    Parameters
    ----------
    sytem: qutip.QobjEvo
        Quantum system in which states evolve.

    options: qutip.SolverOptions
        Options for the solver.
    """
    used_options = []
    description = ""

    def __init__(self, system, options):
        self.system = system
        self._stats = {}
        self.options = options
        self._prepare()

    def _prepare(self):
        """
        Initialize the solver
        """
        raise NotImplementedError

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.
        """
        raise NotImplementedError

    def integrate(self, t, step=False, copy=True):
        """
        Evolve to t.

        If `step` advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time with subsequent calls.

        Before calling `integrate` for the first time, the initial step should
        be set with `set_state`.

        Parameters
        ----------
        t : float
            Time to integrate to, should be larger than the previous time. If
            the last integrate call was use with ``step=True``, the time can be
            between the time at the start of the last call and now.

        step : bool [False]
            When True, integrate for a one internal step of the ODE solver.

        copy : bool [True]
            Whether to return a copy of the state or the state itself.

        Return
        ------
        (t, state) : (float, qutip.Data)
            The state of the solver at ``t``. The returned this can differ from
            the input time only when ``step=True``.
        """
        raise NotImplementedError

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair (t, state).

        Return
        ------
        (t, state) : (float, qutip.Data)
            The state of the solver at ``t``.
        """
        raise NotImplementedError

    def run(self, tlist):
        """
        Integrate the system yielding the state for each times in tlist.

        Parameters
        ----------
        tlist : *list* / *array*
            List of times to yield the state.

        Yield
        -----
        (t, state) : (float, qutip.Data)
            The state of the solver at each ``t`` of tlist.
        """
        for t in tlist[1:]:
            yield self.integrate(t, False)

    def reset(self):
        """Reset internal state of the ODE solver."""
        self.set_state(*self.get_state)

    def update_args(self, args):
        """
        Change the argument of the system.
        Reset the ODE solver to ensure numerical validity.

        Parameters
        ----------
        args : dict
            New arguments
        """
        self.system.arguments(args)
        self.reset()


class _IntegratorCollection:
    """
    Collection of ODE :obj:`Integrator` available to Qutip's solvers.

    :obj:`Integrator` are composed of 2 parts: `method` and `rhs`:
    `method` are ODE integration method such as Runge-Kutta or Adams–Moulton.
    `rhs` are options to control the :obj:`QobjEvo`'s matmul function.

    Parameters
    ----------
    known_solvers : list of str
        list of solver using this ensemble of integrator

    options_class : :func:optionsclass decorated class
        Option object to add integrator's option to the accepted keys.
    """
    def __init__(self, known_solvers, options_class):
        self.known_solvers = known_solvers
        self.options_class = options_class
        # map from method key to integrator
        self.method2integrator = {}
        # map from rhs key to rhs function
        self.rhs2system = {}
        # methods's keys which support alternative rhs
        self.base_methods = []
        # Information about methods
        self.method_data = {}
        # Information about rhs
        self.rhs_data = {}

    def add_method(self, integrator, keys, solver,
                   use_QobjEvo_matmul, time_dependent):
        """
        Add a new integrator to the set available to solvers.

        Parameters
        ----------

        integrator : class derived from :obj:`Integrator`
            New integrator to add.

        keys : list of str
            List of keys supported by the integrator.
            When `options["methods"] in keys`, this integrator will be used.

        solver : list of str
            List of the [...]solve function that are supported by the
            integrator.

        use_QobjEvo_matmul : bool
            Whether the Integrator use `QobjEvo.matmul` as the function of the
            ODE. When `False`, rhs cannot be used.

        time_dependent : bool
            Whether integrator support time-dependent system.
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            if key in self.method2integrator:
                raise ValueError("method '{}' already used".format(key))

        integrator_data = self._complete_data(integrator, solver,
                                              use_QobjEvo_matmul,
                                              time_dependent)
        for key in keys:
            self.method2integrator[key] = integrator
            self.method_data[key] = integrator_data
        if use_QobjEvo_matmul:
            self.base_methods += keys

    def add_rhs(self, integrator, keys, solver, time_dependent):
        """
        Add a new rhs to the set available to solvers.

        Parameters
        ----------

        integrator : callable
            Function with the signature::

                rhs(
                    integrator: class,
                    system: QobjEvo,
                    options: SolverOptions
                ) -> Integrator

            that create the :obj:`Integrator` instance. The integrator can be
            any integrator registered that has `"use_QobjEvo_matmul" == True`.
            `system` is the :obj:`QobjEvo` driving the ODE: 'L', '-i*H', etc.
            `options` is the :obj:`SolverOptions` of the solver.

        keys : list of str
            List of keys supported by the integrator.
            When `options["methods"] in keys`, this integrator will be used.

        solver : list of str
            List of the [...]solve function that are supported by the
            integrator.

        time_dependent : bool
            True if the integrator can solve time dependent systems.
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            if key in self.rhs2system:
                raise ValueError("rhs keyword '{}' already used")
        integrator_data = self._complete_data(integrator, solver,
                                              True,
                                              time_dependent)
        for key in keys:
            self.rhs2system[key] = integrator
            self.rhs_data[key] = integrator_data

    def _complete_data(self, integrator, solver, use_QobjEvo_matmul, td):
        """
        Get the information for the integrator.
        """
        integrator_data = {
            "integrator": integrator,
            "description": "",
            # options used by the integrator, to add to the accepted option
            # by the SolverOptions object.
            "options": [],
            # list of supported solver, sesolve, mesolve, etc.
            "solver": [],
            # The `method` use QobjEvo's matmul.
            # If False, refuse `rhs` option.
            "use_QobjEvo_matmul": use_QobjEvo_matmul,
            # Support of time-dependent system
            "time_dependent": td,
        }
        for sol in solver:
            if sol not in self.known_solvers:
                raise ValueError(f"Unknown solver '{sol}', known solver are"
                                 + str(self.known_solvers))
            integrator_data["solver"].append(sol)

        if hasattr(integrator, 'description'):
            integrator_data['description'] = integrator.description

        if hasattr(integrator, 'used_options'):
            integrator_data['options'] = integrator.used_options
            for opt in integrator.used_options:
                self.options_class.extra_options.add(opt)

        return integrator_data

    def __getitem__(self, key):
        """
        Obtain the integrator from the (method, rhs) key pair.
        """
        method, rhs = key
        try:
            integrator = self.method2integrator[method]
        except KeyError:
            raise KeyError(f"ode method {method} not found")
        if rhs == "":
            build_func = prepare_integrator
        elif self.method_data[method]["use_QobjEvo_matmul"]:
            try:
                build_func = self.rhs2system[rhs]
            except KeyError:
                raise KeyError(f"ode rhs {rhs} not found")
        else:
            raise KeyError(f"ode method {method} does not support rhs")
        return partial(build_func, integrator)

    def _list_keys(self, keytype="methods", solver="",
                   use_QobjEvo_matmul=None, time_dependent=None):
        """
        List integrators available corresponding to the conditions given.
        Used in test.
        """
        if keytype == "methods":
            names = [method for method in self.method2integrator
                     if self._check_condition(method, "", solver,
                                             use_QobjEvo_matmul,
                                             time_dependent)]
        elif keytype == "rhs":
            names = [rhs for rhs in self.rhs2system
                     if self._check_condition("", rhs, solver,
                                             use_QobjEvo_matmul,
                                             time_dependent)]
        elif keytype == "pairs":
            names = [(method, "") for method in self.method2integrator
                     if self._check_condition(method, "", solver,
                                             use_QobjEvo_matmul,
                                             time_dependent)]
            names += [(method, rhs)
                for method, rhs in product(self.base_methods, self.rhs2system)
                if rhs and self._check_condition(method, rhs, solver,
                    use_QobjEvo_matmul, time_dependent)
            ]
        else:
            raise ValueError("keytype must be one of "
                             "'rhs', 'methods' or 'pairs'")
        return names

    def _check_condition(self, method, rhs, solver="",
                         use_QobjEvo_matmul=None, time_dependent=None):
        """
        Verify if the pair (method, rhs) can be used for the given solver
        and whether it support the desired capacities.
        """
        if method in self.method_data:
            data = self.method_data[method]
            if solver and solver not in data['solver']:
                return False
            if (
                use_QobjEvo_matmul is not None and
                use_QobjEvo_matmul != data['use_QobjEvo_matmul']
            ):
                return False
            if (
                time_dependent is not None and
                time_dependent != data['time_dependent']
            ):
                return False

        if rhs in self.rhs_data:
            data = self.rhs_data[rhs]
            if solver and solver not in data['solver']:
                return False
            if (
                time_dependent is not None and
                time_dependent != data['time_dependent']
            ):
                return False

        return True


integrator_collection = _IntegratorCollection(
    ['sesolve', 'mesolve', 'mcsolve'],
    SolverOdeOptions
)

def prepare_integrator(integrator, system, options):
    """Default rhs function"""
    return integrator(system, options)

integrator_collection.add_rhs(prepare_integrator, "",
                              ['sesolve', 'mesolve', 'mcsolve'], True)
