# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['Solver']

from numpy.typing import ArrayLike
from numbers import Number
from typing import Any, Callable
from .. import Qobj, QobjEvo, ket2dm
from .options import _SolverOptions
from ..core import stack_columns, unstack_columns
from .. import settings
from .result import Result
from .integrator import Integrator
from ..ui.progressbar import progress_bars
from ._feedback import _ExpectFeedback
from ..typing import EopsLike
from time import time
import warnings
import numpy as np


class Solver:
    """
    Runner for an evolution.
    Can run the evolution at once using :meth:`run` or step by step using
    :meth:`start` and :meth:`step`.

    Parameters
    ----------
    rhs : :obj:`.Qobj`, :obj:`.QobjEvo`
        Right hand side of the evolution::
            d state / dt = rhs @ state

    options : dict
        Options for the solver
    """
    name = ""

    # State, time and Integrator of the stepper functionnality
    _integrator = None
    _avail_integrators = {}

    # Class of option used by the solver
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        "method": "adams",
    }
    _resultclass = Result

    def __init__(self, rhs, *, options=None):
        if isinstance(rhs, (QobjEvo, Qobj)):
            self.rhs = QobjEvo(rhs)
        else:
            raise TypeError("The rhs must be a QobjEvo")
        self.options = options
        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()
        self.rhs._register_feedback({}, solver=self.name)

    def _initialize_stats(self):
        """ Return the initial values for the solver stats.
        """
        return {
            "method": self._integrator.name,
            "init time": self._init_integrator_time,
            "preparation time": 0.0,
            "run time": 0.0,
        }

    def _prepare_state(self, state):
        """
        Extract the data of the Qobj state.

        Is responsible for dims checks, preparing the data (stack columns, ...)
        determining the dims of the output for :meth:`_restore_state`.

        Should return the state's data such that it can be used by Integrators.
        """
        if self.rhs.issuper and state.isket:
            state = ket2dm(state)

        if (
            self.rhs.dims[1] != state.dims[0]
            and self.rhs.dims[1] != state.dims
        ):
            raise TypeError(f"incompatible dimensions {self.rhs.dims}"
                            f" and {state.dims}")

        self._state_metadata = {
            'dims': state._dims,
            # This is herm flag take for granted that the liouvillian keep
            # hermiticity.  But we do not check user passed super operator for
            # anything other than dimensions.
            'isherm': not (self.rhs.dims == state.dims) and state._isherm,
        }
        if state.isket:
            norm = state.norm()
        elif state._dims.issquare:
            # Qobj.isoper does not differientiate between rectangular operators
            # and normal ones.
            norm = state.tr()
        else:
            norm = -1
        self._normalize_output = (
            self._options.get("normalize_output", False)
            # Don't normalize output if input is not normalized.
            # Use the settings atol instead of the solver one since the second
            # refer to the ODE tolerance and some integrator do not use it.
            and np.abs(norm - 1) <= settings.core["atol"]
            # Only ket and dm can be normalized
            and (self.rhs.dims[1] == state.dims or state.shape[1] == 1)
        )
        if self.rhs.dims[1] == state.dims:
            return stack_columns(state.data)
        return state.data

    def _restore_state(self, data, *, copy=True):
        """
        Retore the Qobj state from its data.
        """
        if self._state_metadata['dims'] == self.rhs._dims[1]:
            state = Qobj(unstack_columns(data),
                         **self._state_metadata, copy=False)
        else:
            state = Qobj(data, **self._state_metadata, copy=copy)

        if self._normalize_output:
            if state.isoper:
                state = state * (1 / state.tr())
            else:
                state = state * (1 / state.norm())

        return state

    def run(
        self,
        state0: Qobj,
        tlist: ArrayLike,
        *,
        e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
        args: dict[str, Any] = None,
    ) -> Result:
        """
        Do the evolution of the Quantum system.

        For a ``state0`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :class:`.Result`. The evolution method and
        stored results are determined by ``options``.

        Parameters
        ----------
        state0 : :obj:`.Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Each times of the list must be increasing, but does not
            need to be uniformy distributed.

        args : dict, optional
            Change the ``args`` of the rhs for the evolution.

        e_ops : Qobj, QobjEvo, callable, list, or dict optional
            Single, list or dict of Qobj, QobjEvo or callable to compute the
            expectation values. Function[s] must have the signature
            f(t : float, state : Qobj) -> expect.

        Returns
        -------
        results : :obj:`.Result`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        _time_start = time()
        _data0 = self._prepare_state(state0)
        self._integrator.set_state(tlist[0], _data0)
        self._argument(args)
        stats = self._initialize_stats()
        results = self._resultclass(
            e_ops, self.options,
            solver=self.name, stats=stats,
        )
        results.add(tlist[0], self._restore_state(_data0, copy=False))
        stats['preparation time'] += time() - _time_start

        progress_bar = progress_bars[self.options['progress_bar']](
            len(tlist)-1, **self.options['progress_kwargs']
        )
        for t, state in self._integrator.run(tlist):
            progress_bar.update()
            results.add(t, self._restore_state(state, copy=False))
        progress_bar.finished()

        stats['run time'] = progress_bar.total_time()
        # TODO: It would be nice if integrator could give evolution statistics
        # stats.update(_integrator.stats)
        return results

    def start(self, state0: Qobj, t0: Number) -> None:
        """
        Set the initial state and time for a step evolution.

        Parameters
        ----------
        state0 : :obj:`.Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.
        """
        _time_start = time()
        self._integrator.set_state(t0, self._prepare_state(state0))
        self.stats["preparation time"] += time() - _time_start

    def step(
        self,
        t: Number,
        *,
        args: dict[str, Any] = None,
        copy: bool = True
    ) -> Qobj:
        """
        Evolve the state to ``t`` and return the state as a :obj:`.Qobj`.

        Parameters
        ----------
        t : double
            Time to evolve to, must be higher than the last call.

        args : dict, optional {None}
            Update the ``args`` of the system.
            The change is effective from the beginning of the interval.
            Changing ``args`` can slow the evolution.

        copy : bool, optional {True}
            Whether to return a copy of the data or the data in the ODE solver.

        Notes
        -----
        The state must be initialized first by calling :meth:`start` or
        :meth:`run`. If :meth:`run` is called, :meth:`step` will continue from
        the last time and state obtained.

        """
        if not self._integrator._is_set:
            raise RuntimeError("The `start` method must called first")
        _time_start = time()
        self._argument(args)
        _, state = self._integrator.integrate(t, copy=False)
        self.stats["run time"] += time() - _time_start
        return self._restore_state(state, copy=copy)

    def _get_integrator(self):
        """ Return the initialted integrator. """
        _time_start = time()
        method = self._options["method"]
        if method in self.avail_integrators():
            integrator = self.avail_integrators()[method]
        elif issubclass(method, Integrator):
            integrator = method
        else:
            raise ValueError("Integrator method not supported.")
        integrator_instance = integrator(self.rhs, self.options)
        self._init_integrator_time = time() - _time_start
        return integrator_instance

    @property
    def sys_dims(self):
        """
        Dimensions of the space that the system use:

        ``qutip.basis(sovler.dims)`` will create a state with proper dimensions
        for this solver.
        """
        return self.rhs.dims[0]

    @property
    def options(self) -> dict[str, Any]:
        """
        method: str
            Which ordinary differential equation integration method to use.
        """
        return self._options

    def _parse_options(self, new_options, default, old_options):
        """
        Do a first read through of the options:
        - Split new options' items included in the default from those that are
          not.
        - Replace ``None`` with the default value.
        - Remove items that are unchanged.
        """
        included_options = {
            key: val
            for key, val in new_options.items()
            if key in default
        }
        extra_options = {
            key: val
            for key, val in new_options.items()
            if key not in default
        }

        # First pass: Have ``None`` refert to the default.
        included_options = {
            key: (val if val is not None else default[key])
            for key, val in included_options.items()
        }
        # Second pass: Remove options that remain unchanged.
        included_options = {
            key: val
            for key, val in included_options.items()
            if not (key in old_options and val == old_options[key])
        }
        return included_options, extra_options

    @options.setter
    def options(self, new_options: dict[str, Any]):
        if not hasattr(self, "_options"):
            self._options = {}
        if new_options is None:
            new_options = {}
        if not isinstance(new_options, dict):
            raise TypeError("options most to be a dictionary.")
        new_solver_options, new_ode_options = self._parse_options(
            new_options, self.solver_options, self.options
        )
        method = new_solver_options.get(
            "method", self.options.get("method", self.solver_options["method"])
        )
        integrator = self.avail_integrators()[method]

        if method == self.options.get("method", None):
            # If method changed, drop old ode options
            old_ode_options = self.options
            old_options = self._options
        else:
            old_ode_options = {}
            old_options, _ = self._parse_options(
                self._options, self.solver_options, {}
            )

        new_ode_options, extra_options = self._parse_options(
            new_ode_options, integrator.integrator_options, self.options
        )
        if extra_options:
            raise KeyError(f"Options {extra_options.keys()} are not supported")
        if self._options and not (new_solver_options or new_ode_options):
            return  # Nothing to do

        self._options = _SolverOptions(
            {**self.solver_options, **integrator.integrator_options},
            self._apply_options,
            self.name + " with " + integrator.method + " integrator",
            self.__class__.options.__doc__ + integrator.options.__doc__,
            **{**old_options, **new_solver_options, **new_ode_options}
        )

        self._apply_options(set(new_options.keys()))

    def _apply_options(self, keys):
        """
        Method called when options are changed, either through
        ``solver.options[key] = value`` or ``solver.options = options``.
        Allow to update the solver with the new options
        """
        from_setter = isinstance(keys, (set))
        if not from_setter:
            keys = set([keys])

        if not from_setter and "method" in keys:
            # Drop the ode's options.
            old_solver_options, _ = self._parse_options(
                self.options, self.solver_options, {}
            )
            method = self.options["method"]
            integrator = self.avail_integrators()[method]

            self._options = _SolverOptions(
                {**self.solver_options, **integrator.integrator_options},
                self._apply_options,
                self.name + " with " + integrator.method + " integrator",
                self.__class__.options.__doc__ + integrator.options.__doc__,
                **old_solver_options
            )

        if self._integrator is None or not keys:
            pass
        elif 'method' in keys and self._integrator._is_set:
            state = self._integrator.get_state()
            self._integrator = self._get_integrator()
            self._integrator.set_state(*state)
        elif "method" in keys:
            self._integrator = self._get_integrator()
        elif keys & self._integrator.integrator_options.keys():
            # Some of the keys are used by the integrator.
            self._integrator.options = self._options
            self._integrator.reset(hard=True)

    def _argument(self, args):
        """Update the args, for the `rhs` and other operators."""
        if args:
            self.rhs.arguments(args)
            self._integrator.arguments(args)

    @classmethod
    def avail_integrators(cls):
        if cls is Solver:
            return cls._avail_integrators.copy()
        return {
            **Solver.avail_integrators(),
            **cls._avail_integrators,
        }

    @classmethod
    def integrator(cls, key):
        return cls.avail_integrators()[key]

    @classmethod
    def add_integrator(cls, integrator, key):
        """
        Register an integrator.

        Parameters
        ----------
        integrator : Integrator
            The ODE solver to register.

        keys : list of str
            Values of the method options that refer to this integrator.
        """
        if not issubclass(integrator, Integrator):
            raise TypeError(f"The integrator {integrator} must be a subclass"
                            " of `qutip.solver.Integrator`")

        cls._avail_integrators[key] = integrator

    @classmethod
    def ExpectFeedback(cls, operator: Qobj | QobjEvo, default: Any = 0.):
        """
        Expectation value of the instantaneous state of the evolution to be
        used by a time-dependent operator.

        When used as an args:

            ``QobjEvo([op, func], args={"E0": Solver.ExpectFeedback(oper)})``

        The ``func`` will receive ``expect(oper, state)`` as ``E0`` during the
        evolution.

        Parameters
        ----------
        operator : Qobj, QobjEvo
            Operator to compute the expectation values of.

        default : float, default : 0.
            Initial value to be used at setup.
        """
        return _ExpectFeedback(operator, default)


def _solver_deprecation(kwargs, options, solver="me"):
    """
    Function to help the transition from v4 to v5.
    Raise warnings for solver input that where moved from parameter to options.
    """
    if options is None:
        options = {}
    # TODO remove by 5.1
    if "progress_bar" in kwargs:
        warnings.warn(
            '"progress_bar" is now included in options:\n Use '
            '`options={"progress_bar": False / True / "tqdm" / "enhanced"}`',
            FutureWarning
        )
        options["progress_bar"] = kwargs.pop("progress_bar")

    if "_safe_mode" in kwargs:
        warnings.warn(
            '"_safe_mode" is no longer supported.',
            FutureWarning
        )
        del kwargs["_safe_mode"]

    if "verbose" in kwargs and solver == "br":
        warnings.warn(
            '"verbose" is no longer supported.',
            FutureWarning
        )
        del kwargs["verbose"]

    if "tol" in kwargs and solver == "br":
        warnings.warn(
            'The "tol" parameter is no longer used. '
            '`qutip.settings.core["auto_tidyup_atol"]` '
            'is now used for rounding small values in sparse arrays.',
            FutureWarning
        )
        del kwargs["tol"]

    if "map_func" in kwargs and solver in ["mc", "stoc"]:
        warnings.warn(
            '"map_func" is now included in options:\n'
            'Use `options={"map": "serial" / "parallel" / "loky"}`',
            FutureWarning
        )
        del kwargs["map_func"]

    if "map_kwargs" in kwargs and solver in ["mc", "stoc"]:
        warnings.warn(
            '"map_kwargs" are now included in options:\n'
            'Use `options={"num_cpus": N}`',
            FutureWarning
        )
        del kwargs["map_kwargs"]

    if "nsubsteps" in kwargs and solver == "stoc":
        warnings.warn(
            '"nsubsteps" is now replaced by "dt" in options:\n'
            'Use `options={"dt": 0.001}`\n'
            'The given value of "nsubsteps" is ignored in this call.',
            FutureWarning
        )
        # Could be (tlist[1] - tlist[0]) / kwargs["nsubsteps"]
        del kwargs["nsubsteps"]

    if "tol" in kwargs and solver == "stoc":
        warnings.warn(
            'The "tol" parameter is now the "atol" options:\n'
            'Use `options={"atol": tol}`',
            FutureWarning
        )
        options["atol"] = kwargs.pop("tol")

    if "store_all_expect" in kwargs and solver == "stoc":
        warnings.warn(
            'The "store_all_expect" parameter is now the '
            '"keep_runs_results" options:\n'
            'Use `options={"keep_runs_results": False / True}`',
            FutureWarning
        )
        options["keep_runs_results"] = kwargs.pop("store_all_expect")

    if "store_measurement" in kwargs and solver == "stoc":
        warnings.warn(
            'The "store_measurement" parameter is now an options:\n'
            'Use `options={"store_measurement": False / True}`',
            FutureWarning
        )
        options["store_measurement"] = kwargs.pop("store_measurement")

    if ("dW_factors" in kwargs or "m_ops" in kwargs) and solver == "stoc":
        raise TypeError(
            '"m_ops" and "dW_factors" are now properties of '
            'the stochastic solver class, use:\n'
            '>>> solver = SMESolver(H, c_ops)\n'
            '>>> solver.m_ops = m_ops\n'
            '>>> solver.dW_factors = dW_factors\n'
        )

    if kwargs:
        raise TypeError(f"unexpected keyword argument {kwargs.keys()}")
    return options


def _kwargs_migration(position, keyword, name):
    if position is not None:
        warnings.warn(
            f"{name} will be keyword only from qutip 5.3 for all solver",
            FutureWarning
        )
        return position
    return keyword
