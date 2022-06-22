__all__ = ['Solver']

from .. import Qobj, QobjEvo, ket2dm
from .options import known_solver, _AttachedSolverOptions, SolverOptions
from ..core import stack_columns, unstack_columns
from .result import Result
from .integrator import Integrator
from ..ui.progressbar import progess_bars
from time import time


class Solver:
    """
    Runner for an evolution.
    Can run the evolution at once using :method:`run` or step by step using
    :method:`start` and :method:`step`.

    Parameters
    ----------
    rhs : Qobj, QobjEvo
        Right hand side of the evolution::
            d state / dt = rhs @ state

    options : dict
        Options for the solver
    """
    name = "generic"

    # State, time and Integrator of the stepper functionnality
    _integrator = None
    _avail_integrators = {}

    # Class of option used by the solver
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": "ket",
        'method': 'adams',
    }
    resultclass = Result

    def __init__(self, rhs, *, options=None):
        if isinstance(rhs, (QobjEvo, Qobj)):
            self.rhs = QobjEvo(rhs)
        else:
            TypeError("The rhs must be a QobjEvo")
        self._options = _AttachedSolverOptions(self)
        if options is not None:
            self.options = options
        _time_start = time()
        self._integrator = self._get_integrator()
        self._init_integrator_time = time() - _time_start
        self._state_metadata = {}
        self.stats = self._initialize_stats()

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
        determining the dims of the output for :method:`_restore_state`.

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
            'dims': state.dims,
            'type': state.type,
            'isherm': state.isherm and not (self.rhs.dims == state.dims)
        }
        if self.rhs.dims[1] == state.dims:
            return stack_columns(state.data)
        return state.data

    def _restore_state(self, data, *, copy=True):
        """
        Retore the Qobj state from its data.
        """
        if self._state_metadata['dims'] == self.rhs.dims[1]:
            state = Qobj(unstack_columns(data),
                         **self._state_metadata, copy=False)
        else:
            state = Qobj(data, **self._state_metadata, copy=copy)

        if data.shape[1] == 1 and self._options['normalize_output']:
            state = state * (1 / state.norm())

        return state

    def run(self, state0, tlist, *, args=None, e_ops=None):
        """
        Do the evolution of the Quantum system.

        For a ``state0`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :cls:`Result`. The evolution method and stored
        results are determined by ``options``.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Each times of the list must be increasing, but does not
            need to be uniformy distributed.

        args : dict, optional {None}
            Change the ``args`` of the rhs for the evolution.

        e_ops : list {None}
            List of Qobj, QobjEvo or callable to compute the expectation
            values. Function[s] must have the signature
            f(t : float, state : Qobj) -> expect.

        Return
        ------
        results : :class:`qutip.solver.Result`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        _time_start = time()
        _data0 = self._prepare_state(state0)
        self._integrator.set_state(tlist[0], _data0)
        self._argument(args)
        stats = self._initialize_stats()
        results = self.resultclass(
            e_ops, self.options,
            solver=self.name, stats=stats,
        )
        results.add(tlist[0], self._restore_state(_data0, copy=False))
        stats['preparation time'] += time() - _time_start

        progress_bar = progess_bars[self.options['progress_bar']]()
        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in self._integrator.run(tlist):
            progress_bar.update()
            results.add(t, self._restore_state(state, copy=False))
        progress_bar.finished()

        stats['run time'] = progress_bar.total_time()
        # TODO: It would be nice if integrator could give evolution statistics
        # stats.update(_integrator.stats)
        return results

    def start(self, state0, t0):
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.
        """
        _time_start = time()
        self._integrator.set_state(t0, self._prepare_state(state0))
        self.stats["preparation time"] += time() - _time_start

    def step(self, t, *, args=None, copy=True):
        """
        Evolve the state to ``t`` and return the state as a :class:`Qobj`.

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

        .. note::
            The state must be initialized first by calling ``start`` or
            ``run``. If ``run`` is called, ``step`` will continue from the last
            time and state obtained.
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
        method = self._options["method"]
        if method in self.avail_integrators():
            integrator = self.avail_integrators()[method]
        elif issubclass(method, Integrator):
            integrator = method
        else:
            raise ValueError("Integrator method not supported.")
        return integrator(self.rhs, self.options)

    @property
    def options(self):
        """
        store_final_state: bool
            Whether or not to store the final state of the evolution in the
            result class.

        store_states": bool, None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        normalize_output: bool
            Normalize output state to hide ODE numerical errors.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.

        progress_kwargs: dict
            kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.

        method: str
            Which ordinary differential equation integration method to use.
        """
        return self._options

    def _parse_options(self, **new_options):
        """
        Do a first read through of the options:
        - Raise an error for unsupported error.
        - Replace ``None`` with the default value.
        - Remove items that are unchanged.
        """
        method = new_options.get('method', self.options["method"])
        default_options = SolverOptions(self.name, method=method)
        extra_keys = new_options.keys() - default_options.supported_keys
        if extra_keys:
            raise KeyError(f"Options {extra_keys} are not supported.")
        # First pass: Have ``None`` refert to the default.
        new_options = {
            key: (val if val is not None else default_options[key])
            for key, val in new_options.items()
        }
        # Second pass: Remove options that remain unchanged.
        new_options = {
            key: val
            for key, val in new_options.items()
            if key in self._options and val != self._options[key]
        }
        return new_options

    @options.setter
    def options(self, new_options):
        new_options = self._parse_options(**new_options)
        if not new_options:
            return  # Nothing to do

        # Create options without the integrator items that are not
        # not supported by the new integrator if changed.
        kept_options = self._options.copy()
        if 'method' in new_options:
            kept_options['method'] = new_options['method']

        self._options = _AttachedSolverOptions(
            self,
            **{**kept_options, **new_options}
        )

        self._apply_options(new_options.keys())

    def _apply_options(self, keys):
        """
        Method called when options are changed, either through
        ``solver.options[key] = value`` or ``solver.options = options``.
        Allow to update the solver with the new options
        """
        if self._integrator is None or not keys:
            pass
        elif 'method' in keys:
            state = self._integrator.get_state()
            self._integrator = self._get_integrator()
            self._integrator.set_state(*state)
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
            **cls._avail_integrators
        }

    @classmethod
    def integrator_options(cls):
        integrators = cls.avail_integrators()
        options = {key: inte.integrator_options
                   for key, inte in integrators.items()}
        return options

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


known_solver['solver'] = Solver
known_solver['Solver'] = Solver
known_solver['generic'] = Solver
