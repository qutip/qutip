__all__ = ['Solver']

from .. import Qobj, QobjEvo, ket2dm
from .options import SolverOptions, SolverOdeOptions
from ..core import stack_columns, unstack_columns
from .result import Result
from .integrator import Integrator
from ..ui.progressbar import progess_bars
from ..core.data import to
from time import time


# SeSolver.avail_integrators should return SeSolver and Solver's integrators.
# Thus we want a property and classmethod
class classproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


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

    options : SolverOptions
        Options for the solver

    attributes
    ----------
    rhs : Qobj, QobjEvo
        Right hand side of the evolution::
            d state / dt = rhs @ state

    options : SolverOptions
        Options for the solver

    stats: dict
        Diverse statistics of the evolution.
    """
    name = "generic"

    # State, time and Integrator of the stepper functionnality
    _t = 0
    _state = None
    _integrator = False
    _avail_integrators = {}

    # Class of option used by the solver
    optionsclass = SolverOptions
    odeoptionsclass = SolverOdeOptions

    def __init__(self, rhs, *, options=None):
        if isinstance(rhs, (QobjEvo, Qobj)):
            self.rhs = QobjEvo(rhs)
        else:
            TypeError("The rhs must be a QobjEvo")
        self.options = options
        self.stats = {"preparation time": 0}
        self._state_metadata = {}

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

        if self.options.ode["state_data_type"]:
            state = state.to(self.options.ode["state_data_type"])

        self._state_metadata = {
            'dims': state.dims,
            'type': state.type,
            'isherm': state.isherm
        }
        if self.rhs.dims[1] == state.dims:
            return stack_columns(state.data)
        return state.data

    def _restore_state(self, state, *, copy=True):
        """
        Retore the Qobj state from the it's data.
        """
        if self._state_metadata['dims'] == self.rhs.dims[1]:
            return Qobj(unstack_columns(state),
                        **self._state_metadata, copy=False)
        else:
            return Qobj(state, **self._state_metadata, copy=copy)

    def run(self, state0, tlist, *, args=None, e_ops=None, options=None):
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

        options : SolverOptions {None}
            Options for the solver

        Return
        ------
        results : :class:`qutip.solver.Result`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        _data0 = self._prepare_state(state0)
        _integrator = self._get_integrator()
        if options is not None:
            self.options = options
        if args:
            _integrator.arguments(args)
        _time_start = time()
        _integrator.set_state(tlist[0], _data0)
        self.stats["preparation time"] += time() - _time_start
        results = Result(e_ops, self.options.results,
                         self.rhs.issuper, _data0.shape[1]!=1)
        results.add(tlist[0], state0)

        progress_bar = progess_bars[self.options['progress_bar']]()
        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in _integrator.run(tlist):
            progress_bar.update()
            results.add(t, self._restore_state(state, copy=False))
        progress_bar.finished()

        self.stats['run time'] = progress_bar.total_time()
        # TODO: It would be nice if integrator could give evolution statistics
        # self.stats.update(_integrator.stats)
        self.stats["method"] = _integrator.name
        results.stats = self.stats.copy()
        results.solver = self.name
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
        self._t = t0
        self._state = self._prepare_state(state0)
        self._integrator = self._get_integrator()
        self._integrator.set_state(self._t, self._state)
        self.stats["preparation time"] += time() - _time_start

    def step(self, t, *, args=None, options=None, copy=True):
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

        options : SolverOptions, optional {None}
            Update the ``options`` of the system.
            The change is effective from the beginning of the interval.
            Changing ``options`` can slow the evolution.

        copy : bool, optional {True}
            Whether to return a copy of the data or the data in the ODE solver.
        """
        if not self._integrator:
            raise RuntimeError("The `start` method must called first")
        if options is not None:
            self.options = options
            self._integrator = self._get_integrator()
            self._integrator.set_state(self._t, self._state)
        if args:
            self._integrator.arguments(args)
            self._integrator.reset()
        _time_start = time()
        self._t, self._state = self._integrator.integrate(t, copy=False)
        self.stats["run time"] += time() - _time_start
        return self._restore_state(self._state, copy=copy)

    def _get_integrator(self):
        """ Return the initialted integrator. """
        if self.options.ode["operator_data_type"]:
            self.rhs = self.rhs.to(
                self.options.ode["operator_data_type"]
            )

        method = self.options.ode["method"]
        if method in self.avail_integrators:
            integrator = self.avail_integrators[method]
        elif issubclass(method, Integrator):
            integrator = method
        else:
            raise ValueError("Integrator method not supported.")
        return integrator(self.rhs, self.options.ode)

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new):
        if new is None:
            new = self.optionsclass()
        elif isinstance(new, dict):
            new = self.optionsclass(**new)
        elif not isinstance(new, self.optionsclass):
            raise TypeError("options must be an instance of" +
                            str(self.optionsclass))
        self._options = new

    @classproperty
    @classmethod
    def avail_integrators(cls):
        if cls is Solver:
            return cls._avail_integrators.copy()
        return {**cls._avail_integrators,
                **Solver._avail_integrators}

    @classmethod
    def add_integrator(cls, integrator, keys):
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
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            cls._avail_integrators[key] = integrator
        if integrator.integrator_options:
            for opt in integrator.integrator_options:
                cls.odeoptionsclass.extra_options.add(opt)
