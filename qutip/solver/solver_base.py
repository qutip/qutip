__all__ = ['Solver']

from .. import Qobj, QobjEvo, ket2dm
from .options import Options, SolverOptions
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

    options : SolverOptions
        Options for the solver
    """
    name = "generic"

    # State, time and Integrator of the stepper functionnality
    _integrator = None
    _avail_integrators = {}

    # Class of option used by the solver
    optionsclass = SolverOptions
    resultclass = Result

    def __init__(self, rhs, *, options=None):
        if isinstance(rhs, (QobjEvo, Qobj)):
            self.rhs = QobjEvo(rhs)
        else:
            TypeError("The rhs must be a QobjEvo")
        self.options = options
        _time_start = time()
        self._integrator = self._get_integrator()
        self.stats = {"preparation time": time() - _time_start}
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

        if self.options["state_data_type"]:
            state = state.to(self.options["state_data_type"])

        self._state_metadata = {
            'dims': state.dims,
            'type': state.type,
            'isherm': state.isherm and not (self.rhs.dims == state.dims)
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

        options : None / dict / :class:`SolverOptions` / :class:`Options`
            Options for the solver

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
        self.stats["preparation time"] += time() - _time_start

        results = self.resultclass(e_ops, self.options,
                                   self.rhs.issuper, _data0.shape[1]!=1)
        results.add(tlist[0], state0)

        progress_bar = progess_bars[self.options['progress_bar']]()
        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in self._integrator.run(tlist):
            progress_bar.update()
            results.add(t, self._restore_state(state, copy=False))
        progress_bar.finished()

        self.stats['run time'] = progress_bar.total_time()
        # TODO: It would be nice if integrator could give evolution statistics
        # self.stats.update(_integrator.stats)
        self.stats["method"] = self._integrator.name
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

        options : Options, optional {None}
            Update the ``options`` of the system.
            The change is effective from the beginning of the interval.
            Changing ``options`` can slow the evolution.

        copy : bool, optional {True}
            Whether to return a copy of the data or the data in the ODE solver.

        .. note :
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
        if self.options["operator_data_type"]:
            self.rhs = self.rhs.to(
                self.options["operator_data_type"]
            )

        method = self.options["method"]
        if method in self.avail_integrators():
            integrator = self.avail_integrators()[method]
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
        self._options = self.optionsclass(new, _frozen=True)

    def _argument(self, args):
        """Update the args, for the `rhs` and other operators."""
        if args:
            self._integrator.arguments(args)
            self.rhs.arguments(args)

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
    def add_integrator(cls, integrator, optioncls, key):
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
        cls.optionsclass._ode_options[key] = optioncls
        Options._add_options(optioncls)
