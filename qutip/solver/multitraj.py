from .. import Qobj, QobjEvo
from .options import SolverOptions, SolverOdeOptions
from .result import Result, MultiTrajResult
from .parallel import get_map
from time import time
from .solver_base import Solver
import numpy as np
from copy import copy


class MultiTrajSolver:
    """
    Basic class for multi-trajectories evolutions.

    As :class:`Solver` it can ``run`` or ``step`` evolution.
    It manages the random seed for each trajectory.

    The actual evolution is done by a single trajectory solver::
        ``_traj_solver_class``

    Parameters
    ----------
    rhs : Qobj, QobjEvo
        Right hand side of the evolution::
            d state / dt = rhs @ state

    options : SolverOptions
        Options for the solver.
    """
    _traj_solver_class = None
    name = "generic multi trajectory"
    optionsclass = SolverOptions
    odeoptionsclass = SolverOdeOptions

    def __init__(self, rhs, c_ops=(), *, options=None):
        self.rhs = rhs
        self._c_ops = c_ops
        self.stats = {}
        self.options = options
        self.seed_sequence = np.random.SeedSequence()
        self.traj_solver = False
        self.result = None

    def _read_seed(self, seed, ntraj):
        """
        Read user provided seed(s) and produce one for each trajectory.
        Let numpy raise error for inputs that cannot be seeds.
        """
        if seed is None:
            seeds = self.seed_sequence.spawn(ntraj)
        elif isinstance(seed, np.random.SeedSequence):
            seeds = copy(seed).spawn(ntraj)
        elif not isinstance(seed, list):
            seeds = np.random.SeedSequence(seed).spawn(ntraj)
        elif isinstance(seed, list) and len(seed) >= ntraj:
            seeds = [
                seed_  if isinstance(seed_, np.random.SeedSequence)
                else np.random.SeedSequence(seed_)
                for seed_ in seed[:ntraj]
            ]
        else:
            raise ValueError("A seed list must be longer than ntraj")
        return seeds

    def start(self, state0, t0, ntraj=1, seed=None, *, safe_ODE=None):
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        ntraj : int {1}
            Number of trajectories to compute with each step. The
            :method:`step` will return a list of ``ntraj`` state when greater
            than 1.

        seed : int, SeedSequence, list, {None}
            Seed for the random number generator. It can be a single seed used to
            spawn seeds for each trajectory or a list of seed, one for each
            trajectory. 

        safe_ODE : bool {None}
            Whether to safe the states in the ODE solver or in the solver.
            Many ODE solver are not re-entrant, thus must be used with
            `safe_ODE` if multiple trajectory are to be ran in parallel.
        """
        seeds = self._read_seed(seed, ntraj)
        self.traj_solvers = []

        if safe_ODE is None:
            safe_ODE = ntraj != 1

        for seed in seeds:
            traj_solver = self._traj_solver_class(self, options=self.options)
            traj_solver.start(state0, t0, seed=seed, safe_ODE=safe_ODE)
            self.traj_solvers.append(traj_solver)

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
        if not self.traj_solvers:
            raise RuntimeError("The `start` method must called first.")
        # TODO: could be done with parallel_map, but it's probably not worth it
        out = [traj_solver.step(t, args=args, options=options, copy=copy)
               for traj_solver in self.traj_solvers]
        return out if len(out) > 1 else out[0]

    def get_single_trajectory_solver(self):
        """
        Get a :cls:`Solver` for a single trajectory.
        """
        return self._traj_solver_class(self, options=self.options)

    def run(self, state0, tlist, ntraj=1, *, args=None, options=None,
            e_ops=None, timeout=0, target_tol=None, seed=None):
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
            evolution. Time in the list must be in increasing order, but does not
            need to be uniformly distributed.

        ntraj : int
            Number of trajectories to add.

        args : dict, optional {None}
            Change the ``args`` of the rhs for the evolution.

        options : SolverOptions, optional {None}
            Update the ``options`` of the system.
            The change is effective from the beginning of the interval.
            Changing ``options`` can slow the evolution.

        e_ops : list
            list of Qobj or QobjEvo to compute the expectation values.
            Alternatively, function[s] with the signature f(t, state) -> expect
            can be used.

        timeout : float, optional
            Maximum time in seconds for the trajectories to run. Once this time
            is reached, the simulation will end even if the number 
            of trajectories is less than ``ntraj``. The map function, set in
            options, can interupt the running trajectory or wait for it to
            finish. Set to ``0`` to disable.

        target_tol : {float, tuple, list}, optional
            If a float, it is read as absolute tolerance.
            If a pair of float: absolute and relative tolerance in that order.
            Lastly, target_tol can be a list of pairs of (atol, rtol) for each
            e_ops.

        seed : {int, list(int)} optional
            Seed or list of seeds for each trajectories.

        Return
        ------
        results : :class:`qutip.solver.MultiTrajResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        if options is not None:
            self.options = options
        if args is not None:
            self._argument(args)

        self.result = MultiTrajResult(ntraj, e_ops or [], self._c_ops,
                                      options=self.options.results,
                                      target_tol=target_tol)

        self._run_args = state0, tlist, e_ops
        self.result.stats['run time'] = 0
        self._traj_solver = self._traj_solver_class(self, options=self.options)
        self._traj_solver._integrator = self._traj_solver._get_integrator()
        self.add_trajectories(ntraj, timeout=timeout, seed=seed)
        return self.result

    def add_trajectories(self, ntraj, *,
                         timeout=0, target_tol=None, seed=None):
        """
        Add ``ntraj`` trajectories. Some trajectories must already be computed
        with :method:`run`.

        Parameters
        ----------
        ntraj : int
            Number of trajectories to add.

        timeout : float, optional
            Maximum time in seconds for the trajectories to run. Once this time
            is reached, the simulation will end even if the number 
            of trajectories is less than ``ntraj``. The map function, set in
            options, can interupt the running trajectory or wait for it to
            finish. Set to ``0`` to disable

        target_tol : {float, tuple, list}, optional
            If a float, it is read as absolute tolerance.
            If a pair of float: absolute and relative tolerance in that order.
            Lastly, target_tol can be a list of pairs of (atol, rtol) for each
            e_ops.

        seed : {int, list(int)} optional
            Seed or list of seeds for each trajectories.

        Return
        ------
        results : :class:`qutip.solver.MultiTrajResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        if self.result is None:
            raise RuntimeError("No previous computation, use `run` first.")
        start_time = time()
        seeds = self._read_seed(seed, ntraj)
        map_func = get_map[self.options.mcsolve['map']]
        map_kw = {
            'timeout': timeout or self.options.mcsolve['timeout'],
            'job_timeout': self.options.mcsolve['job_timeout'],
            'num_cpus': self.options.mcsolve['num_cpus'],
        }
        self.result._target_ntraj = ntraj + self.result.num_traj

        if target_tol:
            self.result._set_expect_tol(target_tol)
        map_func(
            self._traj_solver._run, seeds, self._run_args,
            reduce_func=self.result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        self.result.stats['run time'] += time() - start_time
        self.result.stats.update(self.stats)
        return self.result

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

    def _argument(self, args):
        """Update the args, for the `rhs` and `c_ops` and other operators."""
        self.rhs.arguments(args)


class _TrajectorySolver(Solver):
    """
    :cls:`Solver` for trajectory of a multi-trajectory evolution (``mcsolve``,
    ``photocurrent``, etc.).

    :method:`start`, and :method:`run` take and extra ``seed`` keyword
    argument to initiate the random number generator.

    Evolution between times is done by the :method:`_step` which needs to be
    implemented for the solver problem (include collapse in mcsolve, etc.).
    An :class:`Integrator` is initiated using `rhs` before calling `_step`.

    A new :method:`_argument` is used to update the args in `rhs` and other
    :class:`QobjEvo` and need to be overloaded if other operators are used.
    """
    name = "Generic trajectory solver"

    def __init__(self, rhs, *, options=None):
        if isinstance(rhs, (QobjEvo, Qobj)):
            self.rhs = QobjEvo(rhs)
        else:
            TypeError("The rhs must be a QobjEvo")
        self.options = options
        self.stats = {"preparation time": 0}
        self._state_metadata = {}
        if self.options.mcsolve['bitgenerator']:
            if hasattr(np.random, self.options.mcsolve['bitgenerator']):
                self.bit_gen = getattr(np.random,
                                       self.options.mcsolve['bitgenerator'])
            else:
                raise ValueError("BitGenerator is not know to numpy.random")
        else:
            # We use default_rng so we don't fix a default bit_generator.
            self.bit_gen = np.random.default_rng().bit_generator.__class__

    def start(self, state0, t0, seed=None, *, safe_ODE=False):
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        seed : int, SeedSequence
            Seed for the random number generator.

        safe_ODE : bool {False}
            Whether to save the state locally. Set to ``True`` if using
            multiple solvers for step evolution at once.
        """
        _time_start = time()
        self._set_generator(seed)
        self.safe_ODE = safe_ODE
        self._state = self._prepare_state(state0)
        self._t = t0
        self._integrator = self._get_integrator()
        self._integrator.set_state(self._t, self._state)
        self.stats["preparation time"] += time() - _time_start

    def step(self, t, *, args=None, options=None, copy=True):
        if not self._integrator:
            raise RuntimeError("The `start` method must called first")
        if options is not None:
            self.options = options
            self._integrator = self._get_integrator()
            self._integrator.set_state(self._t, self._state)
        if args:
            self._argument(args)
            self._integrator.reset()
        if self.safe_ODE:
            self._integrator.set_state(self._t, self._state)
        self._t, self._state = self._step(t, copy=self.safe_ODE)
        return self._restore_state(self._state, copy=copy)

    def run(self, state0, tlist, *,
            args=None, e_ops=None, options=None, seed=None):
        """
        Do the evolution of the Quantum system.

        For a ``state0`` at time ``tlist[0]`` do the evolution as directed
        by ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :cls:`Result`. The evolution method and
        stored results are determined by ``options``.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of
            the evolution. Each times of the list must be increasing, but
            does not need to be uniformy distributed.

        args : dict, optional {None}
            Change the ``args`` of the rhs for the evolution.

        e_ops : list {None}
            List of Qobj, QobjEvo or callable to compute the expectation
            values. Function[s] must have the signature
                f(t : float, state : Qobj) -> expect.

        options : SolverOptions {None}
            Options for the solver

        seed : int, SeedSequence
            Seed for the random number generator.

        Return
        ------
        results : :class:`qutip.solver.Result`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        if options is not None:
            self.options = options
        if args:
            self._argument(args)

        self._integrator = self._get_integrator()
        return self._run(seed, state0, tlist, e_ops)

    def _run(self, seed, state0, tlist, e_ops):
        """ Core loop of run for the parallel map"""
        _time_start = time()
        self._set_generator(seed)

        _state = self._prepare_state(state0)
        self._integrator.set_state(tlist[0], _state)

        result = Result(e_ops, self.options.results,
                        self.rhs.issuper, _state.shape[1]!=1)
        result.add(tlist[0], state0)
        for t in tlist[1:]:
            t, state = self._step(t)
            state_qobj = self._restore_state(state, copy=False)
            result.add(t, state_qobj, copy=False)

        result.seed = seed
        result.stats['run time'] = time() - _time_start
        result.stats.update(self.stats)
        result.solver = self.name
        return result

    def _set_generator(self, seed):
        """
        Read the seed and create the random number generator.
        If the ``seed`` has a ``random`` method, it will be used as the
        generator.
        """
        if hasattr(seed, 'random'):
            # We check for the method, not the type to accept pseudo non-random
            # generator for debug purpose.
            self.generator = seed
            return
        if not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(seed)
        self.generator = np.random.Generator(self.bit_gen(seed))

    def _step(self, t, copy=True):
        """Evolve to t, including jumps."""
        raise NotImplementedError

    def _argument(self, args):
        """Update the args, for the `rhs` and `c_ops` and other operators."""
        if self._integrator is not None:
            self._integrator.arguments(args)
        self.rhs.arguments(args)
