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
    Basic class for multi-trajectory evolutions.

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
    resultclass = MultiTrajResult

    def __init__(self, rhs, *, options=None):
        self.rhs = rhs
        self.stats = {}
        self.options = options
        self.seed_sequence = np.random.SeedSequence()

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
                seed_ if isinstance(seed_, np.random.SeedSequence)
                else np.random.SeedSequence(seed_)
                for seed_ in seed[:ntraj]
            ]
        else:
            raise ValueError("A seed list must be longer than ntraj")
        return seeds

    def start(self, state, t0, ntraj=1, seed=None):
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        ntraj : int {1}
            Number of trajectories to compute with each step. The
            :method:`step` will return a list of ``ntraj`` state when greater
            than 1.

        seed : int, SeedSequence, list, {None}
            Seed for the random number generator. It can be a single seed used
            to spawn seeds for each trajectory or a list of seed, one for each
            trajectory.

        .. note ::
            Using the solver with ``start``, ``step`` is independent to use
            with ``run``. Calling ``run`` between ``step`` will not affect
            ``step``'s result.
        """
        seeds = self._read_seed(seed, ntraj)
        self.traj_solvers = []

        for seed in seeds:
            traj_solver = self._traj_solver_class(*self.traj_args,
                                                  options=self.options)
            traj_solver.start(state, t0, seed=seed)
            self.traj_solvers.append(traj_solver)

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

        .. note ::
            Using the solver with ``start``, ``step`` is independent to use
            with ``run``. Calling ``run`` between ``step`` will not affect
            ``step``'s result.
        """
        if not self.traj_solvers:
            raise RuntimeError("The `start` method must called first.")
        # TODO: could be done with parallel_map, but it's probably not worth it
        out = [traj_solver.step(t, args=args, copy=copy)
               for traj_solver in self.traj_solvers]
        return out if len(out) > 1 else out[0]

    def get_single_trajectory_solver(self):
        """
        Get a :cls:`Solver` for a single trajectory.
        """
        return self._traj_solver_class(*self.traj_args, options=self.options)

    def run(self, state, tlist, ntraj=1, *, args=None,
            e_ops=(), timeout=1e8, target_tol=None, seed=None):
        """
        Do the evolution of the Quantum system.

        For a ``state`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :cls:`Result`. The evolution method and stored
        results are determined by ``options``.

        Parameters
        ----------
        state : :class:`Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Time in the list must be in increasing order, but does
            not need to be uniformly distributed.

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

        timeout : float, optional [1e8]
            Maximum time in seconds for the trajectories to run. Once this time
            is reached, the simulation will end even if the number
            of trajectories is less than ``ntraj``. The map function, set in
            options, can interupt the running trajectory or wait for it to
            finish. Set to an arbitrary high number to disable.

        target_tol : {float, tuple, list}, optional [None]
            If a float, it is read as absolute tolerance.
            If a pair of float: absolute and relative tolerance in that order.
            Lastly, target_tol can be a list of pairs of (atol, rtol) for each
            e_ops. Set to ``None`` to disable.

        seed : {int, list(int)} optional
            Seed or list of seeds for each trajectories.

        Return
        ------
        results : :class:`qutip.solver.MultiTrajResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.

        .. note:
            The simulation will end when the first end condition is reached
            between ``ntraj``, ``timeout`` and ``target_tol``.
        """
        self._argument(args)

        result = self.resultclass(
            ntraj, state, tlist, e_ops, id(self), options=self.options.results,
        )

        result.stats['run time'] = 0
        self._traj_solver = self._traj_solver_class(
            *self.traj_args, options=self.options,
        )
        self.add_trajectories(
            result, ntraj, target_tol=target_tol,
            timeout=timeout, seed=seed
        )
        return result

    def add_trajectories(self, result, ntraj, *,
                         timeout=0, target_tol=None, seed=None):
        """
        Add ``ntraj`` trajectories. Some trajectories must already be computed
        with :method:`run`.

        Parameters
        ----------
        result : MultiTrajResult
            Result object obtained by a previous call of run of this object
            into which add more trajectories. The result will be changed
            inplace.

        ntraj : int
            Number of trajectories to add.

        timeout : float, optional [1e8]
            Maximum time in seconds for the trajectories to run. Once this time
            is reached, the simulation will end even if the number
            of trajectories is less than ``ntraj``. The map function, set in
            options, can interupt the running trajectory or wait for it to
            finish. Set to an arbitrary high number to disable.

        target_tol : {float, tuple, list}, optional [None]
            If a float, it is read as absolute tolerance.
            If a pair of float: absolute and relative tolerance in that order.
            Lastly, target_tol can be a list of pairs of (atol, rtol) for each
            e_ops. Set to ``None`` to disable.

        seed : {int, list(int)} optional
            Seed or list of seeds for each trajectories.

        Return
        ------
        results : :class:`qutip.solver.MultiTrajResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.

        .. note:
            The simulation will end when the first end condition is reached
            between ``ntraj``, ``timeout`` and ``target_tol``.
        """
        start_time = time()
        if result.solver_id != id(self):
            raise ValueError("Result class was not created by this solver")
        seeds = self._read_seed(seed, ntraj)
        map_func = get_map[self.options.mcsolve['map']]
        map_kw = {
            'timeout': timeout or self.options.mcsolve['timeout'],
            'job_timeout': self.options.mcsolve['job_timeout'],
            'num_cpus': self.options.mcsolve['num_cpus'],
        }
        result._target_ntraj = ntraj + result.num_traj

        if target_tol:
            result.set_expect_tol(target_tol)
        map_func(
            self._traj_solver._run_map, seeds,
            (result.initial_state, result.tlist, result.e_ops),
            reduce_func=result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        result.stats['run time'] += time() - start_time
        result.stats.update(self.stats)
        return result

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new):
        if new is None:
            new = self.optionsclass()
        elif isinstance(new, dict):
            new = self.optionsclass(**new)
        elif isinstance(new, self.optionsclass):
            new = copy(new)
        else:
            raise TypeError("options must be an instance of" +
                            str(self.optionsclass))
        self._options = new

    def _argument(self, args):
        """Update the args, for the `rhs` and `c_ops` and other operators."""
        if args:
            self.rhs.arguments(args)

    @property
    def traj_args(self):
        return (self.rhs,)


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
        super().__init__(rhs, options=options)
        if self.options.mcsolve['bitgenerator']:
            if hasattr(np.random, self.options.mcsolve['bitgenerator']):
                self.bit_gen = getattr(np.random,
                                       self.options.mcsolve['bitgenerator'])
            else:
                raise ValueError("BitGenerator is not know to numpy.random")
        else:
            # We use default_rng so we don't fix a default bit_generator.
            self.bit_gen = np.random.default_rng().bit_generator.__class__

    def start(self, state, t0, seed=None):
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        seed : int, SeedSequence
            Seed for the random number generator.
        """
        _time_start = time()
        self.generator = self.get_generator(seed)
        self._state = self._prepare_state(state)
        self._t = t0
        self._integrator.set_state(self._t, self._state)
        self.stats["preparation time"] += time() - _time_start

    def step(self, t, *, args=None, copy=True):
        if not self._integrator:
            raise RuntimeError("The `start` method must called first")
        self._argument(args)
        self._t, self._state = self._step(t)
        return self._restore_state(self._state, copy=copy)

    def run(self, state, tlist, *, args=None, e_ops=(), seed=None):
        """
        Do the evolution of the Quantum system.

        For a ``state`` at time ``tlist[0]`` do the evolution as directed
        by ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :cls:`Result`. The evolution method and
        stored results are determined by ``options``.

        Parameters
        ----------
        state : :class:`Qobj`
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
        self._argument(args)
        self.start(state, tlist[0], seed)
        _time_start = time()
        result = Result(e_ops, self.options.results,
                        self.rhs.issuper, state.shape[1] != 1)
        result.add(tlist[0], state)
        for t in tlist[1:]:
            t, state = self._step(t)
            state_qobj = self._restore_state(state, copy=False)
            result.add(t, state_qobj, copy=False)

        result.seed = seed
        result.stats['run time'] = time() - _time_start
        result.stats.update(self.stats)
        result.solver = self.name
        return result

    def _run_map(self, seed, state, tlist, e_ops):
        """
        Our parallel map functions pass the varying variable (seed) as the
        first parameter.
        """
        return self.run(state, tlist, e_ops=e_ops, seed=seed)

    def get_generator(self, seed):
        """
        Read the seed and create the random number generator.
        If the ``seed`` has a ``random`` method, it will be used as the
        generator.
        """
        if hasattr(seed, 'random'):
            # We check for the method, not the type to accept pseudo non-random
            # generator for debug purpose.
            return seed
        if not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(seed)
        return np.random.Generator(self.bit_gen(seed))

    def _step(self, t, copy=True):
        """Evolve to t, including jumps."""
        raise NotImplementedError

    def _argument(self, args):
        """Update the args, for the `rhs` and `c_ops` and other operators."""
        if args:
            self._integrator.arguments(args)
            self.rhs.arguments(args)
