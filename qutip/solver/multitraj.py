from .. import Qobj, QobjEvo
from .options import SolverOptions, SolverOdeOptions
from .result import Result, MultiTrajResult, MultiTrajResultAveraged
from ..ui.progressbar import progess_bars
from ..core.data import to
from .parallel import get_map
from time import time
from .solver_base import Solver, classproperty
import numpy as np


class MultiTrajSolver:
    """
    ... TODO
    """
    _traj_solver_class = None
    name = "generic multi trajectory"
    optionsclass = None
    odeoptionsclass = SolverOdeOptions

    def __init__(self):
        self.seed_sequence = np.random.SeedSequence()
        self.traj_solver = False
        self.result = None
        raise NotImplementedError

    def _read_seed(self, seed, ntraj):
        """
        Read user provided seed(s) and produce one for each trajectories.
        Let numpy raise error for input that cannot be seeds.
        """
        if seed is None:
            seeds = self.seed_sequence.spawn(ntraj)
        elif isinstance(seed, np.random.SeedSequence):
            seeds = seed.spawn(ntraj)
        elif not isinstance(seed, list):
            seeds = np.random.SeedSequence(seed).spawn(ntraj)
        elif isinstance(seed, list) and len(seed) >= ntraj:
            seeds = [np.random.SeedSequence(seed_) for seed_ in seed[:ntraj]]
        else:
            raise ValueError("A seed list must be longer than ntraj")
        return seeds

    def start(self, state0, t0, *, ntraj=1, seed=None):
        seeds = self._read_seed(seed, ntraj)
        self.traj_solvers = []

        for seed in seeds:
            traj_solver = self._traj_solver_class(self, e_ops=self.e_ops,
                                                  options=self.options)
            traj_solver.start(state0, t0, seed)
            self.traj_solvers.append(traj_solver)

    def step(self, t, args=None):
        if not self.traj_solvers:
            raise RuntimeError("The `start` method must called first.")
        multi = len(self.traj_solvers) == 1
        out = [traj_solver.step(t, args, safe=multi)
               for traj_solver in self.traj_solvers]
        return out if len(out) > 1 else out[0]

    def get_single_trajectory_solver(self):
        """
        Get a :cls:`Solver` for a single trajectory.
        """
        return self._traj_solver_class(self, e_ops=self.e_ops,
                                       options=self.options)

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
            evolution. Each times of the list must be increasing, but does not
            need to be uniformy distributed.

        args : dict, optional {None}
            Change the ``args`` of the rhs for the evolution.

        e_ops : list
            list of Qobj or QobjEvo to compute the expectation values.
            Alternatively, function[s] with the signature f(t, state) -> expect
            can be used.

        ntraj : int
            Number of trajectories to add.

        timeout : float, optional
            Maximum time in second for the trajectories to run. Once this time
            is reached, the simulation will end even if ``ntraj`` new
            trajectories have not been computed. The map function, set in
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

        self.result = (MultiTrajResult(e_ops or self.e_ops)
                       if self.options.mcsolve['keep_runs_results']
                       else MultiTrajResultAveraged(e_ops or self.e_ops))
        self._run_args = state0, tlist
        self._run_kwargs = {'args': args, 'e_ops': e_ops or self.e_ops}
        self.result.stats['run time'] = 0
        self.add_trajectories(ntraj, timeout=timeout,
                              target_tol=target_tol, seed=seed)
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
            Maximum time in second for the trajectories to run. Once this time
            is reached, the simulation will end even if ``ntraj`` new
            trajectories have not been computed. The map function, set in
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
        if self.result is None:
            raise RuntimeError("No previous computation, use `run` first.")
        start_time = time()
        seeds = self._read_seed(seed, ntraj)
        map_func = get_map[self.options.mcsolve['map']]
        map_kw = self.options.mcsolve['map_options']
        if timeout:
            map_kw['job_timeout'] = timeout
        if target_tol:
            self.result._set_check_expect_tol(target_tol)
        map_func(
            self._traj_solver_class(self)._run, seeds,
            self._run_args, self._run_kwargs,
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


class _TrajectorySolver(Solver):
    """
    :cls:`Solver` for trajectory of a multi-trajectory evolution (``mcsolve``,
    ``photocurrent``, etc.).

    :method:`start`, and :method:`run` take and extra ``seed`` keyword
    argument to initiate the random number generator.

    Evolution between times is done by the :method:`_step` which need to be
    implemented for the solver problem (indclude collapse in mcsolve, etc.).
    An :class:`Integrator` is initiated using `rhs` before calling `_step`.

    A new :method:`_argument` is used to update the args in `rhs` and other
    :class:`QobjEvo` and need to be overloaded if other operators are used.
    """
    name = "Generic trajectory solver"
    def __init__(self, rhs, *, e_ops=None, options=None):
        if isinstance(rhs, (QobjEvo, Qobj)):
            self.rhs = QobjEvo(rhs)
        else:
            TypeError("The rhs must be a QobjEvo")
        self.e_ops = e_ops
        self.options = options
        self.stats = {"preparation time": 0}
        self._state_metadata = {}

        if self.options.mcsolve['BitGenerator']:
            if hasattr(np_rng, self.options.mcsolve['BitGenerator']):
                self.bit_gen = getattr(np.random,
                                       self.options.mcsolve['BitGenerator'])
            else:
                raise ValueError("BitGenerator is not know to numpy.random")
        else:
            self.bit_gen = np.random.PCG64DXSM

    def start(self, state0, t0, seed=None):
        _time_start = time()
        self._set_generator(seed)
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
        self._t, self._state = self._step(t, copy=copy)
        return self._restore_state(self._state, copy=copy)

    def run(self, state0, tlist, *,
            args=None, e_ops=None, options=None, seed=None):
        _time_start = time()
        self._set_generator(seed)
        if options is not None:
            self.options = options
        if args:
            self._argument(args)

        _state = self._prepare_state(state0)
        self._integrator = self._get_integrator()
        self._integrator.set_state(tlist[0], _state)

        result = Result(e_ops or self.e_ops, self.options.results,
                        self.rhs.issuper, _state.shape[1]!=1)
        result.add(tlist[0], state0)
        for t in tlist[1:]:
            t, state = self._step(t)
            state_qobj = self._restore_state(state, copy=False)
            result.add(t, state_qobj)

        result.seed = seed
        result.stats['run time'] = time() - _time_start
        result.stats.update(self.stats)
        result.solver = self.name
        return result

    def _run(self, seed, state0, tlist, *, args=None, e_ops=None):
        return self.run(state0, tlist, args=args, e_ops=e_ops, seed=seed)

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
        if isinstance(seed, int):
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
