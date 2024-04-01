from .result import Result, MultiTrajResult
from .parallel import _get_map
from time import time
from .solver_base import Solver
from ..core import QobjEvo
import numpy as np

__all__ = ["MultiTrajSolver"]


class _MultiTrajRHS:
    """
    Container for the operators of the solver.
    """
    def __init__(self, rhs):
        self.rhs = rhs

    def arguments(self, args):
        self.rhs.arguments(args)

    def _register_feedback(self, type, val):
        pass

    def __getattr__(self, attr):
        if attr == "rhs":
            raise AttributeError
        if hasattr(self.rhs, attr):
            return getattr(self.rhs, attr)
        raise AttributeError


class MultiTrajSolver(Solver):
    """
    Basic class for multi-trajectory evolutions.

    As :class:`.Solver` it can ``run`` or ``step`` evolution.
    It manages the random seed for each trajectory.

    The actual evolution is done by a single trajectory solver::
        ``_traj_solver_class``

    Parameters
    ----------
    rhs : Qobj, QobjEvo
        Right hand side of the evolution::
            d state / dt = rhs @ state

    options : dict
        Options for the solver.
    """
    name = "generic multi trajectory"
    _resultclass = MultiTrajResult
    _trajectory_resultclass = Result
    _avail_integrators = {}

    # Class of option used by the solver
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "normalize_output": False,
        "method": "",
        "map": "serial",
        "mpi_options": {},
        "num_cpus": None,
        "bitgenerator": None,
    }

    def __init__(self, rhs, *, options=None):
        if isinstance(rhs, QobjEvo):
            self.rhs = _MultiTrajRHS(rhs)
        elif isinstance(rhs, _MultiTrajRHS):
            self.rhs = rhs
        else:
            raise TypeError("The system should be a QobjEvo")
        self.options = options
        self.seed_sequence = np.random.SeedSequence()
        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()

    def start(self, state, t0, seed=None):
        """
        Set the initial state and time for a step evolution.

        Parameters
        ----------
        state : :obj:`.Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        seed : int, SeedSequence, list, optional
            Seed for the random number generator. It can be a single seed used
            to spawn seeds for each trajectory or a list of seed, one for each
            trajectory.

        Notes
        -----
        When using step evolution, only one trajectory can be computed at once.
        """
        seeds = self._read_seed(seed, 1)
        generator = self._get_generator(seeds[0])
        self._integrator.set_state(t0, self._prepare_state(state), generator)

    def step(self, t, *, args=None, copy=True):
        """
        Evolve the state to ``t`` and return the state as a :obj:`.Qobj`.

        Parameters
        ----------
        t : double
            Time to evolve to, must be higher than the last call.

        args : dict, optional
            Update the ``args`` of the system.
            The change is effective from the beginning of the interval.
            Changing ``args`` can slow the evolution.

        copy : bool, default: True
            Whether to return a copy of the data or the data in the ODE solver.
        """
        if not self._integrator._is_set:
            raise RuntimeError("The `start` method must called first.")
        self._argument(args)
        _, state = self._integrator.integrate(t, copy=False)
        return self._restore_state(state, copy=copy)

    def _initialize_run(self, state, ntraj=1, args=None, e_ops=(),
                        timeout=None, target_tol=None, seeds=None):
        start_time = time()
        self._argument(args)
        stats = self._initialize_stats()
        seeds = self._read_seed(seeds, ntraj)

        result = self._resultclass(
            e_ops, self.options, solver=self.name, stats=stats
        )
        result.add_end_condition(ntraj, target_tol)

        map_func, map_kw = _get_map(self.options)
        map_kw.update({
            'timeout': timeout,
            'num_cpus': self.options['num_cpus'],
        })
        state0 = self._prepare_state(state)
        stats['preparation time'] += time() - start_time
        return stats, seeds, result, map_func, map_kw, state0

    def run(self, state, tlist, ntraj=1, *,
            args=None, e_ops=(), timeout=None, target_tol=None, seeds=None):
        """
        Do the evolution of the Quantum system.

        For a ``state`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :class:`.Result`. The evolution method and
        stored results are determined by ``options``.

        Parameters
        ----------
        state : :obj:`.Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Time in the list must be in increasing order, but does
            not need to be uniformly distributed.

        ntraj : int
            Number of trajectories to add.

        args : dict, optional
            Change the ``args`` of the rhs for the evolution.

        e_ops : list
            list of Qobj or QobjEvo to compute the expectation values.
            Alternatively, function[s] with the signature f(t, state) -> expect
            can be used.

        timeout : float, optional
            Maximum time in seconds for the trajectories to run. Once this time
            is reached, the simulation will end even if the number
            of trajectories is less than ``ntraj``. The map function, set in
            options, can interupt the running trajectory or wait for it to
            finish. Set to an arbitrary high number to disable.

        target_tol : {float, tuple, list}, optional
            Target tolerance of the evolution. The evolution will compute
            trajectories until the error on the expectation values is lower
            than this tolerance. The maximum number of trajectories employed is
            given by ``ntraj``. The error is computed using jackknife
            resampling. ``target_tol`` can be an absolute tolerance or a pair
            of absolute and relative tolerance, in that order. Lastly, it can
            be a list of pairs of (atol, rtol) for each e_ops.

        seeds : {int, SeedSequence, list}, optional
            Seed or list of seeds for each trajectories.

        Returns
        -------
        results : :class:`.MultiTrajResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.

        .. note:
            The simulation will end when the first end condition is reached
            between ``ntraj``, ``timeout`` and ``target_tol``.
        """
        stats, seeds, result, map_func, map_kw, state0 = self._initialize_run(
            state,
            ntraj,
            args=args,
            e_ops=e_ops,
            timeout=timeout,
            target_tol=target_tol,
            seeds=seeds,
        )
        start_time = time()
        map_func(
            self._run_one_traj, seeds,
            (state0, tlist, e_ops),
            reduce_func=result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        result.stats['run time'] = time() - start_time
        return result

    def _initialize_run_one_traj(self, seed, state, tlist, e_ops,
                                 **integrator_kwargs):
        result = self._trajectory_resultclass(e_ops, self.options)
        generator = self._get_generator(seed)
        self._integrator.set_state(tlist[0], state, generator,
                                   **integrator_kwargs)
        result.add(tlist[0], self._restore_state(state, copy=False))
        return result

    def _run_one_traj(self, seed, state, tlist, e_ops, **integrator_kwargs):
        """
        Run one trajectory and return the result.
        """
        result = self._initialize_run_one_traj(seed, state, tlist, e_ops,
                                               **integrator_kwargs)
        return self._integrate_one_traj(seed, tlist, result)

    def _integrate_one_traj(self, seed, tlist, result):
        for t, state in self._integrator.run(tlist):
            result.add(t, self._restore_state(state, copy=False))
        return seed, result

    def _read_seed(self, seed, ntraj):
        """
        Read user provided seed(s) and produce one for each trajectory.
        Let numpy raise error for inputs that cannot be seeds.
        """
        if seed is None:
            seeds = self.seed_sequence.spawn(ntraj)
        elif isinstance(seed, np.random.SeedSequence):
            seeds = seed.spawn(ntraj)
        elif not isinstance(seed, list):
            seeds = np.random.SeedSequence(seed).spawn(ntraj)
        elif len(seed) >= ntraj:
            seeds = [
                seed_ if (isinstance(seed_, np.random.SeedSequence)
                          or hasattr(seed_, 'random'))
                else np.random.SeedSequence(seed_)
                for seed_ in seed[:ntraj]
            ]
        else:
            raise ValueError("A seed list must be longer than ntraj")
        return seeds

    def _argument(self, args):
        """Update the args, for the `rhs` and `c_ops` and other operators."""
        if args:
            self.rhs.arguments(args)
            self._integrator.arguments(args)

    def _get_generator(self, seed):
        """
        Read the seed and create the random number generator.
        If the ``seed`` has a ``random`` method, it will be used as the
        generator.
        """
        if hasattr(seed, 'random'):
            # We check for the method, not the type to accept pseudo non-random
            # generator for debug/testing purpose.
            return seed

        if self.options['bitgenerator']:
            bit_gen = getattr(np.random, self.options['bitgenerator'])
            generator = np.random.Generator(bit_gen(seed))
        else:
            generator = np.random.default_rng(seed)
        return generator
