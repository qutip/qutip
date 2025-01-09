# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

from .result import Result
from .multitrajresult import MultiTrajResult
from .parallel import _get_map
from time import time
from .solver_base import Solver
from ..core import QobjEvo, Qobj
from ..core.numpy_backend import np
from numpy.typing import ArrayLike
from numpy.random import SeedSequence, default_rng
from numbers import Number
from typing import Any, Callable
import bisect
from operator import itemgetter


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
        self.seed_sequence = SeedSequence()
        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()

    def start(self, state0: Qobj, t0: float, seed: int | SeedSequence = None):
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
        self._integrator.set_state(t0, self._prepare_state(state0), generator)

    def step(
        self, t: float, *, args: dict[str, Any] = None, copy: bool = True
    ) -> Qobj:
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
        if isinstance(state, (list, tuple)):  # mixed initial conditions
            state0 = [(self._prepare_state(psi), p) for psi, p in state]
        else:
            state0 = self._prepare_state(state)
        stats['preparation time'] += time() - start_time
        return seeds, result, map_func, map_kw, state0

    def run(
        self,
        state: Qobj,
        tlist: ArrayLike,
        ntraj: int = 1,
        *,
        args: dict[str, Any] = None,
        e_ops: dict[Any, Qobj | QobjEvo | Callable[[float, Qobj], Any]] = None,
        target_tol: float | tuple[float, float] | list[tuple[float, float]] = None,
        timeout: float = None,
        seeds: int | SeedSequence | list[int | SeedSequence] = None,
    ) -> MultiTrajResult:
        """
        Do the evolution of the Quantum system.

        For a ``state`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :class:`.MultiTrajResult`. The evolution method
        and stored results are determined by ``options``.

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
        seeds, result, map_func, map_kw, state0 = self._initialize_run(
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
        if "generator" in integrator_kwargs:
            generator = integrator_kwargs.pop("generator")
        else:
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
        return *self._integrate_one_traj(seed, tlist, result), 1

    def _integrate_one_traj(self, seed, tlist, result):
        for t, state in self._integrator.run(tlist):
            result.add(t, self._restore_state(state, copy=False))
        return seed, result

    def _run_one_traj_mixed(self, id, seeds, ics,
                            tlist, e_ops, **integrator_kwargs):
        """
        The serial number `id` identifies which seed and which initial state to
        use for running one trajectory.
        """
        seed = seeds[id]
        state, weight = ics.get_state_and_weight(id)

        seed, result, w = self._run_one_traj(seed, state, tlist, e_ops,
                                             **integrator_kwargs)

        return seed, result, weight * w

    def _run_mixed(
        self,
        initial_conditions: list[tuple[Qobj, float]],
        tlist: ArrayLike,
        ntraj: int | list[int],
        *,
        args: dict[str, Any] = None,
        e_ops: dict[Any, Qobj | QobjEvo | Callable[[float, Qobj], Any]] = None,
        timeout: float = None,
        seeds: int | SeedSequence | list[int | SeedSequence] = None,
    ) -> MultiTrajResult:
        """
        Subclasses can use this method to allow simulations with a mixed
        initial state. The following parameters differ from the `run` method:

        Parameters
        ----------
        initial_conditions : list of (:obj:`.Qobj`, float)
            Statistical ensemble at the beginning of the evolution. The first
            element of each tuple is a state contributing to the mixture, and
            the second element is its weight, i.e., a number between 0 and 1
            describing the fraction of the ensemble in that state. The sum of
            all weights is assumed to be one.

        ntraj : {int, list of int}
            Number of trajectories to add. If a single number is provided, this
            will be the total number of trajectories, which are distributed
            over the initial ensemble automatically. This parameter may also be
            a list of numbers with the same number of entries as in
            `initial_conditions`, specifying the number of trajectories for
            each initial state explicitly.

        .. note:
            The simulation will end when the first end condition is reached
            between ``ntraj`` and ``timeout``. Setting a target tolerance is
            not supported with mixed initial conditions.
        """
        seeds, result, map_func, map_kw, prepared_ics = self._initialize_run(
            initial_conditions, np.sum(ntraj), args=args, e_ops=e_ops,
            timeout=timeout, seeds=seeds)
        ics_info = _InitialConditions(prepared_ics, ntraj)
        start_time = time()
        map_func(
            self._run_one_traj_mixed, range(len(seeds)),
            (seeds, ics_info, tlist, e_ops),
            reduce_func=result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        result.stats['run time'] = time() - start_time
        result.initial_states = [self._restore_state(state, copy=False)
                                 for state, _ in ics_info.state_list]
        result.ntraj_per_initial_state = list(ics_info.ntraj)
        return result

    def _read_seed(self, seed, ntraj):
        """
        Read user provided seed(s) and produce one for each trajectory.
        Let numpy raise error for inputs that cannot be seeds.
        """
        if seed is None:
            seeds = self.seed_sequence.spawn(ntraj)
        elif isinstance(seed, SeedSequence):
            seeds = seed.spawn(ntraj)
        elif not isinstance(seed, list):
            seeds = SeedSequence(seed).spawn(ntraj)
        elif len(seed) >= ntraj:
            seeds = [
                seed_ if (isinstance(seed_, SeedSequence)
                          or hasattr(seed_, 'random'))
                else SeedSequence(seed_)
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
        if self.options['bitgenerator']:
            bit_gen = getattr(np.random, self.options['bitgenerator'])
            generator = np.random.Generator(bit_gen(seed))
        else:
            generator = default_rng(seed)
        return generator


class _InitialConditions:
    """
    Information about mixed initial conditions, and the number of trajectories
    to be used for for each state in the mixed ensemble.

    Parameters
    ----------
    state_list : list of (:obj:`.Qobj`, float)
        A list of tuples (state, weight). We assume that all weights add up to
        one.
    ntraj : {int, list of int}
        This parameter may be either the total number of trajectories, or a
        list specifying the number of trajectories to be used per state. In the
        former case, a list of trajectory numbers is generated such that the
        fraction of trajectories for a given state approximates its weight as
        well as possible, under the following constraints:
        1. the total number of trajectories is exactly `ntraj`
        2. there is at least one trajectory per initial state

    Attributes
    ----------
    state_list : list of (:obj:`.Qobj`, float)
        The provided list of states
    ntraj : list of int
        The number of trajectories to be used per state
    ntraj_total : int
        The total number of trajectories
    """
    def __init__(self,
                 state_list: list[tuple[Qobj, float]],
                 ntraj: int | list[int]):
        if not isinstance(ntraj, (list, tuple)):
            ntraj = self._minimum_roundoff_ensemble(state_list, ntraj)

        self.state_list = state_list
        self.ntraj = ntraj
        self._state_selector = np.cumsum(ntraj)
        self.ntraj_total = self._state_selector[-1]

        if len(ntraj) != len(state_list):
            raise ValueError('The length of the `ntraj` list must equal '
                             'the number of states in the initial mixture')
        if not all(n > 0 for n in ntraj):
            raise ValueError('Each initial state must be use for at least '
                             'one trajectory')

    def _minimum_roundoff_ensemble(self, state_list, ntraj_total):
        """
        Calculate a list ntraj from the given total number, under contraints
        explained above. Algorithm based on https://stackoverflow.com/a/792490
        """
        # First we throw out zero-weight states
        filtered_states = [(index, weight)
                           for index, (_, weight) in enumerate(state_list)
                           if weight > 0]
        if len(filtered_states) > ntraj_total:
            raise ValueError(f'{ntraj_total} trajectories is not enough for '
                             f'initial mixture of {len(filtered_states)} '
                             'states')

        # If the trajectory count of a state reaches one, that is final.
        # Here we store the indices of the states with only one trajectory.
        one_traj_states = []

        # All other states are kept here. This is a list of
        # (state index, target weight = w,
        #  current traj number = n, n / (w * ntraj_total) = r)
        # sorted by the last entry. We first make a too large guess for n,
        # then take away trajectories from the states with largest r
        under_consideration = []

        current_total = 0
        for index, weight in filtered_states:
            guess = int(np.ceil(weight * ntraj_total))
            current_total += guess
            if guess == 1:
                one_traj_states.append(index)
            else:
                ratio = guess / (weight * ntraj_total)
                bisect.insort(under_consideration,
                              (index, weight, guess, ratio),
                              key=itemgetter(3))

        while current_total > ntraj_total:
            index, weight, guess, ratio = under_consideration.pop()
            guess -= 1
            current_total -= 1
            if guess == 1:
                one_traj_states.append(index)
            else:
                ratio = guess / (weight * ntraj_total)
                bisect.insort(under_consideration,
                              (index, weight, guess, ratio),
                              key=itemgetter(3))

        # Finally we arrange the results in a list of ntraj
        ntraj = [0] * len(state_list)
        for index in one_traj_states:
            ntraj[index] = 1
        for index, _, count, _ in under_consideration:
            ntraj[index] = count
        return ntraj

    def get_state_index(self, id):
        """
        For the trajectory id (0 <= id < total_ntraj), returns the index of the
        corresponding initial state in the `state_list`.
        """
        state_index = bisect.bisect(self._state_selector, id)
        if id < 0 or state_index >= len(self.state_list):
            raise IndexError(f'State id {id} must be smaller than number of '
                             f'trajectories {self.ntraj_total}')
        return state_index

    def get_state_and_weight(self, id):
        """
        For the trajectory id (0 <= id < total_ntraj), returns the
        corresponding initial state and a correction weight such that
            correction_weight * (ntraj / ntraj_total) = weight
        where ntraj is the number of trajectories used with this initial state
        and weight the initially provided weight of the state in the ensemble.
        """
        state_index = self.get_state_index(id)
        state, target_weight = self.state_list[state_index]
        state_frequency = self.ntraj[state_index] / self.ntraj_total
        correction_weight = target_weight / state_frequency
        return state, correction_weight
