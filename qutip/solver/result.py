""" Class for solve function results"""
import numpy as np
from copy import copy
from ..core import Qobj, QobjEvo, spre, issuper, expect, ket2dm

__all__ = ["Result", "MultiTrajResult", "McResult"]


class _Expect_Caller:
    """pickable partial(expect, oper) with extra `t` input"""
    def __init__(self, oper):
        self.oper = oper

    def __call__(self, t, state):
        return expect(self.oper, state)


class Result:
    """
    Class for storing simulation results from single trajectory
    dynamics solvers.

    Parameters
    ----------
    e_ops : list of :class:`Qobj` / callback function
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    options : mapping
        Result options, dict or equivalent containing entry for 'store_states',
        'store_final_state' and 'normalize_output'. Only ket and density
        matrices can be normalized.

    tlist : float, iterable, [optional]
        Time of the first time of tlist or the full tlist. If tlist is passed,
        it will be used to recognized when the last state is added if
        options['store_final_state'] is True.

    state0 : Qobj, [optional]
        First state of the evolution.
    """
    def __init__(self, e_ops, options, tlist=None, state0=None):
        # Initialize output data
        self._times = []
        self._states = []
        self._expects = []
        self._last_state = None
        self._last_time = -np.inf
        if hasattr(tlist, '__iter__'):
            # We only want the first and last values.
            self._last_time = tlist[-1]
            tlist = tlist[0]

        # Read e_ops
        e_ops_list = self._e_ops_as_list(e_ops)
        self._e_num = len(e_ops_list)

        self._e_ops = []
        dims = state0.dims[0] if state0 is not None else None
        for e_op in e_ops_list:
            self._e_ops.append(self._read_single_e_op(e_op, dims))
            self._expects.append([])

        # Read options
        self._read_options(options, state0)

        # Write some extra info.
        self.stats = {
            "num_expect": self._e_num,
            "solver": "",
            "method": "",
        }

        if state0 is not None:
            self.add(tlist, state0)

    def _e_ops_as_list(self, e_ops):
        """ Promote ``e_ops`` to a list. """
        self._e_ops_format = False

        if isinstance(e_ops, list):
            pass
        elif e_ops is None:
            e_ops = []
        elif isinstance(e_ops, (Qobj, QobjEvo)):
            e_ops = [e_ops]
            self._e_ops_format = "single"
        elif callable(e_ops):
            e_ops = [e_ops]
            self._e_ops_format = "single"
        elif isinstance(e_ops, dict):
            self._e_ops_format = [e for e in e_ops.keys()]
            e_ops = [e for e in e_ops.values()]
        else:
            raise TypeError("e_ops format not understood.")

        return e_ops

    def _read_single_e_op(self, e_op, dims):
        """ Promote each c_ops to a callable(t, state). """
        if isinstance(e_op, Qobj):
            if dims and e_op.dims[1] != dims:
                raise TypeError("Dimensions of the e_ops do "
                                "not match the state")
            e_op_call = _Expect_Caller(e_op)
        elif isinstance(e_op, QobjEvo):
            if dims and e_op.dims[1] != dims:
                raise TypeError("Dimensions of the e_ops do "
                                "not match the state")
            e_op_call = e_op.expect
        elif callable(e_op):
            e_op_call = e_op
        else:
            raise TypeError("e_ops format not understood.")
        return e_op_call

    def _read_options(self, options, state0):
        """ Read options. """
        if options['store_states'] is not None:
            self._store_states = options['store_states']
        else:
            self._store_states = self._e_num == 0

        self._store_final_state = (
            options['store_final_state']
            and not self._store_states
        )

        # TODO: Reminder for when reworking options
        # By having separate option for mesolve and sesolve, we could simplify
        # the way we decide whether to normalize the state.

        # We can normalize ket and dm, but not operators.
        if state0 is None:
            # Cannot guess the type of state, we trust the option.
            normalize = options['normalize_output'] in {True, 'all'}
        elif state0.isket:
            normalize = options['normalize_output'] in {'ket', True, 'all'}
        elif (
            state0.dims[0] != state0.dims[1]  # rectangular states
            or state0.issuper  # super operator state
            or abs(state0.norm()-1) > 1e-10  # initial state is not normalized
        ):
            # We don't try to normalize those.
            normalize = False
        else:
            # The state is an operator with a trace of 1,
            # While this is not enough to be 100% certain that we are working
            # with a density matrix, odd are good enough that we go with it.
            normalize = options['normalize_output'] in {'dm', True, 'all'}
        self._normalize_outputs = normalize

    def _normalize(self, state):
        return state * (1/state.norm())

    def add(self, t, state, copy=True):
        """
        Add a state to the results for the time t of the evolution.
        The state is expected to be a Qobj with the right dims.
        """
        if self._times and self._times[-1] == t:
            # State added twice
            return
        self._times.append(t)

        if self._normalize_outputs:
            state = self._normalize(state)
            copy = False  # normalization create a copy.

        if self._store_states:
            self._states.append(state.copy() if copy else state)
        elif self._store_final_state and t >= self._last_time:
            self._last_state = state.copy() if copy else state

        for i, e_call in enumerate(self._e_ops):
            self._expects[i].append(e_call(t, state))

    @property
    def times(self):
        return self._times.copy()

    @property
    def states(self):
        return self._states.copy()

    @property
    def final_state(self):
        if self._store_states:
            return self._states[-1]
        elif self._store_final_state:
            return self._last_state
        else:
            return None

    @property
    def expect(self):
        result = []
        for expect_vals in self._expects:
            result.append(np.array(expect_vals))
        if self._e_ops_format == 'single':
            result = result[0]
        elif self._e_ops_format:
            result = {e: result[n] for n, e in enumerate(self._e_ops_format)}
        return result

    @property
    def num_expect(self):
        return self._e_num

    @property
    def num_collapse(self):
        if 'num_collapse' in self.stats:
            return self.stats["num_collapse"]
        else:
            return 0

    def __repr__(self):
        out = ""
        out += self.stats['solver'] + "\n"
        out += "solver : " + self.stats['method'] + "\n"
        out += "number of expect : {}\n".format(self._e_num)
        if self._store_states:
            out += "State saved\n"
        elif self._store_final_state:
            out += "Final state saved\n"
        else:
            out += "State not available\n"
        out += "times from {} to {} in {} steps\n".format(self.times[0],
                                                          self.times[-1],
                                                          len(self.times))
        return out


class MultiTrajResult:
    """
    Contain the results of simulations with multiple trajectories.

    Parameters
    ----------
    e_ops : Qobj, QobjEvo, callable or iterable of these.
        list of Qobj or QobjEvo to compute the expectation values.
        Alternatively, function[s] with the signature f(t, state) -> expect
        can be used.

    options : SolverResultsOptions, [optional]
        Options conserning result to save.

    tlist : array_like
        Times at which the expectation results are desired.

    state : Qobj
        Initial state of the evolution.

    solver_id : int, [optional]
        Identifier of the Solver creating the object.
    """
    _sum_states = None
    _sum_final_states = None
    _sum_expect = None
    _sum2_expect = None
    _target_ntraj = None
    _target_tols = None
    _tol_reached = False
    _first_traj = None
    _trajectories = None

    def __init__(self, e_ops, options, tlist, state0, solver_id=0):
        self._evolution_info = {
            'e_ops': e_ops,
            'options': options,
            'tlist': tlist,
            'state0': state0,
        }

        self.solver_id = solver_id  # Used when adding trajectories to result
        self._save_traj = options['keep_runs_results']
        self.num_e_ops = len(self._e_ops_as_list(e_ops))
        self._num = 0
        self.seeds = []
        self._proj = ket2dm if state0.isket else lambda x: x

        self.stats = {
            "num_expect": self.num_e_ops,
            "solver": "",
            "method": "",
        }

    def spawn(self):
        return Result(*self._evolution_info.values())

    def _add_first_traj(self, one_traj):
        """
        Read the first trajectory, intitializing needed data.
        """
        self._first_traj = one_traj

        if self._save_traj:
            self._trajectories = []
        if not self._save_traj:
            if one_traj.states:
                self._sum_states = [self._proj(state)
                                    for state in one_traj.states]
            elif one_traj.final_state:
                self._sum_final_states = self._proj(one_traj.final_state)

        if not self._save_traj or self._target_tols is not None:
            self._sum_expect = [np.array(expect)
                                for expect in one_traj._expects]
            self._sum2_expect = [np.abs(np.array(expect))**2
                                 for expect in one_traj._expects]

    def _reduce_traj(self, one_traj):
        """
        Sum result from a Result object to average.
        """
        if self._sum_states is not None:
            self._sum_states = [
                accu + self._proj(state)
                for accu, state
                in zip(self._sum_states, one_traj.states)
            ]

        if self._sum_final_states is not None:
            self._sum_final_states += self._proj(one_traj.final_state)

        if self._sum_expect is not None:
            self._sum_expect = [
                np.array(one) + accu
                for one, accu
                in zip(one_traj._expects, self._sum_expect)
            ]
            self._sum2_expect = [
                np.abs(np.array(one))**2 + accu
                for one, accu
                in zip(one_traj._expects, self._sum2_expect)
            ]

    def add(self, one_traj):
        """
        Add a trajectory.
        Return the number of trajectories still needed to reach the desired
        tolerance.
        """
        if self._num == 0:
            self._add_first_traj(one_traj)
        else:
            self._reduce_traj(one_traj)

        if self._save_traj:
            self._trajectories += [one_traj]

        self._num += 1

        if hasattr(one_traj, 'seed'):
            self.seeds.append(one_traj.seed)

        if self._target_ntraj is not None:
            return self._get_traj_to_tol()
        return np.inf

    def _get_traj_to_tol(self):
        """
        Estimate the number of trajectories needed to reach desired tolerance.
        """
        if self._target_tols is None:
            return self._target_ntraj - self._num
        if self._num >= self._next_check:
            traj_left = self._check_expect_tol()

            # We don't check the tol each trajectory since it can be slow.
            # _next_check will be the next time we compute it.
            # For gaussian distribution, this ad hoc method usually reach the
            # target in about 10 tries without over shooting it.
            target = traj_left + self._num
            confidence = 0.5 * (1 - 5 / self._num)
            confidence += 0.5 * min(1 / abs(target - self._estimated_ntraj), 1)
            self._next_check = int(traj_left * confidence + self._num)
            self._estimated_ntraj = target
            return traj_left
        else:
            return max(self._estimated_ntraj - self._num, 1)

    def set_expect_tol(self, target_tol=None, ntraj=None):
        """
        Set the capacity to stop the map when the estimated error on the
        expectation values is within given tolerance.

        Error estimation is done with jackknife resampling.

        target_tol : float, array_like, [optional]
            Target tolerance of the evolution. The evolution will compute
            trajectories until the error on the expectation values is lower
            than this tolerance. The error is computed using jackknife
            resampling. ``target_tol`` can be an absolute tolerance, a pair of
            absolute and relative tolerance, in that order. Lastly, it can be a
            list of pairs of (atol, rtol) for each e_ops.

        ntraj : int, [optional]
            Number of trajectories expected.
        """
        self._estimated_ntraj = ntraj or np.inf
        self._target_ntraj = ntraj
        self._target_tols = None
        self._tol_reached = False

        if not target_tol:
            return

        if not self.num_e_ops:
            raise ValueError("Cannot target a tolerance without e_ops")
        self._next_check = 10

        targets = np.array(target_tol)
        if targets.ndim == 0:
            self._target_tols = np.array([(target_tol, 0.)] * self.num_e_ops)
        elif targets.shape == (2,):
            self._target_tols = np.ones((self.num_e_ops, 2)) * targets
        elif targets.shape == (self.num_e_ops, 2):
            self._target_tols = targets
        else:
            raise ValueError("target_tol must be a number, a pair of (atol, "
                             "rtol) or a list of (atol, rtol) for each e_ops")

    def _check_expect_tol(self):
        """
        Compute the error on the expectation values using jackknife resampling.
        Return the approximate number of trajectories needed to reach the
        desired tolerance.
        """
        if self._num <= 1:
            return np.inf
        avg = np.array(self._mean_expect())
        avg2 = np.array(self._mean_expect2())
        target = np.array([atol + rtol * mean
                           for mean, (atol, rtol)
                           in zip(avg, self._target_tols)])
        traj_left = np.max((avg2 - abs(avg)**2) / target**2 - self._num + 1)
        self._tol_reached = traj_left < 0
        return traj_left

    def _e_ops_as_list(self, e_ops):
        """ Promote ``e_ops`` to a list. """
        self._e_ops_format = False

        if isinstance(e_ops, list):
            pass
        elif e_ops is None:
            e_ops = []
        elif isinstance(e_ops, (Qobj, QobjEvo)):
            e_ops = [e_ops]
            self._e_ops_format = "single"
        elif callable(e_ops):
            e_ops = [e_ops]
            self._e_ops_format = "single"
        elif isinstance(e_ops, dict):
            self._e_ops_format = [e for e in e_ops.keys()]
            e_ops = [e for e in e_ops.values()]
        else:
            raise TypeError("e_ops format not understood.")

        return e_ops

    @property
    def runs_states(self):
        """
        States of every runs as ``states[run][t]``.
        """
        if self._save_traj:
            return [traj.states for traj in self._trajectories]
        else:
            return None

    @property
    def average_states(self):
        """
        States averages as density matrices.
        """
        if self._first_traj.states is None:
            return None
        if self._sum_states is None:
            self._sum_states = [self._proj(state)
                                for state in self._trajectories[0].states]
            for i in range(1, len(self._trajectories)):
                self._sum_states = [
                    self._proj(state) + sums
                    for sums, state
                    in zip(self._sum_states, self._trajectories[i].states)
                ]
        return [final / self._num for final in self._sum_states]

    @property
    def states(self):
        """
        Runs final states if available, average otherwise.
        This imitate v4's behaviour, expect for the steady state which must be
        obtained directly.
        """
        return self.runs_states or self.average_states

    @property
    def runs_final_states(self):
        """
        Last states of each trajectories.
        """
        if self._save_traj:
            return [traj.final_state for traj in self._trajectories]
        else:
            return None

    @property
    def average_final_state(self):
        """
        Last states of each trajectories averaged into a density matrix.
        """
        if self._first_traj.final_state is None:
            return None
        if self._sum_final_states is None:
            if self._sum_states:
                self._sum_final_states = self._sum_states[-1]
            else:
                self._sum_final_states = sum(self._proj(traj.final_state)
                                            for traj in self._trajectories)
        return self._sum_final_states / self._num

    @property
    def final_state(self):
        """
        Runs final states if available, average otherwise.
        This imitate v4's behaviour.
        """
        return self.runs_final_states or self.average_final_state

    def steady_state(self, N=0):
        """
        Average the states of the last ``N`` times of every runs as a density
        matrix. Should converge to the steady state in the right circumstances.

        Parameters
        ----------
        N : int [optional]
            Number of states from the end of ``tlist`` to average. Per default
            all states will be averaged.
        """
        N = int(N) or len(self.times)
        N = len(self.times) if N > len(self.times) else N
        states = self.average_states
        if states is not None:
            return sum(states[-N:]) / N
        else:
            return None

    def _format_expect(self, expect):
        """
        Restore the format of expect.
        """
        if self._e_ops_format == "single":
            expect = expect[0]
        elif self._e_ops_format:
            expect = {e: expect[n] for n, e in enumerate(self._e_ops_format)}
        return expect

    def _mean_expect(self):
        """
        Average of expectation values as list of numpy array.
        """
        if self._sum_expect is None:
            self._sum_expect = [np.sum(
                np.stack([traj._expects[i] for traj in self._trajectories]),
                axis=0
            ) for i in range(self.num_e_ops)]

        return [sum_expect / self._num for sum_expect in self._sum_expect]

    def _mean_expect2(self):
        """
        Average of the square of expectation values as list of numpy array.
        """
        if self._sum2_expect is None:
            self._sum2_expect = [np.sum(
                np.stack([np.abs(traj._expects[i])**2
                          for traj in self._trajectories]),
                axis=0
            ) for i in range(self.num_e_ops)]

        return [sum_expect / self._num for sum_expect in self._sum2_expect]

    @property
    def average_expect(self):
        """
        Average of the expectation values.
        Return a ``dict`` if ``e_ops`` was one.
        """
        result = self._mean_expect()
        return self._format_expect(result)

    @property
    def std_expect(self):
        """
        Standard derivation of the expectation values.
        Return a ``dict`` if ``e_ops`` was one.
        """
        avg = self._mean_expect()
        avg2 = self._mean_expect2()
        result = [np.sqrt(a2 - abs(a*a)) for a, a2 in zip(avg, avg2)]
        return self._format_expect(result)

    @property
    def runs_expect(self):
        """
        Expectation values for each trajectories as ``expect[e_op][run][t]``.
        Return ``None`` is run data is not saved.
        Return a ``dict`` if ``e_ops`` was one.
        """
        if not self._save_traj:
            return None
        result = [np.stack([traj._expects[i] for traj in self._trajectories])
                  for i in range(self.num_e_ops)]
        return self._format_expect(result)

    @property
    def expect(self):
        """
        Runs expectation values if available, average otherwise.
        This imitate v4's behaviour.
        """
        return self.runs_expect or self.average_expect

    def expect_traj_avg(self, ntraj=-1):
        """
        Average of the expectation values for the ``ntraj`` first runs.
        Return a ``dict`` if ``e_ops`` was one.

        Parameters
        ----------
        ntraj : int, [optional]
            Number of trajectories's expect to average.
            Default: all trajectories.
        """
        if not self._save_traj:
            return None
        result = [
            np.mean(np.stack([
                traj._expects[i]
                for traj in self._trajectories[:ntraj]
            ]), axis=0)
            for i in range(self.num_e_ops)
        ]
        return self._format_expect(result)

    def expect_traj_std(self, ntraj=-1):
        """
        Standard derivation of the expectation values for the ``ntraj``
        first runs.
        Return a ``dict`` if ``e_ops`` was one.

        Parameters
        ----------
        ntraj : int, [optional]
            Number of trajectories's expect to compute de standard derivation.
            Default: all trajectories.
        """
        if not self._save_traj:
            return None
        result = [
            np.std(np.stack([
                traj._expects[i]
                for traj in self._trajectories[:ntraj]
            ]), axis=0)
            for i in range(self.num_e_ops)
        ]
        return self._format_expect(result)

    def __repr__(self):
        out = f"Result from {self.stats['solver']}\n"
        out += f"solver : {self.stats['method']}\n"
        out += f"Ended by : " + self.end_condition
        if self._save_traj:
            out += f"{self.num_traj} runs saved\n"
        else:
            out += f"{self.num_traj} trajectories averaged\n"
        out += "number of expect : {self.num_e_ops}\n"
        if self._first_traj._store_states:
            out += "States saved\n"
        elif self._first_traj._store_final_state:
            out += "Final state saved\n"
        else:
            out += "State not available\n"
        out += (f"times from {self.times[0]} to {self.times[-1]}"
                f" in {len(self.times)} steps\n")
        return out

    @property
    def times(self):
        return self._first_traj.times

    @property
    def num_traj(self):
        return self._num

    @property
    def num_expect(self):
        return self.num_e_ops

    @property
    def end_condition(self):
        if self._target_tols is not None and self._tol_reached:
            end_condition = "target tolerance reached"
        elif self._target_ntraj == self._num:
            end_condition = "ntraj reached"
        elif self._target_ntraj is not None:
            end_condition = "timeout"
        else:
            end_condition = "unknown"
        return end_condition


class McResult(MultiTrajResult):
    # Collapse are only produced by mcsolve.
    def __init__(self, e_ops, options, tlist, state0, num_c_ops, solver_id=0):
        self.num_c_ops = num_c_ops
        super().__init__(e_ops, options, tlist, state0, solver_id)
        self._collapse = []

    def add(self, one_traj):
        out = super().add(one_traj)
        self._collapse.append(one_traj.collapse)
        return out

    @property
    def collapse(self):
        """
        For each runs, a list of every collapse as a tuple of the time it
        happened and the corresponding ``c_ops`` index.
        """
        return self._collapse

    @property
    def col_times(self):
        """
        List of the times of the collapses for each runs.
        """
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[0])
            out.append(col)
        return out

    @property
    def col_which(self):
        """
        List of the indexes of the collapses for each runs.
        """
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[1])
            out.append(col)
        return out

    @property
    def photocurrent(self):
        """
        Average photocurrent or measurement of the evolution.
        """
        cols = [[] for _ in range(self.num_c_ops)]
        tlist = self.times
        for collapses in self.collapse:
            for t, which in collapses:
                cols[which].append(t)
        mesurement = [
            np.histogram(cols[i], tlist)[0] / np.diff(tlist) / self._num
            for i in range(self.num_c_ops)
        ]
        return mesurement

    @property
    def runs_photocurrent(self):
        """
        Photocurrent or measurement of each runs.
        """
        tlist = self.times
        measurements = []
        for collapses in self.collapse:
            cols = [[] for _ in range(self.num_c_ops)]
            for t, which in collapses:
                cols[which].append(t)
            measurements.append([
                np.histogram(cols[i], tlist)[0] / np.diff(tlist)
                for i in range(self.num_c_ops)
            ])
        return measurements

    @property
    def num_collapse(self):
        return self.num_c_ops
