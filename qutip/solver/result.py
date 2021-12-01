""" Class for solve function results"""
import numpy as np
from copy import copy
from ..core import Qobj, QobjEvo, spre, issuper, expect

__all__ = ["Result", "MultiTrajResult"]


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

    Property
    --------
    states : list of Qobj
        Every state of the evolution

    final_state : Qobj
        Last state of the evolution

    expect : list
        list of list of expectation values
        expect[e_ops][t]

    times : list
        list of the times at which the expectation values and
        states where taken.

    stats :
        Diverse statistics of the evolution.

    num_expect : int
        Number of expectation value operators in simulation.

    num_collapse : int
        Number of collapse operators in simualation.
    """
    def __init__(self, e_ops, options, _super, oper_state):
        self.times = []
        self._states = []
        self._expects = []
        self._last_state = None
        self.collapse = None

        self._raw_e_ops = e_ops
        self._e_ops_dict = False
        self._e_ops = []
        if isinstance(self._raw_e_ops, (Qobj, QobjEvo)):
            e_ops = [self._raw_e_ops]
        elif isinstance(self._raw_e_ops, dict):
            self._e_ops_dict = self._raw_e_ops
            e_ops = [e for e in self._raw_e_ops.values()]
        elif callable(self._raw_e_ops):
            e_ops = [self._raw_e_ops]
        elif self._raw_e_ops is None:
            e_ops = []
        else:
            e_ops = self._raw_e_ops

        for e in e_ops:
            if isinstance(e, Qobj):
                self._e_ops.append(_Expect_Caller(e))
            elif isinstance(e, QobjEvo):
                self._e_ops.append(e.expect)
            elif callable(e):
                self._e_ops.append(e)
            self._expects.append([])
        self._e_num = len(self._e_ops) if self._e_ops else 0
        self._read_options(options, _super, oper_state)
        self.stats = {
            "num_expect": self._e_num,
            "solver": "",
            "method": "",
        }

    def _read_options(self, options, _super, oper_state):
        if options['store_states'] is not None:
            self._store_states = options['store_states']
        else:
            self._store_states = self._e_num == 0

        self._store_final_state = options['store_final_state']

        if oper_state:
            # No normalization method (yet?) for operator state (propagators).
            self._normalize_outputs = False
        elif _super:
            # Normalization of density matrix only fix the trace to ``1``.
            self._normalize_outputs = \
                options['normalize_output'] in {'dm', True, 'all'}
        else:
            self._normalize_outputs = \
                options['normalize_output'] in {'ket', True, 'all'}

    def _normalize(self, state):
        return state * (1/state.norm())

    def add(self, t, state, last=True, copy=True):
        """
        Add a state to the results for the time t of the evolution.
        The state is expected to be a Qobj with the right dims.

        last : bool {True}
            Whether the given state *can* be the last state.
        """
        self.times.append(t)
        # this is so we don't make a copy if normalize is
        # false and states are not stored
        if self._normalize_outputs:
            state = self._normalize(state)
        elif copy:
            state = state.copy()

        if self._store_states:
            self._states.append(state)
        elif self._store_final_state and last:
            self._last_state = state

        for i, e_call in enumerate(self._e_ops):
            self._expects[i].append(e_call(t, state))

    @property
    def states(self):
        return self._states

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
        if self._e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(self._e_ops_dict.keys())}
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
    Contain result of simulations with multiple trajectories.

    Parameters
    ----------
    ntraj : int
        Number of trajectories expected.

    state : Qobj
        Initial state of the evolution.

    tlist : array_like
        Times at which the expectation results are desired.

    e_ops : Qobj, QobjEvo, callable or iterable of these.
        list of Qobj or QobjEvo to compute the expectation values.
        Alternatively, function[s] with the signature f(t, state) -> expect
        can be used.

    solver_id : int, [optional]
        Identifier of the Solver creating the object.

    options : SolverResultsOptions, [optional]
        Options conserning result to save.
    """
    def __init__(self, ntraj, state, tlist, e_ops, solver_id=0, options=None):
        """
        Parameters:
        -----------
        num_c_ops: int
            Number of collapses operator used in the McSolver
        """
        self.options = copy(options) or SolverResultsOptions()
        self.initial_state = state
        self.tlist = tlist
        self.solver_id = solver_id
        self._save_traj = self.options['keep_runs_results']
        self.trajectories = []
        self._sum_states = None
        self._sum_last_states = None
        self._sum_expect = None
        self._sum2_expect = None
        e_ops = e_ops or []
        if not isinstance(e_ops, (list, dict)):
            e_ops = [e_ops]
        self.num_e_ops = len(e_ops or ())
        self.e_ops = e_ops
        self._e_ops_dict = e_ops if isinstance(e_ops, dict) else False
        self.num_c_ops = 0
        self._target_ntraj = ntraj
        self._num = 0
        self._collapse = []
        self.seeds = []
        self.stats = {
            "num_expect": self.num_e_ops,
            "solver": "",
            "method": "",
        }
        self._target_tols = None
        self._tol_reached = False

    def add(self, one_traj):
        """
        Add a trajectory.
        Return the number of trajectories still needed to reach the desired
        tolerance.
        """
        if self._save_traj:
            self.trajectories.append(one_traj)
        else:
            if self._num == 0:
                self.trajectories = [one_traj]
                if one_traj.states and one_traj.states[0].isket:
                    self._sum_states = [state.proj()
                                        for state in one_traj.states]
                else:
                    self._sum_states = one_traj.states
                if one_traj.final_state and one_traj.final_state.isket:
                    self._sum_last_states = one_traj.final_state.proj()
                else:
                    self._sum_last_states = one_traj.final_state
                self._sum_expect = [np.array(expect)
                                    for expect in one_traj._expects]
                self._sum2_expect = [np.abs(np.array(expect))**2
                                     for expect in one_traj._expects]
            else:
                if self._sum_states and one_traj.states[0].isket:
                    self._sum_states = [
                        state.proj() + accu
                        for accu, state
                        in zip(self._sum_states, one_traj.states)
                    ]
                elif self._sum_states:
                    self._sum_states = [
                        state + accu
                        for accu, state
                        in zip(self._sum_states, one_traj.states)
                    ]
                if self._sum_last_states and one_traj.final_state.isket:
                    self._sum_last_states += one_traj.final_state.proj()
                elif self._sum_last_states:
                    self._sum_last_states += one_traj.final_state

                if self._sum_expect:
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
        self._num += 1
        if hasattr(one_traj, 'collapse'):
            self._collapse.append(one_traj.collapse)
        if hasattr(one_traj, 'seed'):
            self.seeds.append(one_traj.seed)

        if self._target_tols is not None:
            num_traj = self._num
            if num_traj >= self.next_check:
                traj_left = self._check_expect_tol()
                target = traj_left + num_traj
                confidence = 0.5 * (1 - 3 / num_traj)
                confidence += 0.5 * min(1 / abs(target - self.last_target), 1)
                self.next_check = int(traj_left * confidence + num_traj)
                self.last_target = target
                return traj_left
            else:
                return max(self.last_target - self._num, 1)
        else:
            return np.inf

    def set_expect_tol(self, target_tol):
        """
        Set the capacity to stop the map when the estimated error on the
        expectation values is within given tolerance.

        Error estimation is done with jackknife resampling.

        target_tol : float, list
            If a float, it is read as absolute tolerance.
            If a pair of float: absolute and relative tolerance in that order.
            Lastly, target_tol can be a list of pairs of (atol, rtol) for each
            e_ops.



        target_tol : float, list, [optional]
            Target tolerance of the evolution. The evolution will compute
            trajectories until the error on the expectation values is lower than
            this tolerance. The error is computed using jackknife resampling.
            ``target_tol`` can be an absolute tolerance, a pair of absolute and
            relative tolerance, in that order. Lastly, it can be a list of pairs of
            (atol, rtol) for each e_ops.
        """
        self._target_tols = None
        self._tol_reached = False
        if not target_tol:
            return
        if not self.num_e_ops:
            raise ValueError("Cannot target a tolerance without e_ops")
        self.next_check = 5
        self.last_target = np.inf

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

    def spawn(self, _super, oper_state):
        """
        Create a :cls:`Result` for a trajectory of this ``MultiTrajResult``.
        """
        return Result(self.e_ops, self.options, _super, oper_state)

    @property
    def mean_expect(self):
        if self._save_traj:
            return [np.mean(
                np.stack([traj._expects[i] for traj in self.trajectories]),
                axis=0
            ) for i in range(self.num_e_ops)]
        else:
            return [sum_expect / self._num
                    for sum_expect in self._sum_expect]

    @property
    def mean_expect2(self):
        if self._save_traj:
            return [np.mean(
                np.stack([np.abs(traj._expects[i])**2
                          for traj in self.trajectories]),
                axis=0
            ) for i in range(self.num_e_ops)]
        else:
            return [sum_expect / self._num
                    for sum_expect in self._sum2_expect]

    def _check_expect_tol(self):
        """
        Compute the error on the expectation values using jackknife resampling.
        Return the approximate number of trajectories needed to reach the
        desired tolerance.
        """
        if self._num <= 1:
            return np.inf
        avg = np.array(self.mean_expect)
        avg2 = np.array(self.mean_expect2)
        target = np.array([atol + rtol * mean
                  for mean, (atol, rtol) in zip(avg, self._target_tols)])
        traj_left = np.max((avg2 - abs(avg)**2) / target**2 - self._num + 1)
        self._tol_reached = traj_left < 0
        return traj_left

    @property
    def runs_states(self):
        if self._save_traj:
            return [traj.states for traj in self.trajectories]
        else:
            return None

    @property
    def average_states(self):
        if not self._save_traj:
            finals = self._sum_states
        elif self.trajectories[0].states[0].isket:
            finals = [state.proj() for state in self.trajectories[0].states]
            for i in range(1, len(self.trajectories)):
                finals = [state.proj() + final for final, state
                          in zip(finals, self.trajectories[i].states)]
        else:
            finals = [state for state in self.trajectories[0].states]
            for i in range(1, len(self.trajectories)):
                finals = [state + final for final, state
                          in zip(finals, self.trajectories[i].states)]
        return [final / self._num for final in finals]

    @property
    def runs_final_states(self):
        if self._save_traj:
            return [traj.final_state for traj in self.trajectories]
        else:
            return None

    @property
    def average_final_state(self):
        if self.trajectories[0].final_state is None:
            return None
        if not self._save_traj:
            final = self._sum_last_states
        elif self.trajectories[0].states[0].isket:
            final = sum(traj.final_state.proj() for traj in self.trajectories)
        else:
            final = sum(traj.final_state for traj in self.trajectories)
        return final / self._num

    @property
    def steady_state(self):
        return sum(self.average_states) / len(self.times)

    def _format_expect(self, expect):
        if self._e_ops_dict:
            expect = {e: expect[n]
                      for n, e in enumerate(self._e_ops_dict.keys())}
        return expect

    @property
    def average_expect(self):
        result = self.mean_expect
        return self._format_expect(result)

    @property
    def std_expect(self):
        avg = self.mean_expect
        avg2 = self.mean_expect2
        result = [np.sqrt(a2 - abs(a*a)) for a, a2 in zip(avg, avg2)]
        return self._format_expect(result)

    @property
    def runs_expect(self):
        if not self._save_traj:
            return None
        result = [np.stack([traj._expects[i] for traj in self.trajectories])
               for i in range(self.num_e_ops)]
        return self._format_expect(result)

    def expect_traj_avg(self, ntraj=-1):
        if not self._save_traj:
            return None
        result = [
            np.mean(np.stack([
                traj._expects[i]
                for traj in self.trajectories[:ntraj]
            ]), axis=0)
            for i in range(self.num_e_ops)
        ]
        return self._format_expect(result)

    def expect_traj_std(self, ntraj=-1):
        if not self._save_traj:
            return None
        result = [
            np.std(np.stack([
                traj._expects[i]
                for traj in self.trajectories[:ntraj]
            ]), axis=0)
            for i in range(self.num_e_ops)
        ]
        return self._format_expect(result)

    @property
    def collapse(self):
        return self._collapse

    @property
    def col_times(self):
        if self._collapse is None:
            return None
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[0])
            out.append(col)
        return out

    @property
    def col_which(self):
        if self._collapse is None:
            return None
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[1])
            out.append(col)
        return out

    @property
    def photocurrent(self):
        if self._collapse is None:
            return None
        cols = {}
        tlist = self.times
        for traj in self.trajectories:
            for t, which in traj.collapse:
                if which in cols:
                    cols[which].append(t)
                else:
                    cols[which] = [t]
        mesurement = []
        for i in range(self.num_c_ops):
            mesurement += [(np.histogram(cols.get(i,[]), tlist)[0]
                          / np.diff(tlist) / self._num)]
        return mesurement

    @property
    def measurements(self):
        if self._collapse is None:
            return None
        tlist = self.times
        measurements = []
        for collapses in self.collapse:
            cols = [[] for _ in range(self.num_c_ops)]
            for t, which in collapses:
                cols[which].append(t)
            measurement = [(np.histogram(cols[i], tlist)[0] / np.diff(tlist))
                          for i in range(self.num_c_ops)]
            measurements.append(measurement)
        return measurements

    def __repr__(self):
        out = ""
        out += self.stats['solver'] + "\n"
        out += "solver : " + self.stats['method'] + "\n"
        if self._save_traj:
            out += "{} runs saved\n".format(self.num_traj)
        else:
            out += "{} trajectories averaged\n".format(self.num_traj)
        out += "number of expect : {}\n".format(self.num_e_ops)
        if self.trajectories[0]._store_states:
            out += "States saved\n"
        elif self.trajectories[0]._store_final_state:
            out += "Final state saved\n"
        else:
            out += "State not available\n"
        out += "times from {} to {} in {} steps\n".format(
            self.times[0], self.times[-1], len(self.times))
        return out

    @property
    def times(self):
        return self.tlist

    @property
    def states(self):
        return self.average_states

    @property
    def expect(self):
        return self.average_expect

    @property
    def final_state(self):
        return self.average_final_state

    @property
    def num_traj(self):
        return self._num

    @property
    def num_expect(self):
        return self.num_e_ops

    @property
    def num_collapse(self):
        return self.num_c_ops

    @property
    def end_condition(self):
        if self._target_tols is not None and self._tol_reached:
            return "target tolerance reached"
        elif self._target_ntraj == self._num:
            return "ntraj reached"
        else:
            return "timeout"
