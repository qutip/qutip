""" Class for solve function results"""
import numpy as np
from ..core import Qobj, QobjEvo, spre, issuper, expect

__all__ = ["Result", "MultiTrajResult", "MultiTrajResultAveraged"]


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

        self._raw_e_ops = e_ops
        self._states = []
        self._expects = []
        self._last_state = None

        self.collapse = None

        self._e_ops_dict = False
        self._e_num = 0
        self._e_ops = []
        self.stats = {
            "num_expect": self._e_num,
            "solver": "",
            "method": "",
        }

        self._read_options(options, _super, oper_state)

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

        self._e_num = len(e_ops)

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
                options['normalize_output'] in ['dm', True, 'all']
        else:
            self._normalize_outputs = \
                options['normalize_output'] in ['ket', True, 'all']

    def _normalize(self, state):
        return state * (1/state.norm())

    def add(self, t, state):
        """
        Add a state to the results for the time t of the evolution.
        The state is expected to be a Qobj with the right dims.
        """
        self.times.append(t)
        # this is so we don't make a copy if normalize is
        # false and states are not stored
        state_norm = False
        if self._normalize_outputs:
            state_norm = self._normalize(state)

        if self._store_states:
            self._states.append(state_norm or state.copy())
        elif self._store_final_state:
            self._last_state = state_norm or state.copy()

        for i, e_call in enumerate(self._e_ops):
            self._expects[i].append(e_call(t, state_norm or state))

    def copy(self):
        return Result(self._raw_e_ops, self.options, self.super)

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
    Contain result of simulations with multiple trajectories. Keeps all
    trajectories' data.

    Property
    --------

    runs_states : list of list of Qobj
        Every state of the evolution for each trajectories. (ket)

    average_states : list of Qobj
        Average state for each time. (density matrix)

    runs_final_states : Qobj
        Average last state for each trajectories. (ket)

    average_final_state : Qobj
        Average last state. (density matrix)

    steady_state : Qobj
        Average state of each time and trajectories. (density matrix)

    runs_expect : list of list of list of number
        Expectation values for each [e_ops, trajectory, time]

    average_expect : list of list of number
        Averaged expectation values over trajectories.

    std_expect : list of list of number
        Standard derivation of each averaged expectation values.

    expect : list
        list of list of averaged expectation values.

    times : list
        list of the times at which the expectation values and
        states where taken.

    stats :
        Diverse statistics of the evolution.

    num_expect : int
        Number of expectation value operators in simulation.

    num_collapse : int
        Number of collapse operators in simualation.

    num_traj : int/list
        Number of trajectories (for stochastic solvers). A list indicates
        that averaging of expectation values was done over a subset of total
        number of trajectories.

    col_times : list
        Times at which state collpase occurred. Only for Monte Carlo solver.

    col_which : list
        Which collapse operator was responsible for each collapse in
        ``col_times``. Only for Monte Carlo solver.

    collapse : list
        Each collapse per trajectory as a (time, which_oper)

    photocurrent : list
        photocurrent corresponding to each collapse operator.

    Methods
    -------
    expect_traj_avg(ntraj):
        Averaged expectation values over `ntraj` trajectories.

    expect_traj_std(ntraj):
        Standard derivation of expectation values over `ntraj` trajectories.
        Last state of each trajectories. (ket)
    """
    def __init__(self, ntraj, e_ops=(), c_ops=(), target_tol=None):
        """
        Parameters:
        -----------
        num_c_ops: int
            Number of collapses operator used in the McSolver
        """
        self.trajectories = []
        self.num_e_ops = len(e_ops)
        self.num_c_ops = len(c_ops)
        self.tlist = None
        self._target_ntraj = ntraj
        self.stats = {
            "num_expect": self.num_e_ops,
            "solver": "",
            "method": "",
        }
        self._set_expect_tol(target_tol)

    def add(self, one_traj):
        """
        Add a trajectory.
        Return the number of trajectories still needed to reach the desired
        tolerance.
        """
        self.trajectories.append(one_traj)
        if self._target_tols is not None:
            num_traj = len(self.trajectories)
            if num_traj >= self.next_check:
                traj_left = self._check_expect_tol()
                target = traj_left + num_traj
                confidence = 0.5 * (1 - 3 / num_traj)
                confidence += 0.5 * min(1 / abs(target - self.last_target), 1)
                self.next_check = int(traj_left * confidence + num_traj)
                self.last_target = target
                return traj_left
            else:
                return max(self.last_target - len(self.trajectories), 1)
        else:
            return np.inf

    def _set_expect_tol(self, target_tol):
        """
        Set the capacity to stop the map when the estimated error on the
        expectation values is within given tolerance.

        Error estimation is done with jackknife resampling.

        target_tol : float, list
            If a float, it is read as absolute tolerance.
            If a pair of float: absolute and relative tolerance in that order.
            Lastly, target_tol can be a list of pairs of (atol, rtol) for each
            e_ops.
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

    def _check_expect_tol(self):
        """
        Compute the error on the expectation values using jackknife resampling.
        Return the approximate number of trajectories needed to reach the
        desired tolerance.
        """
        num_traj = len(self.trajectories)
        if num_traj <= 1:
            return np.inf
        num_e = self.trajectories[0]._e_num

        avg = np.array([np.mean(
            np.stack([traj._expects[i] for traj in self.trajectories]),
            axis=0
        ) for i in range(num_e)])

        std = np.array([np.std(
            np.stack([traj._expects[i] for traj in self.trajectories]),
            axis=0
        ) for i in range(num_e)])

        target = np.array([atol + rtol * mean
                  for mean, (atol, rtol) in zip(avg, self._target_tols)])
        traj_left = np.max(std**2 / target**2 - num_traj + 1)
        self._tol_reached = traj_left < 0
        return traj_left

    @property
    def runs_states(self):
        return [traj.states for traj in self.trajectories]

    @property
    def average_states(self):
        if self.trajectories[0].states[0].isket:
            finals = [state.proj() for state in self.trajectories[0].states]
            for i in range(1, len(self.trajectories)):
                finals = [state.proj() + final for final, state
                          in zip(finals, self.trajectories[i].states)]
        else:
            finals = [state for state in self.trajectories[0].states]
            for i in range(1, len(self.trajectories)):
                finals = [state + final for final, state
                          in zip(finals, self.trajectories[i].states)]
        return [final / len(self.trajectories) for final in finals]

    @property
    def runs_final_states(self):
        return [traj.final_state for traj in self.trajectories]

    @property
    def average_final_state(self):
        if self.trajectories[0].states[0].isket:
            final = sum(traj.final_state.proj() for traj in self.trajectories)
        else:
            final = sum(traj.final_state for traj in self.trajectories)
        return final / len(self.trajectories)

    @property
    def steady_state(self):
        avg = self.average_states
        return sum(avg) / len(avg)

    @property
    def average_expect(self):
        num_e = self.trajectories[0]._e_num
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        result = [np.mean(np.stack([traj._expects[i]
                    for traj in self.trajectories]), axis=0)
               for i in range(num_e)]

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def std_expect(self):
        num_e = self.trajectories[0]._e_num
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        result = [np.std(np.stack([traj._expects[i]
                    for traj in self.trajectories]), axis=0)
               for i in range(num_e)]

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def runs_expect(self):
        num_e = self.trajectories[0]._e_num
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        result = [np.stack([traj._expects[i] for traj in self.trajectories])
               for i in range(num_e)]

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    def expect_traj_avg(self, ntraj=-1):
        num_e = self.trajectories[0]._e_num
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        result = [np.mean(np.stack([traj._expects[i]
                    for traj in self.trajectories[:ntraj]]), axis=0)
               for i in range(num_e)]

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    def expect_traj_std(self, ntraj=-1):
        num_e = self.trajectories[0]._e_num
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        result = [np.std(np.stack([traj._expects[i]
                    for traj in self.trajectories[:ntraj]]), axis=0)
               for i in range(num_e)]

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def collapse(self):
        return [traj.collapse for traj in self.trajectories]

    @property
    def col_times(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[0])
            out.append(col)
        return out

    @property
    def col_which(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[1])
            out.append(col)
        return out

    @property
    def photocurrent(self):
        cols = {}
        tlist = self.trajectories[0].times
        for traj in self.trajectories:
            for t, which in traj.collapse:
                if which in cols:
                    cols[which].append(t)
                else:
                    cols[which] = [t]
        mesurement = []
        for i in range(self.num_c_ops):
            mesurement += [(np.histogram(cols.get(i,[]), tlist)[0]
                          / np.diff(tlist) / len(self.trajectories))]
        return mesurement

    @property
    def run_stats(self):
        return self.trajectories[0].stats

    def __repr__(self):
        out = ""
        out += self.run_stats['solver'] + "\n"
        out += "solver : " + self.stats['method'] + "\n"
        out += "{} runs saved\n".format(self.num_traj)
        out += "number of expect : {}\n".format(self.trajectories[0]._e_num)
        if self.trajectories[0]._store_states:
            out += "Runs states saved\n"
        elif self.trajectories[0]._store_final_state:
            out += "Runs final state saved\n"
        else:
            out += "State not available\n"
        out += "times from {} to {} in {} steps\n".format(self.times[0],
                                                          self.times[-1],
                                                          len(self.times))
        return out

    @property
    def times(self):
        return self.trajectories[0].times

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
        return len(self.trajectories)

    @property
    def num_expect(self):
        return self.trajectories[0].num_expect

    @property
    def num_collapse(self):
        return self.trajectories[0].num_collapse

    @property
    def end_condition(self):
        if self._target_tols is not None and self._tol_reached:
            return "target tolerance reached"
        elif self._target_ntraj == len(self.trajectories):
            return "ntraj reached"
        else:
            return "Max time reached"

class MultiTrajResultAveraged:
    """
    Contain result of simulations with multiple trajectories.
    Only keeps the averages.

    Property
    --------
    average_states : list of Qobj
        Average state for each time. (density matrix)

    average_final_state : Qobj
        Average last state. (density matrix)

    steady_state : Qobj
        Average state of each time and trajectories. (density matrix)

    average_expect : list of list of number
        Averaged expectation values over trajectories.

    std_expect : list of list of number
        Standard derivation of each averaged expectation values.

    expect : list
        list of list of averaged expectation values.

    times : list
        list of the times at which the expectation values and
        states where taken.

    stats :
        Diverse statistics of the evolution.

    num_expect : int
        Number of expectation value operators in simulation.

    num_collapse : int
        Number of collapse operators in simualation.

    num_traj : int/list
        Number of trajectories (for stochastic solvers). A list indicates
        that averaging of expectation values was done over a subset of total
        number of trajectories.

    col_times : list
        Times at which state collpase occurred. Only for Monte Carlo solver.

    col_which : list
        Which collapse operator was responsible for each collapse in
        ``col_times``. Only for Monte Carlo solver.

    collapse : list
        Each collapse per trajectory as a (time, which_oper)

    photocurrent : list
        photocurrent corresponding to each collapse operator.

    """
    def __init__(self, ntraj, e_ops=(), c_ops=(), target_tol=None):
        """
        Parameters:
        -----------
        num_c_ops: int
            Number of collapses operator used in the McSolver
        """
        self.trajectories = None
        self._sum_states = None
        self._sum_last_states = None
        self._sum_expect = None
        self._sum2_expect = None
        self.num_e_ops = len(e_ops)
        self.num_c_ops = len(c_ops)
        self._target_ntraj = ntraj
        self._num = 0
        self._collapse = []
        self.seeds = []
        self.stats = {
            "num_expect": self.num_e_ops,
            "solver": "",
            "method": "",
        }
        self._set_expect_tol(target_tol)


    def add(self, one_traj):
        _to_dm = one_traj.states and one_traj.states[0].isket
        if self._num == 0:
            self.trajectories = one_traj
            if _to_dm and one_traj.states:
                self._sum_states = [state.proj() for state in one_traj.states]
            else:
                self._sum_states = one_traj.states
            if _to_dm and one_traj.final_state:
                self._sum_last_states = one_traj.final_state.proj()
            else:
                self._sum_last_states = one_traj.final_state
            self._sum_expect = [np.array(expect) for expect in one_traj._expects]
            self._sum2_expect = [np.array(expect)**2 for expect in one_traj._expects]
        else:
            if _to_dm:
                if self._sum_states:
                    self._sum_states = [state.proj() + accu for accu, state
                                    in zip(self._sum_states, one_traj.states)]
                if self._sum_last_states:
                    self._sum_last_states += one_traj.final_state.proj()
            else:
                if self._sum_states:
                    self._sum_states = [state + accu for accu, state
                                    in zip(self._sum_states, one_traj.states)]
                if self._sum_last_states:
                    self._sum_last_states += one_traj.final_state
            if self._sum_expect:
                self._sum_expect = [np.array(one) + accu for one, accu in
                                    zip(one_traj._expects, self._sum_expect)]
                self._sum2_expect = [np.array(one)**2 + accu for one, accu in
                                     zip(one_traj._expects, self._sum2_expect)]
        self._collapse.append(one_traj.collapse)
        if hasattr(one_traj, 'seed'):
            self.seeds.append(one_traj.seed)
        self._num += 1
        if self._target_tols is not None:
            return self._check_expect_tol()
        else:
            return np.inf

    def _set_expect_tol(self, target_tol=None):
        """
        Set the capacity to stop the map when the estimated error on the
        expectation values is within given tolerance.

        Error estimation is done with jackknife resampling.

        target_tol : float, list
            If a float, it is read as absolute tolerance.
            If a pair of float: absolute and relative tolerance in that order.
            Lastly, target_tol can be a list of pairs of (atol, rtol) for each
            e_ops.
        """
        self._target_tols = None
        self._tol_reached = False
        if not target_tol:
            return
        if not self.num_e_ops:
            raise ValueError("Cannot target a tolerance without e_ops")

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
        if self._num <= 1:
            return np.inf
        avg = np.array([sum_expect / self._num for sum_expect in self._sum_expect])
        avg2 = np.array([sum_expect / self._num for sum_expect in self._sum2_expect])
        target = np.array([atol + rtol * mean
                  for mean, (atol, rtol) in zip(avg, self._target_tols)])
        traj_left = np.max((avg2 - avg**2) / target**2 - self._num + 1)
        self._tol_reached = traj_left < 0
        return traj_left

    @property
    def runs_states(self):
        return None

    @property
    def average_states(self):
        return [final / self._num for final in self._sum_states]

    @property
    def runs_final_states(self):
        return None

    @property
    def average_final_state(self):
        return self._sum_last_states / self._num

    @property
    def steady_state(self):
        avg = self._sum_states
        return sum(avg) / len(avg)

    @property
    def average_expect(self):
        num_e = self.trajectories._e_num
        _e_ops_dict = self.trajectories._e_ops_dict
        result = [_sum / self._num for _sum in self._sum_expect]

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def std_expect(self):
        num_e = self.trajectories._e_num
        _e_ops_dict = self.trajectories._e_ops_dict
        avg = [_sum / self._num for _sum in self._sum_expect]
        avg2 = [_sum2 / self._num for _sum2 in self._sum2_expect]
        result = [np.sqrt(a2 - a*a) for a, a2 in zip(avg, avg2)]

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def runs_expect(self):
        return None

    def expect_traj_avg(self, ntraj=-1):
        return None

    def expect_traj_std(self, ntraj=-1):
        return None

    @property
    def collapse(self):
        return self._collapse

    @property
    def col_times(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[0])
            out.append(col)
        return out

    @property
    def col_which(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[1])
            out.append(col)
        return out

    @property
    def photocurrent(self):
        cols = {}
        tlist = self.trajectories.times
        for collapses in self.collapse:
            for t, which in collapses:
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
    def run_stats(self):
        return self.trajectories.stats

    def __repr__(self):
        out = ""
        out += self.run_stats['solver'] + "\n"
        out += "solver : " + self.stats['method'] + "\n"
        out += "{} trajectories averaged\n".format(self.num_traj)
        out += "number of expect : {}\n".format(self.trajectories._e_num)
        if self.trajectories._store_states:
            out += "States saved\n"
        elif self.trajectories._store_final_state:
            out += "Final state saved\n"
        else:
            out += "State not available\n"
        out += "times from {} to {} in {} steps\n".format(self.times[0],
                                                          self.times[-1],
                                                          len(self.times))
        return out

    @property
    def times(self):
        return self.trajectories.times

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
        return self.trajectories.num_expect

    @property
    def num_collapse(self):
        return self.trajectories.num_collapse

    @property
    def end_condition(self):
        if self._target_tols is not None and self._tol_reached:
            return "target tolerance reached"
        elif self._target_ntraj == self._num:
            return "ntraj reached"
        else:
            return "Max time reached"
