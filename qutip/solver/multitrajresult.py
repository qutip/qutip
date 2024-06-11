"""
This module provides result classes for multi-trajectory solvers.
Note that single trajectories are described by regular `Result` objects from the
`qutip.solver.result` module.
"""

from typing import TypedDict
import numpy as np

from copy import copy

from .result import _BaseResult, TrajectoryResult
from ..core import qzero_like, Qobj

__all__ = [
    "MultiTrajResult",
    "McResult",
    "NmmcResult",
]


class _Acc_Average:
    """
    Single attribute running accumulation.
    Trajectories with relative or absolute weight can be added.

    Accumulated value can be array (dW, trace), Qobj (final_state), list of
    array (expect), list of Qobj (states).

    Partial sums are tracked and the average is computed when needed.
    Optinally, the standard derrivation can also be tracked.

    They can be merged, but once merged, individual trajectories can no longer
    be added.

    """
    _sum_rel = None
    _sum_abs = None
    _sum2_rel = None
    _sum2_abs = None
    _rel_traj = None
    _list = False

    def __init__(self, attribute, std=False):
        self._compute_std = std
        self.attribute = attribute
        self._total_abs_weight = 0

    def init(self, trajectory):
        example = getattr(trajectory, self.attribute)
        if isinstance(example, list):
            self._list = True
        else:
            example = [example]
        if isinstance(example[0], Qobj):
            zero_like = qzero_like
        else:
            zero_like = np.zeros_like

        self._sum_rel = [zero_like(_to_dm(val)) for val in example]
        self._sum_abs = [zero_like(_to_dm(val)) for val in example]
        if self._compute_std:
            self._sum2_rel = [zero_like(_to_dm(val)) for val in example]
            self._sum2_abs = [zero_like(_to_dm(val)) for val in example]
        self._rel_traj = 0

    def add_rel(self, state, weight=1):
        if not self._list:
            state = [state]
        for _sum_rel, val in zip(self._sum_rel, state):
            _sum_rel += _to_dm(val) * weight
        if self._compute_std:
            for _sum2_rel, val in zip(self._sum2_rel, state):
                _sum2_rel += _to_dm(val) * _to_dm(val) * weight
        self._rel_traj += 1

    def add_abs(self, state, weight=None):
        self._total_abs_weight += weight
        if not self._list:
            state = [state]
        for _sum_abs, val in zip(self._sum_abs, state):
            _sum_abs += _to_dm(val) * weight
        if self._compute_std:
            for _sum2_abs, val in zip(self._sum2_abs, state):
                _sum2_abs += _to_dm(val) * _to_dm(val) * weight

    def add(self, trajectory):
        val = getattr(trajectory, self.attribute)
        if trajectory.has_absolute_weight:
            self.add_abs(val, trajectory.total_weight)
        else:
            self.add_rel(val, trajectory.total_weight)

    def _avg(self):
        out = [val.copy() for val in self._sum_abs]
        if self._rel_traj:
            for avg, rel in zip(out, self._sum_rel):
                avg += 1 / self._rel_traj * rel
        #        out += (1 - self._total_abs_weight) / self._rel_traj * rel
        return out

    def average(self):
        out = self._avg()
        if not self._list:
            out = out[0]
        return out

    def _avg2(self):
        out2 = [val.copy() for val in self._sum2_abs]
        if self._sum2_rel:
            for avg, rel in zip(out2, self._sum2_rel):
                avg += 1 / self._rel_traj * rel
        #        out2 += (1 - self._total_abs_weight) / self._rel_traj * rel

        return out2

    def std(self):
        if not self._compute_std:
            return None

        # mean(expect**2) - mean(expect)**2 can something be very small
        # negative (-1e-15) which raise an error for float sqrt.
        std = [
            np.sqrt(np.abs(avg2 - np.abs(avg**2)))
            for avg, avg2 in zip(self._avg(), self._avg2())
        ]

        if not self._list:
            std = std[0]
        return std

    def merge(self, right, w_self):
        """
        Merge 2 _Acc_Average with `w_self` being the weight of this accumulator
        and `1 - w_self` the weight of the other.

        The output is ``frozen``, new trajectories cannot be added to it as it
        loses the rel / abs separations. Merging is still possible.
        """
        out = _Acc_Average(self.attribute, self._compute_std)
        out._list = self._list
        out._sum_abs = [
            avg_left * w_self + avg_right * (1 - w_self)
            for avg_left, avg_right in zip(self._avg(), right._avg())
        ]
        if self._compute_std:
            out._sum2_abs = [
                avg2_left * w_self + avg2_right * (1 - w_self)
                for avg2_left, avg2_right in zip(self._avg2(), right._avg2())
            ]
        out._total_abs_weight = 1.
        return out


class MultiTrajResultOptions(TypedDict):
    store_states: bool
    store_final_state: bool
    keep_runs_results: bool


class MultiTrajResult(_BaseResult):
    """
    Base class for storing results for solver using multiple trajectories.

    Parameters
    ----------
    e_ops : :obj:`.Qobj`, :obj:`.QobjEvo`, function or list or dict of these
        The ``e_ops`` parameter defines the set of values to record at
        each time step ``t``. If an element is a :obj:`.Qobj` or
        :obj:`.QobjEvo` the value recorded is the expectation value of that
        operator given the state at ``t``. If the element is a function, ``f``,
        the value recorded is ``f(t, state)``.

        The values are recorded in the ``.expect`` attribute of this result
        object. ``.expect`` is a list, where each item contains the values
        of the corresponding ``e_op``.

        Function ``e_ops`` must return a number so the average can be computed.

    options : dict
        The options for this result class.

    solver : str or None
        The name of the solver generating these results.

    stats : dict or None
        The stats generated by the solver while producing these results. Note
        that the solver may update the stats directly while producing results.

    kw : dict
        Additional parameters specific to a result sub-class.

    Attributes
    ----------
    times : list
        A list of the times at which the expectation values and states were
        recorded.

    average_states : list of :obj:`.Qobj`
        The state at each time ``t`` (if the recording of the state was
        requested) averaged over all trajectories as a density matrix.

    runs_states : list of list of :obj:`.Qobj`
        The state for each trajectory and each time ``t`` (if the recording of
        the states and trajectories was requested)

    final_state : :obj:`.Qobj`:
        The final state (if the recording of the final state was requested)
        averaged over all trajectories as a density matrix.

    runs_final_state : list of :obj:`.Qobj`
        The final state for each trajectory (if the recording of the final
        state and trajectories was requested).

    average_expect : list of array of expectation values
        A list containing the values of each ``e_op`` averaged over each
        trajectories. The list is in the same order in which the ``e_ops`` were
        supplied and empty if no ``e_ops`` were given.

        Each element is itself an array and contains the values of the
        corresponding ``e_op``, with one value for each time in ``.times``.

    std_expect : list of array of expectation values
        A list containing the standard derivation of each ``e_op`` over each
        trajectories. The list is in the same order in which the ``e_ops`` were
        supplied and empty if no ``e_ops`` were given.

        Each element is itself an array and contains the values of the
        corresponding ``e_op``, with one value for each time in ``.times``.

    runs_expect : list of array of expectation values
        A list containing the values of each ``e_op`` for each trajectories.
        The list is in the same order in which the ``e_ops`` were
        supplied and empty if no ``e_ops`` were given. Only available if the
        storing of trajectories was requested.

        The order of the elements is ``runs_expect[e_ops][trajectory][time]``.

        Each element is itself an array and contains the values of the
        corresponding ``e_op``, with one value for each time in ``.times``.

    average_e_data : dict
        A dictionary containing the values of each ``e_op`` averaged over each
        trajectories. If the ``e_ops`` were supplied as a dictionary, the keys
        are the same as in that dictionary. Otherwise the keys are the index of
        the ``e_op`` in the ``.expect`` list.

        The lists of expectation values returned are the *same* lists as
        those returned by ``.expect``.

    average_e_data : dict
        A dictionary containing the standard derivation of each ``e_op`` over
        each trajectories. If the ``e_ops`` were supplied as a dictionary, the
        keys are the same as in that dictionary. Otherwise the keys are the
        index of the ``e_op`` in the ``.expect`` list.

        The lists of expectation values returned are the *same* lists as
        those returned by ``.expect``.

    runs_e_data : dict
        A dictionary containing the values of each ``e_op`` for each
        trajectories. If the ``e_ops`` were supplied as a dictionary, the keys
        are the same as in that dictionary. Otherwise the keys are the index of
        the ``e_op`` in the ``.expect`` list. Only available if the storing
        of trajectories was requested.

        The order of the elements is ``runs_expect[e_ops][trajectory][time]``.

        The lists of expectation values returned are the *same* lists as
        those returned by ``.expect``.

    runs_weights : list
        For each trajectory, the weight with which that trajectory enters
        averages.

    solver : str or None
        The name of the solver generating these results.

    stats : dict or None
        The stats generated by the solver while producing these results.

    options : :obj:`~SolverResultsOptions`
        The options for this result class.
    """

    options: MultiTrajResultOptions

    def __init__(
        self, e_ops, options: MultiTrajResultOptions, *,
        solver=None, stats=None, **kw,
    ):
        super().__init__(options, solver=solver, stats=stats)
        self._raw_ops = self._e_ops_to_dict(e_ops)

        self.trajectories = []
        self.deterministic_trajectories = []
        self.num_trajectories = 0
        self.seeds = []
        self._final_state_acc = None
        self._states_acc = None
        self._expect_acc = None

        # Will be initialized at the first trajectory
        self.times = None
        self.e_ops = None

        self._trajectories_weight_info = []
        self._deterministic_weight_info = []

        self._post_init(**kw)

    @property
    def _store_average_density_matrices(self) -> bool:
        return (
            self.options["store_states"]
            or (self.options["store_states"] is None and self._raw_ops == {})
        ) and not self.options["keep_runs_results"]

    @property
    def _store_final_density_matrix(self) -> bool:
        return (
            self.options["store_final_state"]
            and not self._store_average_density_matrices
            and not self.options["keep_runs_results"]
        )

    def _add_first_traj(self, trajectory):
        """
        Read the first trajectory, intitializing needed data.
        """
        if self.times is None:
            self.times = trajectory.times
            self.e_ops = trajectory.e_ops
            for acc in self._acc:
                acc.init(trajectory)

    def _store_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def _store_weight_info(self, trajectory):
        self._trajectories_weight_info.append(trajectory.total_weight)

    def _no_end(self):
        """
        Remaining number of trajectories needed to finish cannot be determined
        by this object.
        """
        return np.inf

    def _fixed_end(self):
        """
        Finish at a known number of trajectories.
        """
        ntraj_left = self._target_ntraj - self.num_trajectories
        if ntraj_left == 0:
            self.stats["end_condition"] = "ntraj reached"
        return ntraj_left

    def _average_computer(self):
        avg = np.array(self._expect_acc._sum_rel) / self._expect_acc._rel_traj
        avg2 = np.array(self._expect_acc._sum2_rel) / self._expect_acc._rel_traj
        return avg, avg2

    def _target_tolerance_end(self):
        """
        Compute the error on the expectation values using jackknife resampling.
        Return the approximate number of trajectories needed to have this
        error within the tolerance fot all e_ops and times.
        """
        if self.num_trajectories >= self._target_ntraj:
            # First make sure that "ntraj" setting is always respected
            self.stats["end_condition"] = "ntraj reached"
            return 0

        num_rel_traj =self._expect_acc._rel_traj
        total_abs_weight = self._expect_acc._total_abs_weight

        if num_rel_traj <= 1:
            return np.inf
        avg, avg2 = self._average_computer()
        target = np.array(
            [
                atol + rtol * mean
                for mean, (atol, rtol) in zip(avg, self._target_tols)
            ]
        )

        one = np.array(1)
        if num_rel_traj < self.num_trajectories:
            # We only include traj. without abs. weights in this calculation.
            # Since there are traj. with abs. weights., the weights don't add
            # up to one. We have to consider that as follows:
            #   <(x - <x>)^2> / <1> = <x^2> / <1> - <x>^2 / <1>^2
            # and "<1>" is one minus the sum of all absolute weights
            one = one - total_abs_weight

        target_ntraj = np.max((avg2 / one - (abs(avg) ** 2) / (one ** 2)) /
                              target**2 + 1)

        self._estimated_ntraj = min(target_ntraj - num_rel_traj,
                                    self._target_ntraj - self.num_trajectories)
        if self._estimated_ntraj <= 0:
            self.stats["end_condition"] = "target tolerance reached"
        return self._estimated_ntraj

    def _post_init(self):
        self._target_ntraj = None
        self._target_tols = None
        self._early_finish_check = self._no_end
        self._acc = []

        self.add_processor(self._add_first_traj)
        store_trajectory = self.options["keep_runs_results"]
        if store_trajectory:
            self.add_processor(self._store_trajectory)
        else:
            self.add_processor(self._store_weight_info)
        if self._store_average_density_matrices:
            self._states_acc = _Acc_Average("states")
            self._acc.append(self._states_acc)
            self.add_processor(self._states_acc.add)
        if self._store_final_density_matrix:
            self._final_state_acc = _Acc_Average("final_state")
            self._acc.append(self._final_state_acc)
            self.add_processor(self._final_state_acc.add)
        if self._raw_ops:
            self._expect_acc = _Acc_Average("expect", std=True)
            self._acc.append(self._expect_acc)
            self.add_processor(self._expect_acc.add)

        self.stats["end_condition"] = "unknown"

    def add(self, trajectory_info):
        """
        Add a trajectory to the evolution.

        Trajectories can be saved or average canbe extracted depending on the
        options ``keep_runs_results``.

        Parameters
        ----------
        trajectory_info : tuple of seed and trajectory
            - seed: int, SeedSequence
              Seed used to generate the trajectory.
            - trajectory : :class:`Result`
              Run result for one evolution over the times.

        Returns
        -------
        remaing_traj : number
            Return the number of trajectories still needed to reach the target
            tolerance. If no tolerance is provided, return infinity.
        """
        seed, trajectory = trajectory_info
        self.seeds.append(seed)
        self.num_trajectories += 1

        if not isinstance(trajectory, TrajectoryResult):
            trajectory.has_weight = False
            trajectory.has_absolute_weight = False
            trajectory.has_time_dependent_weight = False
            trajectory.total_weight = 1

        for op in self._state_processors:
            op(trajectory)

        return self._early_finish_check()

    def add_deterministic(self, trajectory):
        for op in self._state_processors:
            op(trajectory)
        if self.options["keep_runs_results"]:
            self.deterministic_trajectories.append(self.trajectories.pop())
        else:
            self._deterministic_weight_info.append(
                self._trajectories_weight_info.pop()
            )

    def add_end_condition(self, ntraj, target_tol=None):
        """
        Set the condition to stop the computing trajectories when the certain
        condition are fullfilled.
        Supported end condition for multi trajectories computation are:

        - Reaching a number of trajectories.
        - Error bar on the expectation values reach smaller than a given
          tolerance.

        Parameters
        ----------
        ntraj : int
            Number of trajectories expected.

        target_tol : float, array_like, [optional]
            Target tolerance of the evolution. The evolution will compute
            trajectories until the error on the expectation values is lower
            than this tolerance. The error is computed using jackknife
            resampling. ``target_tol`` can be an absolute tolerance, a pair of
            absolute and relative tolerance, in that order. Lastly, it can be a
            list of pairs of (atol, rtol) for each e_ops.

            Error estimation is done with jackknife resampling.
        """
        self._target_ntraj = ntraj
        self.stats["end_condition"] = "timeout"

        if target_tol is None:
            self._early_finish_check = self._fixed_end
            return

        num_e_ops = len(self._raw_ops)

        if not num_e_ops:
            raise ValueError("Cannot target a tolerance without e_ops")

        self._estimated_ntraj = ntraj

        targets = np.array(target_tol)
        if targets.ndim == 0:
            self._target_tols = np.array([(target_tol, 0.0)] * num_e_ops)
        elif targets.shape == (2,):
            self._target_tols = np.ones((num_e_ops, 2)) * targets
        elif targets.shape == (num_e_ops, 2):
            self._target_tols = targets
        else:
            raise ValueError(
                "target_tol must be a number, a pair of (atol, "
                "rtol) or a list of (atol, rtol) for each e_ops"
            )

        self._early_finish_check = self._target_tolerance_end

    @property
    def runs_states(self):
        """
        States of every runs as ``states[run][t]``.
        """
        if self.trajectories and self.trajectories[0].states:
            return [traj.states for traj in self.trajectories]
        else:
            return None

    @property
    def average_states(self):
        """
        States averages as density matrices.
        """
        trajectory_states_available = (self.trajectories and
                                       self.trajectories[0].states)

        if not self._states_acc and trajectory_states_available:
            self._states_acc = _Acc_Average("states")
            self._states_acc.init(self.trajectories[0])
            for trajectory in self.deterministic_trajectories:
                self._states_acc.add(trajectory)
            for trajectory in self.trajectories:
                self._states_acc.add(trajectory)

        if self._states_acc:
            return self._states_acc.average()
        return None

    @property
    def states(self):
        """
        Runs final states if available, average otherwise.
        """
        return self.runs_states or self.average_states

    @property
    def runs_final_states(self):
        """
        Last states of each trajectories.
        """
        if self.trajectories and self.trajectories[0].final_state:
            return [traj.final_state for traj in self.trajectories]
        else:
            return None

    @property
    def average_final_state(self):
        """
        Last states of each trajectories averaged into a density matrix.
        """
        trajectory_final_states_available = (self.trajectories and
                                             self.trajectories[0].final_state)
        if (
            not (self._final_state_acc or self._states_acc)
            and trajectory_final_states_available
        ):
            self._final_state_acc = _Acc_Average("final_state")
            self._final_state_acc.init(self.trajectories[0])
            for trajectory in self.deterministic_trajectories:
                self._final_state_acc.add(trajectory)
            for trajectory in self.trajectories:
                self._final_state_acc.add(trajectory)

        if self._final_state_acc:
            return self._final_state_acc.average()
        elif self._states_acc:
            return self._states_acc.average()[-1]
        else:
            return None

    @property
    def final_state(self):
        """
        Runs final states if available, average otherwise.
        """
        return self.runs_final_states or self.average_final_state

    @property
    def average_expect(self):
        if not self._raw_ops:
            return None
        return self._expect_acc.average()

    @property
    def std_expect(self):
        if not self._raw_ops:
            return None
        return self._expect_acc.std()

    @property
    def runs_expect(self):
        if not self._raw_ops or not self.trajectories:
            return None
        return list(zip(*[traj.expect for traj in self.trajectories]))

    @property
    def expect(self):
        return self.runs_expect or self.average_expect

    @property
    def average_e_data(self):
        if not self._raw_ops:
            return None
        return {
            key: values
            for key, values in zip(self.e_ops.keys(), self.average_expect)
        }

    @property
    def std_e_data(self):
        if not self._raw_ops:
            return None
        return {
            key: values
            for key, values in zip(self.e_ops.keys(), self.std_expect)
        }

    @property
    def runs_e_data(self):
        if not self._raw_ops:
            return None
        return {
            key: values
            for key, values in zip(self.e_ops.keys(), self.runs_expect)
        }

    @property
    def e_data(self):
        return self.runs_e_data or self.average_e_data

    @property
    def fixed_weights(self):
        result = []
        if self._deterministic_weight_info:
            result = self._deterministic_weight_info.copy()
        else:
            for traj in self.deterministic_trajectories:
                result.append(traj.total_weight)
        return result

    @property
    def runs_weights(self):
        result = []
        if self._trajectories_weight_info:
            for w in self._trajectories_weight_info:
                result.append(w / self.num_trajectories)
        else:
            for traj in self.trajectories:
                w = traj.total_weight
                result.append(w / self.num_trajectories)
        return result

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

    def __repr__(self):
        lines = [
            f"<{self.__class__.__name__}",
            f"  Solver: {self.solver}",
        ]
        if self.stats:
            lines.append("  Solver stats:")
            lines.extend(f"    {k}: {v!r}" for k, v in self.stats.items())
        if self.times:
            lines.append(
                f"  Time interval: [{self.times[0]}, {self.times[-1]}]"
                f" ({len(self.times)} steps)"
            )
        lines.append(f"  Number of e_ops: {len(self.e_data)}")
        if self.states:
            lines.append("  States saved.")
        elif self.final_state is not None:
            lines.append("  Final state saved.")
        else:
            lines.append("  State not saved.")
        lines.append(f"  Number of trajectories: {self.num_trajectories}")
        if self.trajectories:
            lines.append("  Trajectories saved.")
        else:
            lines.append("  Trajectories not saved.")
        lines.append(">")
        return "\n".join(lines)

    def merge(self, other, p=None):
        r"""
        Merges two multi-trajectory results.

        If this result represent an ensemble :math:`\rho`, and `other`
        represents an ensemble :math:`\rho'`, then the merged result
        represents the ensemble

        .. math::
            \rho_{\mathrm{merge}} = p \rho + (1 - p) \rho'

        where p is a parameter between 0 and 1. Its default value is
        :math:`p_{\textrm{def}} = N / (N + N')`, N and N' being the number of
        trajectories in the two result objects. (In the case of weighted
        trajectories, only trajectories without absolute weights are counted.)

        Parameters
        ----------
        other : MultiTrajResult
            The multi-trajectory result to merge with this one
        p : float [optional]
            The relative weight of this result in the combination. By default,
            will be chosen such that all trajectories contribute equally
            to the merged result.
        """
        if not isinstance(other, MultiTrajResult):
            return NotImplemented
        if self._raw_ops != other._raw_ops:
            raise ValueError("Shared `e_ops` is required to merge results")
        if self.times != other.times:
            raise ValueError("Shared `times` are is required to merge results")
        if self.stats["solver"] != other.stats["solver"]:
            raise ValueError("Can't merge results of different solver")

        new = self.__class__(
            self._raw_ops, self.options, solver=self.solver, stats=self.stats
        )
        new.times = self.times
        new.e_ops = self.e_ops

        # TODO: This change from the description...
        new.num_trajectories = self.num_trajectories + other.num_trajectories
        new.seeds = self.seeds + other.seeds

        p_equal = self.num_trajectories / new.num_trajectories
        if p is None:
            p = p_equal

        if bool(self.trajectories) != bool(other.trajectories):
            # Only one result as trajectories stored
            # Merged will not have them stored, therefore we need to reduce
            # data to merge. Computing the averages once will do it for us.
            if self.trajectories:
                self.average_states
                self.average_final_state
            else:
                other.average_states
                other.average_final_state

        if self.trajectories and other.trajectories:
            new.trajectories, new.deterministic_trajectories = (
                self._merge_trajectories(other, p, p_equal)
            )
        else:
            new.options["keep_runs_results"] = False
            new._trajectories_weight_info, new._deterministic_weight_info = (
                self._merge_weight_info(other, p, new.num_trajectories)
            )

        if self._states_acc and other._states_acc:
            new._states_acc = self._states_acc.merge(other._states_acc, p)
        else:
            new.options["store_states"] = False

        if self._final_state_acc and other._final_state_acc:
            new._final_state_acc = self._final_state_acc.merge(
                other._final_state_acc, p
            )
        else:
            new.options["store_final_state"] = False

        if self._expect_acc and other._expect_acc:
            new._expect_acc = self._expect_acc.merge(other._expect_acc, p)

        new.stats["run time"] += other.stats["run time"]
        new.stats["end_condition"] = "Merged results"
        no_jump_run_time = (
            self.stats.get("no jump run time", 0)
            + other.stats.get("no jump run time", 0)
        )
        if no_jump_run_time:
            new.stats["no jump run time"] = no_jump_run_time
        if other.stats["method"] != new.stats["method"]:
            new.stats["method"] = "various"
        if (
            other.stats.get("num_collapse", 0)
            != new.stats.get("num_collapse", 0)
        ):
            new.stats["num_collapse"] = "various"

        return new

    def _merge_weight_info(self, other, p, ntraj):
        new_traj_weight_info = [
            weight * p * ntraj
            for weight in self.runs_weights
        ] + [
            weight * (1 - p) * ntraj
            for weight in other.runs_weights
        ]

        new_fixed_weight_info = [
            weight * p
            for weight in self.fixed_weights
        ] + [
            weight * (1 - p)
            for weight in other.fixed_weights
        ]

        return new_traj_weight_info, new_fixed_weight_info

    def _merge_trajectories(self, other, p, p_equal):
        if (
            p == p_equal
            and not self.deterministic_trajectories
            and not other.deterministic_trajectories
        ):
            return self.trajectories + other.trajectories, []

        new_trajs = []
        for traj in self.trajectories:
            if (mweight := p / p_equal) != 1:
                traj = copy(traj)
                traj.add_relative_weight(mweight)
            new_trajs.append(traj)
        for traj in other.trajectories:
            if (mweight := (1 - p) / (1 - p_equal)) != 1:
                traj = copy(traj)
                traj.add_relative_weight(mweight)
            new_trajs.append(traj)

        new_d_trajs = []
        for traj in self.deterministic_trajectories:
            traj = copy(traj)
            traj.add_relative_weight(p)
            new_d_trajs.append(traj)
        for traj in other.deterministic_trajectories:
            traj = copy(traj)
            traj.add_relative_weight(1 - p)
            new_d_trajs.append(traj)

        return new_trajs, new_d_trajs

    def __add__(self, other):
        return self.merge(other, p=None)


class McResult(MultiTrajResult):
    """
    Class for storing Monte-Carlo solver results.

    Parameters
    ----------
    e_ops : :obj:`.Qobj`, :obj:`.QobjEvo`, function or list or dict of these
        The ``e_ops`` parameter defines the set of values to record at
        each time step ``t``. If an element is a :obj:`.Qobj` or
        :obj:`.QobjEvo` the value recorded is the expectation value of that
        operator given the state at ``t``. If the element is a function, ``f``,
        the value recorded is ``f(t, state)``.

        The values are recorded in the ``.expect`` attribute of this result
        object. ``.expect`` is a list, where each item contains the values
        of the corresponding ``e_op``.

    options : :obj:`~SolverResultsOptions`
        The options for this result class.

    solver : str or None
        The name of the solver generating these results.

    stats : dict
        The stats generated by the solver while producing these results. Note
        that the solver may update the stats directly while producing results.
        Must include a value for "num_collapse".

    kw : dict
        Additional parameters specific to a result sub-class.

    Attributes
    ----------
    collapse : list
        For each run, a list of every collapse as a tuple of the time it
        happened and the corresponding ``c_ops`` index.
    """

    # Collapse are only produced by mcsolve.
    def _add_collapse(self, trajectory):
        if not trajectory.has_absolute_weight:
            self.collapse.append(trajectory.collapse)

    def _post_init(self):
        super()._post_init()
        self.num_c_ops = self.stats["num_collapse"]
        self._time_dependent_weights = False
        self.collapse = []
        self.add_processor(self._add_collapse)

    @property
    def col_times(self):
        """
        List of the times of the collapses for each runs.
        """
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = [] if len(col) == 0 else col[0]
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
            col = [] if len(col) == 0 else col[1]
            out.append(col)
        return out

    @property
    def photocurrent(self):
        """
        Average photocurrent or measurement of the evolution.
        """
        if self._time_dependent_weights:
            raise NotImplementedError("photocurrent is not implemented "
                                      "for this solver.")

        collapse_times = [[] for _ in range(self.num_c_ops)]
        collapse_weights = [[] for _ in range(self.num_c_ops)]
        tlist = self.times
        for collapses, weight in zip(self.collapse, self.runs_weights):
            for t, which in collapses:
                collapse_times[which].append(t)
                collapse_weights[which].append(weight)

        mesurement = [
            np.histogram(times, bins=tlist, weights=weights)[0]
            / np.diff(tlist)
            for times, weights in zip(collapse_times, collapse_weights)
        ]
        return mesurement

    @property
    def runs_photocurrent(self):
        """
        Photocurrent or measurement of each runs.
        """
        if self._time_dependent_weights:
            raise NotImplementedError("runs_photocurrent is not implemented "
                                      "for this solver.")

        tlist = self.times
        measurements = []
        for collapses in self.collapse:
            collapse_times = [[] for _ in range(self.num_c_ops)]
            for t, which in collapses:
                collapse_times[which].append(t)
            measurements.append(
                [
                    np.histogram(times, tlist)[0] / np.diff(tlist)
                    for times in collapse_times
                ]
            )
        return measurements

    def merge(self, other, p=None):
        new = super().merge(other, p)
        new.collapse = self.collapse + other.collapse
        new._time_dependent_weights = (
            self._time_dependent_weights or other._time_dependent_weights)
        return new


class NmmcResult(McResult):
    """
    Class for storing the results of the non-Markovian Monte-Carlo solver.

    Parameters
    ----------
    e_ops : :obj:`.Qobj`, :obj:`.QobjEvo`, function or list or dict of these
        The ``e_ops`` parameter defines the set of values to record at
        each time step ``t``. If an element is a :obj:`.Qobj` or
        :obj:`.QobjEvo` the value recorded is the expectation value of that
        operator given the state at ``t``. If the element is a function, ``f``,
        the value recorded is ``f(t, state)``.

        The values are recorded in the ``.expect`` attribute of this result
        object. ``.expect`` is a list, where each item contains the values
        of the corresponding ``e_op``.

    options : :obj:`~SolverResultsOptions`
        The options for this result class.

    solver : str or None
        The name of the solver generating these results.

    stats : dict
        The stats generated by the solver while producing these results. Note
        that the solver may update the stats directly while producing results.
        Must include a value for "num_collapse".

    kw : dict
        Additional parameters specific to a result sub-class.

    Attributes
    ----------
    average_trace : list
        The average trace (i.e., averaged over all trajectories) at each time.

    std_trace : list
        The standard deviation of the trace at each time.

    runs_trace : list of lists
        For each recorded trajectory, the trace at each time.
        Only present if ``keep_runs_results`` is set in the options.
    """

    def _post_init(self):
        super()._post_init()
        self._time_dependent_weights = True
        # Use marginal scaled versions
        for acc in self._acc:
            acc.attribute = "_scaled_" + acc.attribute
        self._trace_acc = _Acc_Average("trace")
        self._acc.append(self._trace_acc)
        self.add_processor(self._trace_acc.add)

    def _add_first_traj(self, trajectory):
        super()._add_first_traj(trajectory)

    def _add_trace(self, trajectory):
        self.runs_trace.append(trajectory.trace)

    @property
    def average_trace(self):
        return self._trace_acc.average()

    @property
    def std_trace(self):
        return self._trace_acc.std()

    @property
    def runs_trace(self):
        if self.trajectories:
            return [traj.trace for traj in self.trajectories]
        return None

    @property
    def trace(self):
        """
        Refers to ``average_trace`` or ``runs_trace``, depending on whether
        ``keep_runs_results`` is set in the options.
        """
        return self.runs_trace or self.average_trace

    def merge(self, other, p=None):
        new = super().merge(other, p)

        p_eq = self.num_trajectories / new.num_trajectories
        if p is None:
            p = p_eq

        new._trace_acc = self._trace_acc.merge(other._trace_acc, p)

        return new


def _to_dm(state):
    if isinstance(state, Qobj) and state.type == "ket":
        state = state.proj()
    return state
