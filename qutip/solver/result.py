""" Class for solve function results"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

from typing import TypedDict, Any, Callable
import numpy as np
from numpy.typing import ArrayLike
from ..core import Qobj, QobjEvo, expect

__all__ = [
    "Result",
    "TrajectoryResult",
]


class _QobjExpectEop:
    """
    Pickable e_ops callable that calculates the expectation value for a given
    operator.

    Parameters
    ----------
    op : :obj:`.Qobj`
        The expectation value operator.
    """

    def __init__(self, op):
        self.op = op

    def __call__(self, t, state):
        return expect(self.op, state)


class ExpectOp:
    """
    A result e_op (expectation operation).

    Parameters
    ----------
    op : object
        The original object used to define the e_op operation, e.g. a
        :~obj:`Qobj` or a function ``f(t, state)``.

    f : function
        A callable ``f(t, state)`` that will return the value of the e_op
        for the specified state and time.

    append : function
        A callable ``append(value)``, e.g. ``expect[k].append``, that will
        store the result of the e_ops function ``f(t, state)``.

    Attributes
    ----------
    op : object
        The original object used to define the e_op operation.
    """

    def __init__(self, op, f, append):
        self.op = op
        self._f = f
        self._append = append

    def __call__(self, t, state):
        """
        Return the expectation value for the given time, ``t`` and
        state, ``state``.
        """
        return self._f(t, state)

    def _store(self, t, state):
        """
        Store the result of the e_op function. Should only be called by
        :class:`~Result`.
        """
        self._append(self._f(t, state))


class _BaseResult:
    """
    Common method for all ``Result``.
    """

    def __init__(self, options, *, solver=None, stats=None):
        self.solver = solver
        if stats is None:
            stats = {}
        self.stats = stats

        self._state_processors = []
        self._state_processors_require_copy = False

        # make sure not to store a reference to the solver
        options_copy = options.copy()
        if hasattr(options_copy, "_feedback"):
            options_copy._feedback = None
        self.options = options_copy

    def _e_ops_to_dict(self, e_ops):
        """Convert the supplied e_ops to a dictionary of Eop instances."""
        if e_ops is None:
            e_ops = {}
        elif isinstance(e_ops, (list, tuple)):
            e_ops = {k: e_op for k, e_op in enumerate(e_ops)}
        elif isinstance(e_ops, dict):
            pass
        else:
            e_ops = {0: e_ops}
        return e_ops

    def add_processor(self, f, requires_copy=False):
        """
        Append a processor ``f`` to the list of state processors.

        Parameters
        ----------
        f : function, ``f(t, state)``
            A function to be called each time a state is added to this
            result object. The state is the state passed to ``.add``, after
            applying the pre-processors, if any.

        requires_copy : bool, default False
            Whether this processor requires a copy of the state rather than
            a reference. A processor must never modify the supplied state, but
            if a processor stores the state it should set ``require_copy`` to
            true.
        """
        self._state_processors.append(f)
        self._state_processors_require_copy |= requires_copy


class ResultOptions(TypedDict):
    store_states: bool | None
    store_final_state: bool


class Result(_BaseResult):
    """
    Base class for storing solver results.

    Parameters
    ----------
    e_ops : :obj:`.Qobj`, :obj:`.QobjEvo`, function or list or dict of these
        The ``e_ops`` parameter defines the set of values to record at
        each time step ``t``. If an element is a :obj:`.Qobj` or
        :obj:`.QobjEvo` the value recorded is the expectation value of that
        operator given the state at ``t``. If the element is a function, ``f``,
        the value recorded is ``f(t, state)``.

        The values are recorded in the ``e_data`` and ``expect`` attributes of
        this result object. ``e_data`` is a dictionary and ``expect`` is a
        list, where each item contains the values of the corresponding
        ``e_op``.

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

    states : list of :obj:`.Qobj`
        The state at each time ``t`` (if the recording of the state was
        requested).

    final_state : :obj:`.Qobj`:
        The final state (if the recording of the final state was requested).

    expect : list of arrays of expectation values
        A list containing the values of each ``e_op``. The list is in
        the same order in which the ``e_ops`` were supplied and empty if
        no ``e_ops`` were given.

        Each element is itself a list and contains the values of the
        corresponding ``e_op``, with one value for each time in ``.times``.

        The same lists of values may be accessed via the ``.e_data`` dictionary
        and the original ``e_ops`` are available via the ``.e_ops`` attribute.

    e_data : dict
        A dictionary containing the values of each ``e_op``. If the ``e_ops``
        were supplied as a dictionary, the keys are the same as in
        that dictionary. Otherwise the keys are the index of the ``e_op``
        in the ``.expect`` list.

        The lists of expectation values returned are the *same* lists as
        those returned by ``.expect``.

    e_ops : dict
        A dictionary containing the supplied e_ops as ``ExpectOp`` instances.
        The keys of the dictionary are the same as for ``.e_data``.
        Each value is object where ``.e_ops[k](t, state)`` calculates the
        value of ``e_op`` ``k`` at time ``t`` and the given ``state``, and
        ``.e_ops[k].op`` is the original object supplied to create the
        ``e_op``.

    solver : str or None
        The name of the solver generating these results.

    stats : dict or None
        The stats generated by the solver while producing these results.

    options : dict
        The options for this result class.
    """

    times: list[float]
    states: list[Qobj]
    options: ResultOptions
    e_data: dict[Any, list[Any]]

    def __init__(
        self,
        e_ops: dict[Any, Qobj | QobjEvo | Callable[[float, Qobj], Any]],
        options: ResultOptions,
        *,
        solver: str = None,
        stats: dict[str, Any] = None,
        **kw,
    ):
        super().__init__(options, solver=solver, stats=stats)
        raw_ops = self._e_ops_to_dict(e_ops)
        self.e_data = {k: [] for k in raw_ops}
        self.e_ops = {}
        for k, op in raw_ops.items():
            f = self._e_op_func(op)
            self.e_ops[k] = ExpectOp(op, f, self.e_data[k].append)
            self.add_processor(self.e_ops[k]._store)

        self.times = []
        self.states = []
        self._final_state = None

        self._post_init(**kw)

    def _e_op_func(self, e_op):
        """
        Convert an e_op entry into a function, ``f(t, state)`` that returns
        the appropriate value (usually an expectation value).

        Sub-classes may override this function to calculate expectation values
        in different ways.
        """
        if isinstance(e_op, Qobj):
            return _QobjExpectEop(e_op)
        elif isinstance(e_op, QobjEvo):
            return e_op.expect
        elif callable(e_op):
            return e_op
        raise TypeError(f"{e_op!r} has unsupported type {type(e_op)!r}.")

    def _post_init(self):
        """
        Perform post __init__ initialisation. In particular, add state
        processors or pre-processors.

        Sub-class may override this. If the sub-class wishes to register the
        default processors for storing states, it should call this parent
        ``.post_init()`` method.

        Sub-class ``.post_init()`` implementation may take additional keyword
        arguments if required.
        """
        store_states = self.options["store_states"]
        store_states = store_states or (
            len(self.e_ops) == 0 and store_states is None
        )
        if store_states:
            self.add_processor(self._store_state, requires_copy=True)

        store_final_state = self.options["store_final_state"]
        if store_final_state and not store_states:
            self.add_processor(self._store_final_state, requires_copy=True)

    def _store_state(self, t, state):
        """Processor that stores a state in ``.states``."""
        self.states.append(state)

    def _store_final_state(self, t, state):
        """Processor that writes the state to ``._final_state``."""
        self._final_state = state

    def _pre_copy(self, state):
        """Return a copy of the state. Sub-classes may override this to
        copy a state in different manner or to skip making a copy
        altogether if a copy is not necessary.
        """
        return state.copy()

    def add(self, t, state):
        """
        Add a state to the results for the time ``t`` of the evolution.

        Adding a state calculates the expectation value of the state for
        each of the supplied ``e_ops`` and stores the result in ``.expect``.

        The state is recorded in ``.states`` and ``.final_state`` if specified
        by the supplied result options.

        Parameters
        ----------
        t : float
            The time of the added state.

        state : typically a :obj:`.Qobj`
            The state a time ``t``. Usually this is a :obj:`.Qobj` with
            suitable dimensions, but it sub-classes of result might support
            other forms of the state.

        Notes
        -----
        The expectation values, i.e. ``e_ops``, and states are recorded by
        the state processors (see ``.add_processor``).

        Additional processors may be added by sub-classes.
        """
        self.times.append(t)

        if self._state_processors_require_copy:
            state = self._pre_copy(state)

        for op in self._state_processors:
            op(t, state)

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
        lines.append(f"  Number of e_ops: {len(self.e_ops)}")
        if self.states:
            lines.append("  States saved.")
        elif self.final_state is not None:
            lines.append("  Final state saved.")
        else:
            lines.append("  State not saved.")
        lines.append(">")
        return "\n".join(lines)

    @property
    def expect(self) -> list[ArrayLike]:
        return [np.array(e_op) for e_op in self.e_data.values()]

    @property
    def final_state(self) -> Qobj:
        if self._final_state is not None:
            return self._final_state
        if self.states:
            return self.states[-1]
        return None


class TrajectoryResult(Result):
    r"""
    Result class used for single trajectories in multi-trajectory simulations.

    A trajectory may come with a weight. The trajectory average of an
    observable O is then performed as

    .. math::
        \langle O \rangle = \sum_k w(k) O(k) ,

    where O is an observable, w(k) the weight of the k-th trajectory, and O(k)
    the observable on the k-th trajectory. The weight may be time-dependent.

    There may be an absolute weight `wa` and / or a relative weight `wr`.
    The total weight is `w = wa * wr` if the absolute weight is set, and
    `w = wr / N` otherwise (where N is the number of trajectories with no
    absolute weight specified).

    Attributes
    ----------
    rel_weight: float or list
        The relative weight, constant or time-dependent.

    abs_weight: float or list or None
        The absolute weight, constant or time-dependent.
        None if no absolute weight has been set.
    """

    def _post_init(self):
        super()._post_init()

        self.rel_weight = 1
        self.abs_weight = None
        self._time_weight = None
        self._has_weight = False

    def add_absolute_weight(self, new_weight):
        """
        Adds the given weight (which may be either a number or an array of the
        same length as the list of times) as an absolute weight.
        """
        new_weight = new_weight
        if self.abs_weight is None:
            self.abs_weight = new_weight
        else:
            self.abs_weight = self.abs_weight * new_weight
        self._has_weight = True

    def add_relative_weight(self, new_weight):
        """
        Adds the given weight (which may be either a number or an array of the
        same length as the list of times) as a relative weight.
        """
        new_weight = new_weight
        self.rel_weight = self.rel_weight * new_weight
        self._has_weight = True

    def add_time_weight(self, new_weight):
        """
        Adds the given weight (which may be either a number or an array of the
        same length as the list of times) as a relative weight.
        """
        new_weight = np.array(new_weight)
        if len(new_weight) != len(self.times):
            raise RuntimeError("Marginal lenght does not match evolution.")
        if self._time_weight is None:
            self._time_weight = new_weight
        else:
            self._time_weight = self._time_weight * new_weight
        self._has_weight = True

    @property
    def has_weight(self):
        """Whether any weight has been set."""
        return self._has_weight

    @property
    def has_absolute_weight(self):
        """Whether an absolute weight has been set."""
        return (self.abs_weight is not None)

    @property
    def has_time_dependent_weight(self):
        """Whether the total weight is time-dependent."""
        return self._time_weight is not None

    @property
    def total_weight(self):
        """
        Returns the total weight as a single number. If no absolute weight was
        set, this is only the relative weight. If an absolute weight was
        set, this is the product of the absolute and the relative weights.
        """
        if self.has_absolute_weight:
            return self.abs_weight * self.rel_weight
        return self.rel_weight

    @property
    def _scaled_states(self):
        """
        states of the evolution after being scaled by the marginal.
        """
        if not self.has_time_dependent_weight:
            raise RuntimeError(
                "This result does not have time-dependent weight"
            )

        def _to_dm(qobj):
            # TODO: Paul: Can nmmcsolve support open systems?
            if qobj.isket:
                return qobj.proj()
            return qobj

        return [
            _to_dm(state) * w
            for state, w in zip(self.states, self._time_weight)
        ]

    @property
    def _scaled_final_state(self):
        """
        final_state of the evolution after being scaled by the marginal.
        """
        if not self.has_time_dependent_weight:
            raise RuntimeError(
                "This result does not have time-dependent weight"
            )
        return self.final_state * self._time_weight[-1]

    @property
    def _scaled_expect(self):
        """
        expect of the evolution after being scaled by the marginal.
        """
        if not self.has_time_dependent_weight:
            raise RuntimeError(
                "This result does not have time-dependent weight"
            )
        return [e_val * self._time_weight for e_val in self.expect]
