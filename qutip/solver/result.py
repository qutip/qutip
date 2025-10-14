""" Class for solve function results"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations
# Add this import with the other imports at the top

import numpy as np
from typing import TypedDict, Any, Callable
from ..core.numpy_backend import np
from numpy.typing import ArrayLike
from ..core import Qobj, QobjEvo, expect
# For matplotlib check
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

# Fallback cyclic colors function
def _cyclic_colors(n):
    """Cyclic color generator following matplotlib's default color cycle"""
    if matplotlib_available:
        # Use matplotlib's default color cycle
        return plt.rcParams['axes.prop_cycle'].by_key()['color'][:n]
    else:
        # Fallback colors (matplotlib default colors)
        colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf'   # cyan
        ]
        return colors[:min(n, len(colors))]

__all__ = ["Result"]


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
        # Almost all integrators already return a copy that is safe to use.
        self._integrator_return_copy = options.get("method", None) in [
            "adams", "lsoda", "bdf", "dop853", "diag",
            "euler", "platen", "explicit1.5",
            "milstein", "pred_corr", "taylor1.5",
            "milstein_imp", "taylor1.5_imp", "rouchon",
        ]

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

        if (
            self._state_processors_require_copy
            and not self._integrator_return_copy
        ):
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
    def plot_expect(self, *, fig=None, axes=None, show=True, **kwargs):
        """
        Plot expectation values from the result object.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to use for plotting. If not provided, a new figure is created.
        axes : matplotlib.axes.Axes or array of Axes, optional
            Axes to use for plotting. If not provided, new axes are created.
        show : bool, optional
            Whether to show the plot immediately. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        axes : matplotlib.axes.Axes or array of Axes
            The axes containing the plot.
            
        Notes
        -----
        For mcsolve results, additional keyword arguments can be used:
        - trajectories: int or list, number or indices of trajectories to plot
        - average: bool, whether to plot the average of trajectories
        - photocurrent: bool, whether to plot photocurrent instead of expectation values
        """
        if not matplotlib_available:
            raise ImportError("Matplotlib not installed. Please install it to use plotting functions.")
        
        import matplotlib.pyplot as plt
        
        # Extract mcsolve-specific parameters from kwargs
        mcsolve_kwargs = {}
        for key in ['trajectories', 'average', 'photocurrent']:
            if key in kwargs:
                mcsolve_kwargs[key] = kwargs.pop(key)
        
        # Determine what type of result we have
        if hasattr(self, 'trajectories') and getattr(self, 'trajectories', None):
            return self._plot_mcsolve_expect(fig=fig, axes=axes, show=show, **mcsolve_kwargs)
        else:
            return self._plot_standard_expect(fig=fig, axes=axes, show=show, **kwargs)

    def _plot_standard_expect(self, *, fig=None, axes=None, show=True, **kwargs):
        """Plot expectation values for standard solvers."""
        import matplotlib.pyplot as plt
        
        # Get expectation values data
        if hasattr(self, 'expect'):
            data = self.expect
            times = self.times
        else:
            raise ValueError("Result object does not contain expectation values")
        
        # Handle e_ops as dictionary
        labels = []
        if hasattr(self, 'e_ops'):
            if isinstance(self.e_ops, dict):
                labels = list(self.e_ops.keys())
            elif isinstance(self.e_ops, list):
                labels = [f"Expectation {i}" for i in range(len(self.e_ops))]
        else:
            labels = [f"Expectation {i}" for i in range(len(data))]
        
        # Handle figure and axes following QuTiP pattern
        if fig is None and axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        elif axes is None:
            axes = fig.gca()
        
        if not hasattr(axes, '__len__'):
            axes = [axes]
        
        # Use QuTiP's color cycling
        colors = _cyclic_colors(len(data))
        
        # Plot each expectation value
        for i, (label, values) in enumerate(zip(labels, data)):
            if i < len(axes):
                ax = axes[i]
            else:
                ax = axes[0]
            
            # Use the color from our cycle, but allow kwargs to override
            plot_kwargs = kwargs.copy()
            if 'color' not in plot_kwargs:
                plot_kwargs['color'] = colors[i]
            
            ax.plot(times, values, label=label, **plot_kwargs)
            ax.set_xlabel("Time")
            ax.set_ylabel("Expectation value")
            ax.legend()
            ax.grid(True)
        
        # Set title
        solver_name = getattr(self, 'solver', 'Unknown solver')
        if len(axes) == 1:
            axes[0].set_title(f"Expectation values - {solver_name}")
        else:
            fig.suptitle(f"Expectation values - {solver_name}")
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return fig, axes[0] if len(axes) == 1 else axes

    def _plot_mcsolve_expect(self, *, fig=None, axes=None, show=True, 
                            trajectories=None, average=True, photocurrent=False, **kwargs):
        """Plot expectation values for mcsolve results."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if photocurrent:
            # Plot photocurrent data if available
            if not hasattr(self, 'photocurrent'):
                raise ValueError("No photocurrent data available in result")
            
            photocurrent_data = self.photocurrent
            times = self.times
            
            # Handle figure and axes following QuTiP pattern
            if fig is None and axes is None:
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            elif axes is None:
                axes = fig.gca()
            
            if not hasattr(axes, '__len__'):
                axes = [axes]
            
            if isinstance(photocurrent_data, list):
                # Multiple photocurrents
                colors = _cyclic_colors(len(photocurrent_data))
                for i, pc in enumerate(photocurrent_data):
                    if i < len(axes):
                        ax = axes[i]
                    else:
                        ax = axes[0]
                    
                    ax.step(times, pc, where='post', color=colors[i], 
                        label=f"Photocurrent {i}", **kwargs)
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Photocurrent")
                    ax.legend()
                    ax.grid(True)
            else:
                # Single photocurrent
                axes[0].step(times, photocurrent_data, where='post', 
                            label="Photocurrent", **kwargs)
                axes[0].set_xlabel("Time")
                axes[0].set_ylabel("Photocurrent")
                axes[0].legend()
                axes[0].grid(True)
            
            axes[0].set_title("Photocurrent")
            
        else:
            # Plot expectation values
            if not hasattr(self, 'expect'):
                raise ValueError("No expectation values available in result")
            
            expect_data = self.expect
            times = self.times
            
            # Handle trajectories selection
            if trajectories is None:
                trajectories = []
            elif isinstance(trajectories, int):
                trajectories = list(range(min(trajectories, len(expect_data[0]))))
            
            # Determine number of subplots needed
            n_plots = len(expect_data)
            
            # Handle figure and axes following QuTiP pattern
            if fig is None and axes is None:
                if n_plots > 1:
                    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
                else:
                    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            elif axes is None:
                axes = fig.gca()
            
            if n_plots == 1:
                axes = [axes]
            elif not hasattr(axes, '__len__'):
                axes = [axes]
            
            # Handle e_ops as dictionary
            labels = []
            if hasattr(self, 'e_ops'):
                if isinstance(self.e_ops, dict):
                    labels = list(self.e_ops.keys())
                elif isinstance(self.e_ops, list):
                    labels = [f"Expectation {i}" for i in range(len(self.e_ops))]
            else:
                labels = [f"Expectation {i}" for i in range(n_plots)]
            
            # Use QuTiP's color cycling
            colors = _cyclic_colors(len(labels))
            
            # Plot each expectation value type
            for i, (label, exp_values) in enumerate(zip(labels, expect_data)):
                if i < len(axes):
                    ax = axes[i]
                else:
                    ax = axes[0]
                
                # Plot individual trajectories
                traj_colors = _cyclic_colors(len(trajectories))
                for j, traj_idx in enumerate(trajectories):
                    if traj_idx < len(exp_values):
                        # Create copy of kwargs without alpha if it exists
                        traj_kwargs = kwargs.copy()
                        if 'alpha' not in traj_kwargs:
                            traj_kwargs['alpha'] = 0.3
                        
                        ax.plot(times, exp_values[traj_idx], color=traj_colors[j],
                            label=f"Trajectory {traj_idx}" if traj_idx < 5 else None, **traj_kwargs)
                
                # Plot average
                if average and len(exp_values) > 0:
                    avg_values = np.mean(exp_values, axis=0)
                    ax.plot(times, avg_values, 'k-', linewidth=2, label="Average", **kwargs)
                
                ax.set_title(label)
                ax.set_xlabel("Time")
                ax.set_ylabel("Expectation value")
                if trajectories or average:
                    ax.legend()
                ax.grid(True)
            
            # Set main title
            solver_name = getattr(self, 'solver', 'mcsolve')
            if n_plots > 1:
                fig.suptitle(f"Expectation values - {solver_name}")
            else:
                axes[0].set_title(f"Expectation values - {solver_name}")
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return fig, axes[0] if len(axes) == 1 else axes