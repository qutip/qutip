# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['mcsolve', "MCSolver"]

from ..core.numpy_backend import np
from numpy.typing import ArrayLike
from numpy.random import SeedSequence
from time import time
from typing import Any
import warnings

from ..core import QobjEvo, spre, spost, Qobj, unstack_columns, qzero_like
from ..typing import QobjEvoLike, EopsLike
from .multitraj import MultiTrajSolver, _MultiTrajRHS, _InitialConditions
from .solver_base import (
    Solver, Integrator, _solver_deprecation, _kwargs_migration
)
from .multitrajresult import McResult
from .mesolve import mesolve, MESolver
from ._feedback import _QobjFeedback, _DataFeedback, _CollapseFeedback
import qutip.core.data as _data


def mcsolve(
    H: QobjEvoLike,
    state: Qobj,
    tlist: ArrayLike,
    c_ops: QobjEvoLike | list[QobjEvoLike] = (),
    _e_ops = None,
    _ntraj = None,
    *,
    e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
    ntraj: int = 500,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
    seeds: int | SeedSequence | list[int | SeedSequence] = None,
    target_tol: float | tuple[float, float] | list[tuple[float, float]] = None,
    timeout: float = None,
    **kwargs,
) -> McResult:
    r"""
    Monte Carlo evolution of a state vector :math:`|\psi \rangle` for a
    given Hamiltonian and sets of collapse operators. Options for the
    underlying ODE solver are given by the Options class.

    Parameters
    ----------
    H : :class:`.Qobj`, :class:`.QobjEvo`, ``list``, callable.
        System Hamiltonian as a Qobj, QobjEvo. It can also be any input type
        that QobjEvo accepts (see :class:`.QobjEvo`'s documentation).
        ``H`` can also be a superoperator (liouvillian) if some collapse
        operators are to be treated deterministically.

    state : :class:`.Qobj`
        Initial state vector or density matrix.

    tlist : array_like
        Times at which results are recorded.

    c_ops : list
        A ``list`` of collapse operators in any input type that QobjEvo accepts
        (see :class:`.QobjEvo`'s documentation). They must be operators
        even if ``H`` is a superoperator. If none are given, the solver will
        defer to ``sesolve`` or ``mesolve``.

    e_ops : :obj:`.Qobj`, callable, list or dict, optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.

    ntraj : int, default: 500
        Maximum number of trajectories to run. Can be cut short if a time limit
        is passed with the ``timeout`` keyword or if the target tolerance is
        reached, see ``target_tol``.

    args : dict, optional
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : dict, optional
        Dictionary of options for the solver.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          | How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.
        - | progress_kwargs : dict
          | kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - | method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          | Which differential equation integration method to use.
        - | atol, rtol : float
          | Absolute and relative tolerance of the ODE integrator.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
            ``tlist`` step.
        - | max_step : float
          | Maximum length of one internal step. When using pulses, it should
            be less than half the width of the thinnest pulse.
        - | keep_runs_results : bool, [False]
          | Whether to store results from all trajectories or just store the
            averages.
        - | map : str {"serial", "parallel", "loky", "mpi"}
          | How to run the trajectories. "parallel" uses the multiprocessing
            module to run in parallel while "loky" and "mpi" use the "loky" and
            "mpi4py" modules to do so.
        - | num_cpus : int
          | Number of cpus to use when running in parallel. ``None`` detect the
            number of available cpus.
        - | norm_t_tol, norm_tol, norm_steps : float, float, int
          | Parameters used to find the collapse location. ``norm_t_tol`` and
            ``norm_tol`` are the tolerance in time and norm respectively.
            An error will be raised if the collapse could not be found within
            ``norm_steps`` tries.
        - | mc_corr_eps : float
          | Small number used to detect non-physical collapse caused by
            numerical imprecision.
        - | improved_sampling : Bool
          | Whether to use the improved sampling algorithm from Abdelhafez et
            al. PRA (2019)

        Additional options are listed under
        `options <./classes.html#qutip.solver.mcsolve.MCSolver.options>`__.
        More options may be available depending on the selected
        differential equation integration method, see
        `Integrator <./classes.html#classes-ode>`_.

    seeds : int, SeedSequence, list, optional
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seeds, one for each
        trajectory. Seeds are saved in the result and they can be reused with::

            seeds=prev_result.seeds

    target_tol : float, tuple, list, optional
        Target tolerance of the evolution. The evolution will compute
        trajectories until the error on the expectation values is lower than
        this tolerance. The maximum number of trajectories employed is
        given by ``ntraj``. The error is computed using jackknife resampling.
        ``target_tol`` can be an absolute tolerance or a pair of absolute and
        relative tolerance, in that order. Lastly, it can be a list of pairs of
        (atol, rtol) for each e_ops.

    timeout : float, optional
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed.

    Returns
    -------
    results : :class:`.McResult`
        Object storing all results from the simulation. Which results is saved
        depends on the presence of ``e_ops`` and the options used. ``collapse``
        and ``photocurrent`` is available to Monte Carlo simulation results.
        If the initial condition is mixed, the result has additional attributes
        ``initial_states`` and ``ntraj_per_initial_state``.

    Notes
    -----
    The simulation will end when the first end condition is reached between
    ``ntraj``, ``timeout`` and ``target_tol``. If the initial condition is
    mixed, ``target_tol`` is not supported. If the initial condition is mixed,
    and the end condition is not ``ntraj``, the results returned by this
    function should be considered invalid.
    """
    options = _solver_deprecation(kwargs, options, "mc")
    e_ops = _kwargs_migration(_e_ops, e_ops, "e_ops")
    ntraj = _kwargs_migration(_ntraj, ntraj, "ntraj")

    H = QobjEvo(H, args=args, tlist=tlist)
    if not isinstance(c_ops, (list, tuple)):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]

    if len(c_ops) == 0:
        if options is None:
            options = {}
        options = {
            key: options[key]
            for key in options
            if key in MESolver.solver_options
        }
        return mesolve(H, state, tlist, e_ops=e_ops, args=args,
                       options=options)

    if not isinstance(state, Qobj):
        raise TypeError(
            "The initial state for mcsolve must be a Qobj. Use the MCSolver "
            "class for more options of specifying mixed initial states."
        )
    if isinstance(ntraj, (list, tuple)):
        raise TypeError(
            "ntraj must be an integer. "
            "A list of numbers is not longer supported."
        )
    mc = MCSolver(H, c_ops, options=options)

    result = mc.run(state, tlist=tlist, ntraj=ntraj, e_ops=e_ops,
                    seeds=seeds, target_tol=target_tol, timeout=timeout)
    return result


class _MCRHS(_MultiTrajRHS):
    """
    Container for the operators of the solver.
    """

    def __init__(self, H, c_ops, n_ops):
        self.rhs = H
        self.c_ops = c_ops
        self.n_ops = n_ops

    def __call__(self):
        return self.rhs

    def arguments(self, args):
        self.rhs.arguments(args)
        for c_op in self.c_ops:
            c_op.arguments(args)
        for n_op in self.n_ops:
            n_op.arguments(args)

    def _register_feedback(self, key, val):
        self.rhs._register_feedback({key: val}, solver="McSolver")
        for c_op in self.c_ops:
            c_op._register_feedback({key: val}, solver="McSolver")
        for n_op in self.n_ops:
            n_op._register_feedback({key: val}, solver="McSolver")


class MCIntegrator:
    """
    Integrator like object for mcsolve trajectory.
    """
    name = "mcsolve"

    def __init__(self, integrator, system, options=None):
        self._integrator = integrator
        self.system = system
        self._c_ops = system.c_ops
        self._n_ops = system.n_ops
        self.options = options
        self._generator = None
        self.method = f"{self.name} {self._integrator.method}"
        self._is_set = False
        self.issuper = self._c_ops[0].issuper

    def set_state(self, t, state0, generator,
                  no_jump=False, jump_prob_floor=0.0):
        """
        Set the state of the ODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.

        generator : numpy.random.generator
            Random number generator.

        no_jump: Bool
            whether or not to sample the no-jump trajectory.
            If so, the "random number" should be set to zero

        jump_prob_floor: float
            if no_jump == False, this is set to the no-jump
            probability. This setting ensures that we sample
            a trajectory with jumps
        """
        self.collapses = []
        self.system._register_feedback("CollapseFeedback", self.collapses)
        self._generator = generator
        if no_jump:
            self.target_norm = 0.0
        else:
            self.target_norm = (
                self._generator.random() * (1 - jump_prob_floor)
                + jump_prob_floor
            )
        self._integrator.set_state(t, state0)
        self._is_set = True

    def get_state(self, copy=True):
        return *self._integrator.get_state(copy), self._generator

    def integrate(self, t, copy=False):
        t_old, y_old = self._integrator.get_state(copy=False)
        norm_old = self._prob_func(y_old)
        while t_old < t:
            t_step, state = self._integrator.mcstep(t, copy=False)
            norm = self._prob_func(state)
            if norm <= self.target_norm:
                t_col, state = self._find_collapse_time(norm_old, norm,
                                                        t_old, t_step)
                self._do_collapse(t_col, state)
                t_old, y_old = self._integrator.get_state(copy=False)
                norm_old = 1.
            else:
                t_old, y_old = t_step, state
                norm_old = norm

        return t_old, _data.mul(y_old, 1 / self._norm_func(y_old))

    def run(self, tlist):
        for t in tlist[1:]:
            yield self.integrate(t, False)

    def reset(self, hard=False):
        self._integrator.reset(hard)

    def _prob_func(self, state):
        if self.issuper:
            return _data.trace_oper_ket(state).real
        return _data.norm.l2(state)**2

    def _norm_func(self, state):
        if self.issuper:
            return _data.trace_oper_ket(state).real
        return _data.norm.l2(state)

    def _find_collapse_time(self, norm_old, norm, t_prev, t_final):
        """Find the time of the collapse and state just before it."""
        tries = 0
        ratio_cutoff = self.options['norm_min_step']
        while tries < self.options['norm_steps']:
            tries += 1
            if (t_final - t_prev) < self.options['norm_t_tol']:
                t_guess = t_final
                _, state = self._integrator.get_state(copy=False)
                break
            dt = t_final - t_prev
            ratio = (
                np.log(norm_old / self.target_norm)
                / np.log(norm_old / norm)
            )
            # We have a bias to guessing times after the collapse.
            # It can get stuck in slow converging pattern when ratio is close
            # to 1. By forcing a minimum step we can avoid worst cases.
            if ratio < ratio_cutoff:
                ratio = ratio_cutoff
            if ratio > (1 - ratio_cutoff):
                ratio = (1 - ratio_cutoff)
            t_guess = t_prev + dt * ratio

            if (t_guess - t_prev) < self.options['norm_t_tol']:
                t_guess = t_prev + self.options['norm_t_tol']
            _, state = self._integrator.mcstep(t_guess, copy=False)
            norm2_guess = self._prob_func(state)
            if (
                np.abs(self.target_norm - norm2_guess) <
                self.options['norm_tol'] * self.target_norm
            ):
                break
            elif (norm2_guess < self.target_norm):
                # t_guess is still > t_jump
                t_final = t_guess
                norm = norm2_guess
            else:
                # t_guess < t_jump
                t_prev = t_guess
                norm_old = norm2_guess

        if tries >= self.options['norm_steps']:
            raise RuntimeError(
                "Could not find the collapse time within desired tolerance. "
                "Increase accuracy of the ODE solver or lower the tolerance "
                "with the options 'norm_steps', 'norm_tol', 'norm_t_tol'.")

        return t_guess, state

    def _do_collapse(self, collapse_time, state):
        """
        Do the collapse:
        - Find which operator did the collapse.
        - Update the state and Integrator.
        - Next collapse norm location
        - Store collapse info.
        """
        # collapse_time, state is at the collapse
        num_ops = len(self._n_ops)
        if num_ops == 1:
            which = 0
        else:
            probs = [
                n_op.expect_data(collapse_time, state).real
                for n_op in self._n_ops
            ]
            target = sum(probs) * self._generator.random() - probs[0]
            which = 0
            while target > 0 and which <= num_ops:
                which = which + 1
                target -= probs[which]

        state_new = self._c_ops[which].matmul_data(collapse_time, state)
        new_norm = self._norm_func(state_new)
        if new_norm < self.options['mc_corr_eps']:
            # This happen when the collapse is caused by numerical error
            state_new = _data.mul(state, 1 / self._norm_func(state))
        else:
            state_new = _data.mul(state_new, 1 / new_norm)
            self.collapses.append((collapse_time, which))
            # this does not need to be modified for improved sampling:
            # as noted in Abdelhafez PRA (2019),
            # after a jump we reset to the full range [0, 1)
            self.target_norm = self._generator.random()
        self._integrator.set_state(collapse_time, state_new)

    def arguments(self, args):
        if args:
            self._integrator.arguments(args)
            for c_op in self._c_ops:
                c_op.arguments(args)
            for n_op in self._n_ops:
                n_op.arguments(args)

    @property
    def integrator_options(self):
        return self._integrator.integrator_options


# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class MCSolver(MultiTrajSolver):
    r"""
    Monte Carlo Solver of a state vector :math:`|\psi \rangle` for a
    given Hamiltonian and sets of collapse operators. Options for the
    underlying ODE solver are given by the Options class.

    Parameters
    ----------
    H : :class:`.Qobj`, :class:`.QobjEvo`, list, callable.
        System Hamiltonian as a Qobj, QobjEvo. It can also be any input type
        that QobjEvo accepts (see :class:`.QobjEvo`'s documentation).
        ``H`` can also be a superoperator (liouvillian) if some collapse
        operators are to be treated deterministically.

    c_ops : list
        A ``list`` of collapse operators in any input type that QobjEvo accepts
        (see :class:`.QobjEvo`'s documentation). They must be operators
        even if ``H`` is a superoperator.

    options : dict, [optional]
        Options for the evolution.
    """
    name = "mcsolve"
    _resultclass = McResult
    _mc_integrator_class = MCIntegrator
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "map": "serial",
        "mpi_options": {},
        "num_cpus": None,
        "bitgenerator": None,
        "method": "vern7",
        "mc_corr_eps": 1e-10,
        "norm_steps": 25,
        "norm_t_tol": 1e-6,
        "norm_tol": 1e-4,
        "norm_min_step": 0.1,
        "improved_sampling": False,
    }

    def __init__(
        self,
        H: Qobj | QobjEvo,
        c_ops: Qobj | QobjEvo | list[Qobj | QobjEvo],
        *,
        options: dict[str, Any] = None,
    ):
        _time_start = time()

        if isinstance(c_ops, (Qobj, QobjEvo)):
            c_ops = [c_ops]
        c_ops = [QobjEvo(c_op) for c_op in c_ops]

        if H.issuper:
            self._c_ops = [
                spre(c_op) * spost(c_op.dag()) if c_op.isoper else c_op
                for c_op in c_ops
            ]
            self._n_ops = self._c_ops
            rhs = QobjEvo(H)
            for c_op in c_ops:
                cdc = c_op.dag() @ c_op
                rhs -= 0.5 * (spre(cdc) + spost(cdc))
        else:
            self._c_ops = c_ops
            self._n_ops = [c_op.dag() * c_op for c_op in c_ops]
            rhs = -1j * QobjEvo(H)
            for n_op in self._n_ops:
                rhs -= 0.5 * n_op

        self._num_collapse = len(self._c_ops)
        self.options = options

        system = _MCRHS(rhs, self._c_ops, self._n_ops)
        super().__init__(system, options=options)

    def _restore_state(self, data, *, copy=True):
        """
        Retore the Qobj state from its data.
        """
        # Duplicated from the Solver class, but removed the check for the
        # normalize_output option, since MCSolver doesn't have that option.
        if self._state_metadata['dims'] == self.rhs._dims[1]:
            state = Qobj(unstack_columns(data),
                         **self._state_metadata, copy=False)
        else:
            state = Qobj(data, **self._state_metadata, copy=copy)

        return state

    def _initialize_stats(self):
        stats = super()._initialize_stats()
        stats.update({
            "method": self.options["method"],
            "solver": "Master Equation Evolution",
            "num_collapse": self._num_collapse,
        })
        return stats

    def _no_jump_simulation(self, state, tlist, e_ops):
        """
        Simulates the no-jump trajectory from the initial state `state0`.
        Returns a tuple containing the `Result` describing
        this trajectory and the trajectory's probability.
        """
        _, no_jump_result, _ = self._run_one_traj(
            None, state, tlist, e_ops, no_jump=True)
        _, state, _ = self._integrator.get_state(copy=False)
        no_jump_prob = self._integrator._prob_func(state)

        return no_jump_result, no_jump_prob

    def _run_one_traj(self, seed, state, tlist, e_ops, **integrator_kwargs):
        """
        Run one trajectory and return the result.
        """
        jump_prob_floor = integrator_kwargs.get('jump_prob_floor', 0)
        if jump_prob_floor >= 1 - self.options["norm_tol"]:
            # The no-jump probability is one, but we are asked to generate
            # a trajectory with at least one jump.
            # This can happen when a user uses "improved sampling" with a dark
            # initial state, or a mixed initial state containing a dark state.
            # Our best option is to return a trajectory result containing only
            # zeroes. This also ensures that the final multi-trajectory
            # result will contain the requested number of trajectories.
            zero = qzero_like(self._restore_state(state, copy=False))
            result = self._trajectory_resultclass(e_ops, self.options)
            result.collapse = []
            for t in tlist:
                result.add(t, zero)
            return seed, result, 0.

        seed, result, weight = super()._run_one_traj(
            seed, state, tlist, e_ops, **integrator_kwargs
        )

        result.collapse = self._integrator.collapses
        return seed, result, weight * (1 - jump_prob_floor)

    def run(
        self,
        state: Qobj | list[tuple[Qobj, float]],
        tlist: ArrayLike,
        ntraj: int | list[int] = None,
        *,
        args: dict[str, Any] = None,
        e_ops: EopsLike | list[EopsLike] | dict[Any, EopsLike] = None,
        target_tol: float | tuple[float, float] | list[tuple[float, float]] = None,
        timeout: float = None,
        seeds: int | SeedSequence | list[int | SeedSequence] = None,
    ) -> McResult:
        """
        Do the evolution of the Quantum system.

        For a ``state`` at time ``tlist[0]``, do up to ``ntraj`` simulations of
        the  Monte-Carlo evolution. For each time in ``tlist`` store the state
        and/or expectation values in a :class:`.MultiTrajResult`. The evolution
        method and stored results are determined by ``options``.

        Parameters
        ----------
        state : {:obj:`.Qobj`, list of (:obj:`.Qobj`, float)}
            Initial state of the evolution. May be either a pure state or a
            statistical ensemble. An ensemble can be provided either as a
            density matrix, or as a list of tuples. In the latter case, the
            first element of each tuple is a pure state, and the second element
            is its weight, i.e., a number between 0 and 1 describing the
            fraction of the ensemble in that state. The sum of all weights must
            be one.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Time in the list must be in increasing order, but does
            not need to be uniformly distributed.

        ntraj : {int, list of int}
            Number of trajectories to add. If the initial state is pure, this
            must be single number. If the initial state is a mixed ensemble,
            specified as a list of pure states, this parameter may also be a
            list of numbers with the same number of entries. It then specifies
            the number of trajectories for each pure state. If the initial
            state is mixed and this parameter is a single number, it specifies
            the total number of trajectories, which are distributed over the
            initial ensemble automatically.

        args : dict, optional
            Change the ``args`` of the rhs for the evolution.

        e_ops : :obj:`.Qobj`, callable, list or dict, optional
            Single operator, or list or dict of operators, for which to
            evaluate expectation values. Operator can be Qobj, QobjEvo or
            callables with the signature `f(t: float, state: Qobj) -> Any`.

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
        results : :class:`.McResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options. If the initial condition
            is mixed, the result has additional attributes ``initial_states``
            and ``ntraj_per_initial_state``.

        .. note:
            The simulation will end when the first end condition is reached
            between ``ntraj``, ``timeout`` and ``target_tol``. If the initial
            condition is mixed, ``target_tol`` is not supported. If the initial
            condition is mixed, and the end condition is not ``ntraj``, the
            results returned by this function should be considered invalid.
        """
        # We process the arguments and pass on to other functions depending on
        # whether "improved sampling" is turned on, and whether the initial
        # state is mixed.
        if isinstance(state, (list, tuple)):
            is_mixed = True
        else:  # state is Qobj, either pure state or dm
            if isinstance(ntraj, (list, tuple)):
                raise ValueError('The ntraj parameter can only be a list if '
                                 'the initial conditions are mixed and given '
                                 'in the form of a list of pure states')
            is_mixed = state.isoper and not self.rhs.issuper
            if is_mixed:
                # Mixed state given as density matrix. Decompose into list
                # format, i.e., into eigenstates and eigenvalues
                eigenvalues, eigenstates = state.eigenstates()
                state = [(psi, p) for psi, p
                         in zip(eigenstates, eigenvalues) if p > 0]

        if is_mixed and target_tol is not None:
            warnings.warn('Monte Carlo simulations with mixed initial '
                          'state do not support target tolerance')

        # Default value for ntraj: as small as possible
        if ntraj is None:
            if is_mixed:
                ntraj = len(state)
            else:
                ntraj = 1

        if not self.options["improved_sampling"]:
            if is_mixed:
                return super()._run_mixed(
                    state, tlist, ntraj, args=args, e_ops=e_ops,
                    timeout=timeout, seeds=seeds)
            else:
                return super().run(
                    state, tlist, ntraj, args=args, e_ops=e_ops,
                    target_tol=target_tol, timeout=timeout, seeds=seeds)
        if is_mixed:
            return self._run_improved_sampling_mixed(
                state, tlist, ntraj, args=args, e_ops=e_ops,
                timeout=timeout, seeds=seeds)
        return self._run_improved_sampling(
            state, tlist, ntraj, args=args, e_ops=e_ops,
            target_tol=target_tol, timeout=timeout, seeds=seeds)

    def _run_improved_sampling(
            self, state, tlist, ntraj, *,
            args, e_ops, target_tol, timeout, seeds):
        # Sample the no-jump trajectory first. Then, the no-jump probability
        # is used as a lower-bound for random numbers in future MC runs
        seeds, result, map_func, map_kw, state0 = self._initialize_run(
            state, ntraj, args=args, e_ops=e_ops,
            timeout=timeout, target_tol=target_tol, seeds=seeds
        )

        # first run the no-jump trajectory
        start_time = time()
        no_jump_traj, no_jump_prob = (
            self._no_jump_simulation(state0, tlist, e_ops))
        result.add_deterministic(no_jump_traj, no_jump_prob)
        result.stats['no jump run time'] = time() - start_time

        # run the remaining trajectories with the random number floor
        # set to the no jump probability such that we only sample
        # trajectories with jumps
        start_time = time()
        map_func(
            self._run_one_traj, seeds,
            task_args=(state0, tlist, e_ops),
            task_kwargs={'no_jump': False, 'jump_prob_floor': no_jump_prob},
            reduce_func=result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        result.stats['run time'] = time() - start_time
        return result

    def _run_improved_sampling_mixed(
            self, initial_conditions, tlist, ntraj, *,
            args, e_ops, timeout, seeds):
        seeds, result, map_func, map_kw, prepared_ics = self._initialize_run(
            initial_conditions, np.sum(ntraj), args=args, e_ops=e_ops,
            timeout=timeout, seeds=seeds)

        # Run the no-jump trajectories
        start_time = time()
        no_jump_results = map_func(
            self._no_jump_simulation,
            [state for (state, _) in prepared_ics],
            task_kwargs={'tlist': tlist, 'e_ops': e_ops},
            map_kw=map_kw,
        )
        if None in no_jump_results:  # timeout reached
            return result

        # Process results of no-traj runs
        no_jump_probs = []
        for (res, prob), (_, weight) in (zip(no_jump_results, prepared_ics)):
            result.add_deterministic(res, prob * weight)
            no_jump_probs.append(prob)
        result.stats['no jump run time'] = time() - start_time

        # Run the remaining trajectories
        start_time = time()
        ics_info = _InitialConditions(prepared_ics, ntraj)
        arguments = [(id, no_jump_probs[ics_info.get_state_index(id)])
                     for id in range(ics_info.ntraj_total)]
        map_func(
            _unpack_arguments(self._run_one_traj_mixed,
                              ('id', 'jump_prob_floor')),
            arguments,
            task_kwargs={'seeds': seeds, 'ics': ics_info,
                         'tlist': tlist, 'e_ops': e_ops, 'no_jump': False},
            reduce_func=result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        result.stats['run time'] = time() - start_time
        result.initial_states = [self._restore_state(state, copy=False)
                                 for state, _ in ics_info.state_list]
        result.ntraj_per_initial_state = ics_info.ntraj
        return result

    def _get_integrator(self):
        _time_start = time()
        method = self.options["method"]
        if method in self.avail_integrators():
            integrator = self.avail_integrators()[method]
        elif issubclass(method, Integrator):
            integrator = method
        else:
            raise ValueError("Integrator method not supported.")
        integrator_instance = integrator(self.rhs(), self.options)
        mc_integrator = self._mc_integrator_class(
            integrator_instance, self.rhs, self.options
        )
        self._init_integrator_time = time() - _time_start
        return mc_integrator

    @property
    def options(self) -> dict:
        """
        Options for monte carlo solver:

        store_final_state: bool, default: False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: bool, default: None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}, default: "text"
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.

        progress_kwargs: dict, default: {"chunk_size":10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        keep_runs_results: bool, default: False
            Whether to store results from all trajectories or just store the
            averages.

        method: str, default: "adams"
            Which differential equation integration method to use.

        map: str {"serial", "parallel", "loky", "mpi"}, default: "serial"
            How to run the trajectories. "parallel" uses the multiprocessing
            module to run in parallel while "loky" and "mpi" use the "loky" and
            "mpi4py" modules to do so.

        mpi_options: dict, default: {}
            Only applies if map is "mpi". This dictionary will be passed as
            keyword arguments to the `mpi4py.futures.MPIPoolExecutor`
            constructor. Note that the `max_workers` argument is provided
            separately through the `num_cpus` option.

        num_cpus: None, int
            Number of cpus to use when running in parallel. ``None`` detect the
            number of available cpus.

        bitgenerator: {None, "MT19937", "PCG64", "PCG64DXSM", ...}
            Which of numpy.random's bitgenerator to use. With ``None``, your
            numpy version's default is used.

        mc_corr_eps: float, default: 1e-10
            Small number used to detect non-physical collapse caused by
            numerical imprecision.

        norm_t_tol: float, default: 1e-6
            Tolerance in time used when finding the collapse.

        norm_tol: float, default: 1e-4
            Tolerance in norm used when finding the collapse.

        norm_steps: int, default: 25
            Maximum number of tries to find the collapse.

        norm_min_step: float, default: 0.10
            Minimum step used when finding the collapse time, given as a
            fraction of the search interval. Must be between 0 and 0.5.
            A small non-zero value can help avoid the worst cases of
            convergence at the cost of increased average steps required to find
            the collapse.

        improved_sampling: Bool, default: False
            Whether to use the improved sampling algorithm
            of Abdelhafez et al. PRA (2019)
        """
        return self._options

    @options.setter
    def options(self, new_options: dict[str, Any]):
        MultiTrajSolver.options.fset(self, new_options)

    @classmethod
    def avail_integrators(cls):
        if cls is Solver:
            return cls._avail_integrators.copy()
        return {
            **Solver.avail_integrators(),
            **cls._avail_integrators,
        }

    @classmethod
    def CollapseFeedback(cls, default: list = None):
        """
        Collapse of the trajectory argument for time dependent systems.

        When used as an args:

            ``QobjEvo([op, func], args={"cols": MCSolver.CollapseFeedback()})``

        The ``func`` will receive a list of ``(time, operator number)`` for
        each collapses of the trajectory as ``cols``.

        .. note::

            CollapseFeedback can't be added to a running solver when updating
            arguments between steps: ``solver.step(..., args={})``.

        Parameters
        ----------
        default : list, default : []
            Argument value to use outside of solver.

        """
        return _CollapseFeedback(default)

    @classmethod
    def StateFeedback(
        cls,
        default: Qobj | _data.Data = None,
        raw_data: bool = False,
        prop: bool = False
    ):
        """
        State of the evolution to be used in a time-dependent operator.

        When used as an args:

            ``QobjEvo([op, func], args={"state": MCSolver.StateFeedback()})``

        The ``func`` will receive the density matrix as ``state`` during the
        evolution.

        Parameters
        ----------
        default : Qobj or qutip.core.data.Data, default : None
            Initial value to be used at setup of the system.

        open : bool, default False
            Set to ``True`` when using the monte carlo solver for open systems.

        raw_data : bool, default : False
            If True, the raw matrix will be passed instead of a Qobj.
            For density matrices, the matrices can be column stacked or square
            depending on the integration method.
        """
        if raw_data:
            return _DataFeedback(default, open=open)
        return _QobjFeedback(default, open=open)


class _unpack_arguments:
    """
    If `f = _unpack_arguments(func, ('a', 'b'))`
    then calling `f((3, 4), ...)` is equivalent to `func(a=3, b=4, ...)`.

    Useful since the map functions in `qutip.parallel` only allow one
    of the parameters of the task to be variable.
    """
    def __init__(self, func, argument_names):
        self.func = func
        self.argument_names = argument_names

    def __call__(self, args, **kwargs):
        rearranged = dict(zip(self.argument_names, args))
        return self.func(**rearranged, **kwargs)
