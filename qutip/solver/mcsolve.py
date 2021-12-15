__all__ = ['mcsolve', "McSolver"]

import warnings

import numpy as np
from copy import copy
from ..core import QobjEvo, spre, spost, Qobj, unstack_columns, liouvillian
from .options import SolverOptions
from .multitraj import MultiTrajSolver, _TrajectorySolver
from .mesolve import mesolve
import qutip.core.data as _data
from time import time


def mcsolve(H, state, tlist, c_ops=(), e_ops=None, ntraj=1, *,
            args=None, options=None, seeds=None, target_tol=None, timeout=1e8):
    r"""
    Monte Carlo evolution of a state vector :math:`|\psi \rangle` for a
    given Hamiltonian and sets of collapse operators. Options for the
    underlying ODE solver are given by the Options class.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`, ``list``, callable.
        System Hamiltonian as a Qobj, QobjEvo. It can also be any input type
        that QobjEvo accepts (see :class:`qutip.QobjEvo`'s documentation).
        ``H`` can also be a superoperator (liouvillian) if some collapse
        operators are to be treated deterministically.

    state : :class:`qutip.Qobj`
        Initial state vector.

    tlist : array_like
        Times at which results are recorded.

    c_ops : list
        A ``list`` of collapse operators in any input type that QobjEvo accepts
        (see :class:`qutip.QobjEvo`'s documentation). They must be operators
        even if ``H`` is a superoperator. If none are given, the solver will
        defer to ``sesolve`` or ``mesolve``.

    e_ops : list, [optional]
        A ``list`` of operator as Qobj, QobjEvo or callable with signature of
        (t, state: Qobj) for calculating expectation values. When no ``e_ops``
        are given, the solver will default to save the states.

    ntraj : int
        Maximum number of trajectories to run. Can be cut short if a time limit
        is passed with the ``timeout`` keyword or if the target tolerance is
        reached, see ``target_tol``.

    args : dict, [optional]
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : SolverOptions, [optional]
        Options for the evolution.

    seeds : int, SeedSequence, list, [optional]
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seeds, one for each
        trajectory. Seeds are saved in the result and they can be reused with::
            seeds=prev_result.seeds

    target_tol : float, list, [optional] {None}
        Target tolerance of the evolution. The evolution will compute
        trajectories until the error on the expectation values is lower than
        this tolerance. The error is computed using jackknife resampling.
        ``target_tol`` can be an absolute tolerance or a pair of absolute and
        relative tolerance, in that order. Lastly, it can be a list of pairs of
        (atol, rtol) for each e_ops.

    timeout : float [optional] {1e8}
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed. Overwrite the option of the same name.

    Returns
    -------
    results : :class:`qutip.solver.Result`
        Object storing all results from the simulation. Which results is saved
        depends on the presence of ``e_ops`` and the options used. ``collapse``
        and ``photocurrent`` is available to Monte Carlo simulation results.

    .. note:
        The simulation will end when the first end condition is reached between
        ``ntraj``, ``timeout`` and ``target_tol``.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    if not isinstance(c_ops, (list, tuple)):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]

    if len(c_ops) == 0:
        return mesolve(H, state, tlist, e_ops=e_ops, args=args,
                       options=options)

    if isinstance(ntraj, list):
        raise TypeError("No longer supported, use `result.expect_traj_avg`"
                        "with the options `keep_runs_results=True`.")

    mc = McSolver(H, c_ops, options=options)
    result = mc.run(state, tlist=tlist, ntraj=ntraj, e_ops=e_ops,
                    seed=seeds, target_tol=target_tol, timeout=timeout)
    return result


class _McTrajectorySolver(_TrajectorySolver):
    """
    Solver for one mcsolve trajectory. Created by a :class:`McSolver`.
    """
    name = "mcsolve"
    _avail_integrators = {}

    def __init__(self, rhs, c_ops, n_ops, *, options=None, _prepare=False):
        self._c_ops = c_ops
        self._n_ops = n_ops
        super().__init__(rhs, options=options, _prepare=_prepare)

    def _run(self, seed, state, tlist, e_ops):
        self.collapses = []
        self.generator = self.get_generator(seed)
        self.target_norm = self.generator.random()
        result = super()._run(self.generator, state, tlist, e_ops)
        result.collapse = list(self.collapses)
        result.seed = seed
        return result

    def start(self, state, t0, seed=None):
        super().start(state, t0, seed=seed)
        self.target_norm = self.generator.random()
        self.collapses = []

    def _prob_func(self, state):
        if self.rhs.issuper:
            return _data.norm.trace(unstack_columns(state))
        return _data.norm.l2(state)**2

    def _norm_func(self, state):
        if self.rhs.issuper:
            return _data.norm.trace(unstack_columns(state))
        return _data.norm.l2(state)

    def _step(self, t, copy=True):
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

    def _find_collapse_time(self, norm_old, norm, t_prev, t_final):
        """Find and apply a collapse."""
        tries = 0
        while tries < self.options.mcsolve['norm_steps']:
            tries += 1
            if (t_final - t_prev) < self.options.mcsolve['norm_t_tol']:
                t_guess = t_final
                _, state = self._integrator.get_state()
                break
            t_guess = (
                t_prev
                + ((t_final - t_prev)
                   * np.log(norm_old / self.target_norm)
                   / np.log(norm_old / norm))
            )
            if (t_guess - t_prev) < self.options.mcsolve['norm_t_tol']:
                t_guess = t_prev + self.options.mcsolve['norm_t_tol']
            _, state = self._integrator.mcstep(t_guess, copy=False)
            norm2_guess = self._prob_func(state)
            if (
                np.abs(self.target_norm - norm2_guess) <
                self.options.mcsolve['norm_tol'] * self.target_norm
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

        if tries >= self.options.mcsolve['norm_steps']:
            raise RuntimeError(
                "Could not find the collapse time within desired tolerance. "
                "Increase accuracy of the ODE solver or lower the tolerance "
                "with the options 'norm_steps', 'norm_tol', 'norm_t_tol'.")

        return t_guess, state

    def _do_collapse(self, collapse_time, state):
        # collapse_time, state is at the collapse
        if len(self._n_ops) == 1:
            which = 0
        else:
            probs = np.zeros(len(self._n_ops))
            for i, n_op in enumerate(self._n_ops):
                probs[i] = n_op.expect_data(collapse_time, state).real
            probs = np.cumsum(probs)
            which = np.searchsorted(probs, probs[-1] * self.generator.random())

        state_new = self._c_ops[which].matmul_data(collapse_time, state)
        new_norm = self._norm_func(state_new)
        if new_norm < self.options.mcsolve['mc_corr_eps']:
            # This happen when the collapse is caused by numerical error
            state_new = _data.mul(state, 1 / self._norm_func(state))
        else:
            state_new = _data.mul(state_new, 1 / new_norm)
            self.collapses.append((collapse_time, which))
            self.target_norm = self.generator.random()
        self._integrator.set_state(collapse_time, state_new)

    def _argument(self, args):
        if self._integrator:
            self._integrator.arguments(args)
        self.rhs.arguments(args)
        for c_op in self._c_ops:
            c_op.arguments(args)
        for n_op in self._n_ops:
            n_op.arguments(args)


# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class McSolver(MultiTrajSolver):
    r"""
    Monte Carlo Solver of a state vector :math:`|\psi \rangle` for a
    given Hamiltonian and sets of collapse operators. Options for the
    underlying ODE solver are given by the Options class.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`, ``list``, callable.
        System Hamiltonian as a Qobj, QobjEvo. It can also be any input type
        that QobjEvo accepts (see :class:`qutip.QobjEvo`'s documentation).
        ``H`` can also be a superoperator (liouvillian) if some collapse
        operators are to be treated deterministically.

    c_ops : list
        A ``list`` of collapse operators in any input type that QobjEvo accepts
        (see :class:`qutip.QobjEvo`'s documentation). They must be operators
        even if ``H`` is a superoperator.

    options : SolverOptions, [optional]
        Options for the evolution.

    seed : int, SeedSequence, list, [optional]
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seed, one for each
        trajectory. Seeds are saved in the result and can be reused with::
            seeds=prev_result.seeds
    """
    _traj_solver_class = _McTrajectorySolver
    name = "mcsolve"
    optionsclass = SolverOptions

    def __init__(self, H, c_ops, *, options=None, seed=None):
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

        super().__init__(rhs, options=options)

        self.stats['solver'] = "MonteCarlo Evolution"
        self.stats['num_collapse'] = len(c_ops)
        self.stats["preparation time"] = time() - _time_start

    @MultiTrajSolver.options.setter
    def options(self, new):
        super(McSolver, self.__class__).options.fset(self, new)
        self.options.results['normalize_output'] = False

    def _argument(self, args):
        self.rhs.arguments(args)
        for c_op in self._c_ops:
            c_op.arguments(args)
        for n_op in self._n_ops:
            n_op.arguments(args)

    @property
    def traj_args(self):
        return (self.rhs, self._c_ops, self._n_ops)

    def run(self, state, tlist, ntraj=1, *, args=None, options=None,
            e_ops=(), timeout=1e8, target_tol=None, seed=None):
        result = super().run(
            state, tlist, ntraj, e_ops=e_ops,
            args=args, options=options, seed=seed,
            timeout=timeout, target_tol=target_tol,
        )
        result.num_c_ops = len(self._c_ops)
        return result
