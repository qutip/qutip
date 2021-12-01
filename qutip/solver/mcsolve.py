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


def mcsolve(H, psi0, tlist, c_ops=None, e_ops=None, ntraj=1, *,
            args=None, options=None, seeds=None, target_tol=None, timeout=0):
    r"""
    Monte Carlo evolution of a state vector :math:`|\psi \rangle` for a
    given Hamiltonian and sets of collapse operators. Options for the
    underlying ODE solver are given by the Options class.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`, ``list``, callable.
        System Hamiltonian as a Qobj, QobjEvo. It can also be any input type that QobjEvo accepts
        (see :class:`qutip.QobjEvo`'s documentation). ``H`` can also be a superoperator (liouvillian) 
        if some collapse operators are to be treated deterministically.

    psi0 : :class:`qutip.Qobj`
        Initial state vector.

    tlist : array_like
        Times at which results are recorded.

    ntraj : int
        Maximum number of trajectories to run. Can be cut short if a time limit
        is passed in options (per default, mcsolve will stop after 1e8 sec)::
            ``options.mcsolve['map_options']['timeout'] = max_sec``
        Or if the target tolerance is reached, see ``target_tol``.

    c_ops : ``list``
        A ``list`` of collapse operators. They must be operators even if ``H``
        is a superoperator.

    e_ops : ``list``, [optional]
        A ``list`` of operator as Qobj, QobjEvo or callable with signature of
        (t, state: Qobj) for calculating expectation values. When no ``e_ops``
        are given, the solver will default to save the states.

    args : dict, [optional]
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : SolverOptions, [optional]
        Options for the evolution.

    seeds : int, SeedSequence, list, [optional]
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seeds, one for each
        trajectory. Seeds are saved in the result and they can be reused with::
            seeds=prev_result.seeds

    target_tol : float, list, [optional]
        Target tolerance of the evolution. The evolution will compute
        trajectories until the error on the expectation values is lower than
        this tolerance. The error is computed using jackknife resampling.
        ``target_tol`` can be an absolute tolerance or a pair of absolute and
        relative tolerance, in that order. Lastly, it can be a list of pairs of
        (atol, rtol) for each e_ops.

    timeout : float [optional]
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed. Overwrite the option of the same name.

    Returns
    -------
    results : :class:`qutip.solver.Result`
        Object storing all results from the simulation. Which results is saved
        depends on the presence of ``e_ops`` and the options used. ``collapse``
        and ``photocurrent`` is available to Monte Carlo simulation results.
    """
    H = QobjEvo(H, args=args, tlist=tlist)
    c_ops = c_ops if c_ops is not None else []
    if not isinstance(c_ops, (list, tuple)):
        c_ops = [c_ops]
    c_ops = [QobjEvo(c_op, args=args, tlist=tlist) for c_op in c_ops]

    if len(c_ops) == 0:
        return mesolve(H, psi0, tlist, e_ops=e_ops, args=args, options=options)

    if isinstance(ntraj, list):
        if isinstance(options, dict):
            options = SolverOptions(**options)
        options = copy(options) or SolverOptions()
        options.results['keep_runs_results'] = True
        max_ntraj = max(ntraj)
    else:
        max_ntraj = ntraj

    mc = McSolver(H, c_ops, options=options)
    result = mc.run(psi0, tlist=tlist, ntraj=max_ntraj, e_ops=e_ops,
                    seed=seeds, target_tol=target_tol, timeout=timeout)
    if isinstance(ntraj, list):
        result.traj_batch = ntraj
    return result


class _McTrajectorySolver(_TrajectorySolver):
    """
    Solver for one mcsolve trajectory. Created by a :class:`McSolver`.
    """
    name = "mcsolve"
    _avail_integrators = {}

    def __init__(self, parent, *, options=None):
        rhs = parent.rhs
        self._c_ops = parent._c_ops
        self._n_ops = parent._n_ops
        super().__init__(rhs, options=options)

    def _run(self, seed, state0, tlist, e_ops):
        self.collapses = []
        self._set_generator(seed)
        self.target_norm = self.generator.random()
        result = super()._run(self.generator, state0, tlist, e_ops)
        result.collapse = list(self.collapses)
        result.seed = seed
        return result

    def start(self, state0, t0, seed=None, *, safe_ODE=None):
        self.collapses = []
        self._set_generator(seed)
        self.target_norm = self.generator.random()
        super().start(state0, t0, seed=self.generator, safe_ODE=safe_ODE)

    def prob_func(self, state):
        if self.rhs.issuper:
            return _data.norm.trace(unstack_columns(state))
        return _data.norm.l2(state)**2

    def norm_func(self, state):
        if self.rhs.issuper:
            return _data.norm.trace(unstack_columns(state))
        return _data.norm.l2(state)

    def _step(self, t, copy=True):
        t_old, y_old = self._integrator.get_state(copy=False)
        norm_old = self.prob_func(y_old)
        while t_old < t:
            t_step, state = self._integrator.mcstep(t, copy=False)
            norm = self.prob_func(state)
            if norm <= self.target_norm:
                self._do_collapse(norm_old, norm, t_old)
                t_old, y_old = self._integrator.get_state(copy=False)
                norm_old = 1.
            else:
                t_old, y_old = t_step, state
                norm_old = norm

        return t_old, _data.mul(y_old, 1 / self.norm_func(y_old))

    def _do_collapse(self, norm_old, norm, t_prev):
        """Find and apply a collapse."""
        t_final, _ = self._integrator.get_state()
        tries = 0
        #print(self.target_norm)
        #print(self.rhs(0))
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
            norm2_guess = self.prob_func(state)
            #print(t_prev, t_guess, t_final)
            #print(norm_old, norm2_guess, norm)
            #print(state.to_array())
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
            raise Exception("Norm tolerance not reached. " +
                            "Increase accuracy of ODE solver or " +
                            "SolverOptions.mcsolve['norm_steps'].")

        # t_guess, state is at the collapse
        probs = np.zeros(len(self._n_ops))
        for i, n_op in enumerate(self._n_ops):
            probs[i] = n_op.expect_data(t_guess, state).real
        probs = np.cumsum(probs)
        which = np.searchsorted(probs, probs[-1] * self.generator.random())

        state_new = self._c_ops[which].matmul_data(t_guess, state)
        new_norm = self.norm_func(state_new)
        if new_norm < self.options.mcsolve['mc_corr_eps']:
            # This happen when the collapse is caused by numerical error
            state_new = _data.mul(state, 1 / self.norm_func(state))
        else:
            state_new = _data.mul(state_new, 1 / new_norm)
            self.collapses.append((t_guess, which))
            self.target_norm = self.generator.random()
        self._integrator.set_state(t_guess, state_new)

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
    """
    Monte Carlo Solver of a state vector :math:`|\psi \rangle` for a
    given Hamiltonian and sets of collapse operators. Options for the
    underlying ODE solver are given by the Options class.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`, ``list``, callable.
        System Hamiltonian as a Qobj, QobjEvo, can also be a function or list
        that can be made into a Qobjevo. (See :class:`qutip.QobjEvo`'s
        documentation). ``H`` can be a superoperator (liouvillian) if some
        collapse operators are to be treated deterministicly.

    c_ops : ``list``
        A ``list`` of collapse operators. They must be operators even if ``H``
        is a superoperator.

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
            rhs = -1j* QobjEvo(H)
            for n_op in self._n_ops:
                rhs -= 0.5 * n_op

        super().__init__(rhs, self._c_ops, options=options)

        self.stats['solver'] = "MonteCarlo Evolution"
        self.stats['num_collapse'] = len(c_ops)
        self.stats["preparation time"] = time() - _time_start

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new):
        if new is None:
            new = self.optionsclass()
        elif isinstance(new, dict):
            new = self.optionsclass(**new)
        elif not isinstance(new, self.optionsclass):
            raise TypeError("options must be an instance of" +
                            str(self.optionsclass))
        new.results['normalize_output'] = False
        self._options = new

    def _argument(self, args):
        self.rhs.arguments(args)
        for c_op in self._c_ops:
            c_op.arguments(args)
        for n_op in self._n_ops:
            n_op.arguments(args)
