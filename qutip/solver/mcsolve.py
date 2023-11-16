__all__ = ['mcsolve', "MCSolver"]

import numpy as np
from ..core import QobjEvo, spre, spost, Qobj, unstack_columns
from .multitraj import MultiTrajSolver
from .solver_base import Solver, Integrator, _solver_deprecation
from .result import McResult, McTrajectoryResult, McResultImprovedSampling
from .mesolve import mesolve, MESolver
import qutip.core.data as _data
from time import time


def mcsolve(H, state, tlist, c_ops=(), e_ops=None, ntraj=500, *,
            args=None, options=None, seeds=None, target_tol=None, timeout=None,
            **kwargs):
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

    args : None / dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : None / dict
        Dictionary of options for the solver.

        - store_final_state : bool, [False]
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, NoneType, [None]
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}, ['text']
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict, [{"chunk_size": 10}]
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - method : str {"adams", "bdf", "dop853", "vern9", etc.}, ["adams"]
          Which differential equation integration method to use.
        - keep_runs_results : bool, [False]
          Whether to store results from all trajectories or just store the
          averages.
        - map : str {"serial", "parallel", "loky"}, ["serial"]
          How to run the trajectories. "parallel" uses concurent module to run
          in parallel while "loky" use the module of the same name to do so.
        - job_timeout : NoneType, int, [None]
          Maximum time to compute one trajectory.
        - num_cpus : NoneType, int, [None]
          Number of cpus to use when running in parallel. ``None`` detect the
          number of available cpus.
        - norm_t_tol, norm_tol, norm_steps : float, float, int, [1e-6, 1e-4, 5]
          Parameters used to find the collapse location. ``norm_t_tol`` and
          ``norm_tol`` are the tolerance in time and norm respectively.
          An error will be raised if the collapse could not be found within
          ``norm_steps`` tries.
        - mc_corr_eps : float, [1e-10]
          Small number used to detect non-physical collapse caused by numerical
          imprecision.
        - atol, rtol : float, [1e-8, 1e-6]
          Absolute and relative tolerance of the ODE integrator.
        - nsteps : int [2500]
          Maximum number of (internally defined) steps allowed in one ``tlist``
          step.
        - max_step : float, [0]
          Maximum lenght of one internal step. When using pulses, it should be
          less than half the width of the thinnest pulse.
        - improved_sampling : Bool
          Whether to use the improved sampling algorithm from Abdelhafez et al.
          PRA (2019)

    seeds : int, SeedSequence, list, [optional]
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seeds, one for each
        trajectory. Seeds are saved in the result and they can be reused with::

            seeds=prev_result.seeds

    target_tol : float, tuple, list, [optional]
        Target tolerance of the evolution. The evolution will compute
        trajectories until the error on the expectation values is lower than
        this tolerance. The maximum number of trajectories employed is
        given by ``ntraj``. The error is computed using jackknife resampling.
        ``target_tol`` can be an absolute tolerance or a pair of absolute and
        relative tolerance, in that order. Lastly, it can be a list of pairs of
        (atol, rtol) for each e_ops.

    timeout : float, [optional]
        Maximum time for the evolution in second. When reached, no more
        trajectories will be computed.

    Returns
    -------
    results : :class:`qutip.solver.McResult`
        Object storing all results from the simulation. Which results is saved
        depends on the presence of ``e_ops`` and the options used. ``collapse``
        and ``photocurrent`` is available to Monte Carlo simulation results.

    .. note:
        The simulation will end when the first end condition is reached between
        ``ntraj``, ``timeout`` and ``target_tol``.
    """
    options = _solver_deprecation(kwargs, options, "mc")
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

    if isinstance(ntraj, (list, tuple)):
        raise TypeError(
            "ntraj must be an integer. "
            "A list of numbers is not longer supported."
        )
    mc = MCSolver(H, c_ops, options=options)

    result = mc.run(state, tlist=tlist, ntraj=ntraj, e_ops=e_ops,
                    seed=seeds, target_tol=target_tol, timeout=timeout)
    return result


class MCIntegrator:
    """
    Integrator like object for mcsolve trajectory.
    """
    name = "mcsolve"

    def __init__(self, integrator, c_ops, n_ops, options=None):
        self._integrator = integrator
        self._c_ops = c_ops
        self._n_ops = n_ops
        self.options = options
        self._generator = None
        self.method = f"{self.name} {self._integrator.method}"
        self._is_set = False
        self.issuper = c_ops[0].issuper

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
        while tries < self.options['norm_steps']:
            tries += 1
            if (t_final - t_prev) < self.options['norm_t_tol']:
                t_guess = t_final
                _, state = self._integrator.get_state()
                break
            t_guess = (
                t_prev
                + ((t_final - t_prev)
                   * np.log(norm_old / self.target_norm)
                   / np.log(norm_old / norm))
            )
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
        if len(self._n_ops) == 1:
            which = 0
        else:
            probs = np.zeros(len(self._n_ops))
            for i, n_op in enumerate(self._n_ops):
                probs[i] = n_op.expect_data(collapse_time, state).real
            probs = np.cumsum(probs)
            which = np.searchsorted(probs,
                                    probs[-1] * self._generator.random())

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
    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`, ``list``, callable.
        System Hamiltonian as a Qobj, QobjEvo. It can also be any input type
        that QobjEvo accepts (see :class:`qutip.QobjEvo`'s documentation).
        ``H`` can also be a superoperator (liouvillian) if some collapse
        operators are to be treated deterministically.

    c_ops : list
        A ``list`` of collapse operators in any input type that QobjEvo accepts
        (see :class:`qutip.QobjEvo`'s documentation). They must be operators
        even if ``H`` is a superoperator.

    options : dict, [optional]
        Options for the evolution.
    """
    name = "mcsolve"
    trajectory_resultclass = McTrajectoryResult
    mc_integrator_class = MCIntegrator
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "keep_runs_results": False,
        "method": "adams",
        "map": "serial",
        "job_timeout": None,
        "num_cpus": None,
        "bitgenerator": None,
        "mc_corr_eps": 1e-10,
        "norm_steps": 5,
        "norm_t_tol": 1e-6,
        "norm_tol": 1e-4,
        "improved_sampling": False,
    }

    def __init__(self, H, c_ops, *, options=None):
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

        super().__init__(rhs, options=options)

    def _restore_state(self, data, *, copy=True):
        """
        Retore the Qobj state from its data.
        """
        if self._state_metadata['dims'] == self.rhs.dims[1]:
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

    def _argument(self, args):
        self._integrator.arguments(args)
        self.rhs.arguments(args)
        for c_op in self._c_ops:
            c_op.arguments(args)
        for n_op in self._n_ops:
            n_op.arguments(args)

    def _initialize_run_one_traj(self, seed, state, tlist, e_ops,
                                 no_jump=False, jump_prob_floor=0.0):
        result = self.trajectory_resultclass(e_ops, self.options)
        generator = self._get_generator(seed)
        self._integrator.set_state(tlist[0], state, generator,
                                   no_jump=no_jump,
                                   jump_prob_floor=jump_prob_floor)
        result.add(tlist[0], self._restore_state(state, copy=False))
        return result

    def _run_one_traj(self, seed, state, tlist, e_ops, no_jump=False,
                      jump_prob_floor=0.0):
        """
        Run one trajectory and return the result.
        """
        result = self._initialize_run_one_traj(seed, state, tlist, e_ops,
                                               no_jump=no_jump,
                                               jump_prob_floor=jump_prob_floor)
        seed, result = self._integrate_one_traj(seed, tlist, result)
        result.collapse = self._integrator.collapses
        return seed, result

    def run(self, state, tlist, ntraj=1, *,
            args=None, e_ops=(), timeout=None, target_tol=None, seed=None):
        """
        Do the evolution of the Quantum system.
        See the overridden method for further details. The modification
        here is to sample the no-jump trajectory first. Then, the no-jump
        probability is used as a lower-bound for random numbers in future
        monte carlo runs
        """
        if not self.options.get("improved_sampling", False):
            return super().run(state, tlist, ntraj=ntraj, args=args,
                               e_ops=e_ops, timeout=timeout,
                               target_tol=target_tol, seed=seed)
        stats, seeds, result, map_func, map_kw, state0 = self._initialize_run(
            state,
            ntraj,
            args=args,
            e_ops=e_ops,
            timeout=timeout,
            target_tol=target_tol,
            seed=seed,
        )
        # first run the no-jump trajectory
        start_time = time()
        seed0, no_jump_result = self._run_one_traj(seeds[0], state0, tlist,
                                                   e_ops, no_jump=True)
        _, state, _ = self._integrator.get_state(copy=False)
        no_jump_prob = self._integrator._prob_func(state)
        result.no_jump_prob = no_jump_prob
        result.add((seed0, no_jump_result))
        result.stats['no jump run time'] = time() - start_time

        # run the remaining trajectories with the random number floor
        # set to the no jump probability such that we only sample
        # trajectories with jumps
        start_time = time()
        map_func(
            self._run_one_traj, seeds[1:],
            (state0, tlist, e_ops, False, no_jump_prob),
            reduce_func=result.add, map_kw=map_kw,
            progress_bar=self.options["progress_bar"],
            progress_bar_kwargs=self.options["progress_kwargs"]
        )
        result.stats['run time'] = time() - start_time
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
        integrator_instance = integrator(self.rhs, self.options)
        mc_integrator = self.mc_integrator_class(
            integrator_instance, self._c_ops, self._n_ops, self.options
        )
        self._init_integrator_time = time() - _time_start
        return mc_integrator

    @property
    def resultclass(self):
        if self.options.get("improved_sampling", False):
            return McResultImprovedSampling
        else:
            return McResult

    @property
    def options(self):
        """
        Options for monte carlo solver:

        store_final_state: bool, default=False
            Whether or not to store the final state of the evolution in the
            result class.

        store_states: bool, default=None
            Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.

        progress_bar: str {'text', 'enhanced', 'tqdm', ''}, default="text"
            How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error if
            not installed. Empty string or False will disable the bar.

        progress_kwargs: dict, default={"chunk_size":10}
            Arguments to pass to the progress_bar. Qutip's bars use
            ``chunk_size``.

        keep_runs_results: bool
          Whether to store results from all trajectories or just store the
          averages.

        method: str, default="adams"
            Which ODE integrator methods are supported.

        map: str {"serial", "parallel", "loky"}
            How to run the trajectories. "parallel" uses concurent module to
            run in parallel while "loky" use the module of the same name to do
            so.

        job_timeout: None, int
            Maximum time to compute one trajectory.

        num_cpus: None, int
            Number of cpus to use when running in parallel. ``None`` detect the
            number of available cpus.

        bitgenerator: {None, "MT19937", "PCG64", "PCG64DXSM", ...}
            Which of numpy.random's bitgenerator to use. With ``None``, your
            numpy version's default is used.

        mc_corr_eps: float
            Small number used to detect non-physical collapse caused by
            numerical imprecision.

        norm_t_tol: float
            Tolerance in time used when finding the collapse.

        norm_tol: float
            Tolerance in norm used when finding the collapse.

        norm_steps: int
            Maximum number of tries to find the collapse.

        improved_sampling: Bool
            Whether to use the improved sampling algorithm
            of Abdelhafez et al. PRA (2019)
        """
        return self._options

    @options.setter
    def options(self, new_options):
        MultiTrajSolver.options.fset(self, new_options)

    @classmethod
    def avail_integrators(cls):
        if cls is Solver:
            return cls._avail_integrators.copy()
        return {
            **Solver.avail_integrators(),
            **cls._avail_integrators,
        }
