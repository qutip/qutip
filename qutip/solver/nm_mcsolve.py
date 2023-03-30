__all__ = ['nm_mcsolve', 'NonMarkovianMCSolver']

import functools
import numbers

import numpy as np
import scipy

from .multitraj import MultiTrajSolver
from .mcsolve import MCSolver
from .mesolve import MESolver, mesolve
from .result import NmmcResult, NmmcTrajectoryResult
from .cy.nm_mcsolve import RateShiftCoefficient
from ..core import (
    CoreOptions, Qobj, QobjEvo, isket, ket2dm, qeye, coefficient, Coefficient,
)


def nm_mcsolve(H, state, tlist, ops_and_rates=(), e_ops=None, ntraj=500, *,
               args=None, options=None, seeds=None, target_tol=None,
               timeout=None):
    """
    Monte-Carlo evolution corresponding to a Lindblad equation with "rates"
    that may be negative. Usage of this function is analogous to ``mcsolve``,
    but the ``c_ops`` parameter is replaced by an ``ops_and_rates`` parameter
    to allow for negative rates. Options for the underlying ODE solver are
    given by the Options class.

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

    ops_and_rates : list
        A ``list`` of tuples ``(L, Gamma)``, where the Lindblad operator ``L``
        is a :class:`qutip.Qobj` and the corresponding rate ``Gamma`` is
        callable. That is, ``Gamma(t)`` returns the rate at time ``t``, which
        is allowed to be negative. The Lindblad operators must be operators
        even if ``H`` is a superoperator. If none are given, the solver will
        defer to ``sesolve`` or ``mesolve``. The ``Gamma(t)`` may be
        specified using any format accepted by :func:`qutip.coefficient`.

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
          Whether or not to store the state density matrices.
          On ``None`` the states will be saved if no expectation operators are
          given.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}, ['text']
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict, [{"chunk_size": 10}]
          kwargs to pass to the progress_bar. Qutip's bars use ``chunk_size``.
        - method : str {"adams", "bdf", "dop853", "vern9", etc.}, ["adams"]
          Which differential equation integration method to use.
        - keep_runs_results : bool, [False]
          Whether to store results from all trajectories or just store the
          averages.
        - map : str {"serial", "parallel", "loky"}, ["serial"]
          How to run the trajectories. "parallel" uses concurrent module to run
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
          Maximum length of one internal step. When using pulses, it should be
          less than half the width of the thinnest pulse.
        - completeness_rtol, completeness_atol : float, float, [1e-5, 1e-8]
          Parameters used in determining whether the given Lindblad operators
          satisfy a certain completeness relation. If they do not, an
          additional Lindblad operator is added automatically (with zero rate).

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
        Maximum time for the evolution in seconds. When reached, no more
        trajectories will be computed.

    Returns
    -------
    results : :class:`qutip.solver.NmmcResult`
        Object storing all results from the simulation. Compared to a result
        returned by ``mcsolve``, this result contains the additional field
        ``trace`` (and ``runs_trace`` if ``store_final_state`` is set). Note
        that the states on the individual trajectories are not normalized. This
        field contains the average of their trace, which will converge to one
        in the limit of sufficiently many trajectories.
    """
    H = QobjEvo(H, args=args, tlist=tlist)

    if len(ops_and_rates) == 0:
        if options is None:
            options = {}
        options = {
            key: options[key]
            for key in options
            if key in MESolver.solver_options
        }
        return mesolve(
            H, state, tlist, e_ops=e_ops, args=args, options=options,
        )
    else:
        ops_and_rates = [
            _parse_op_and_rate(op, rate, tlist=tlist, args=args or {})
            for op, rate in ops_and_rates
        ]

    nmmc = NonMarkovianMCSolver(H, ops_and_rates, options=options)
    result = nmmc.run(state, tlist=tlist, ntraj=ntraj, e_ops=e_ops,
                      seed=seeds, target_tol=target_tol, timeout=timeout)
    return result


def _parse_op_and_rate(op, rate, **kw):
    """ Sanity check the op and convert rates to coefficients. """
    if not isinstance(op, Qobj):
        raise ValueError("NonMarkovianMCSolver ops must be of type Qobj")
    if isinstance(rate, numbers.Number):
        rate = coefficient.const(rate)
    else:
        rate = coefficient(rate, **kw)
    return op, rate


class NonMarkovianMCSolver(MCSolver):
    """
    Monte Carlo Solver for Lindblad equations with "rates" that may be
    negative. The ``c_ops`` parameter of :class:`qutip.MCSolver` is replaced by
    an ``ops_and_rates`` parameter to allow for negative rates. Options for the
    underlying ODE solver are given by the Options class.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`, ``list``, callable.
        System Hamiltonian as a Qobj, QobjEvo. It can also be any input type
        that QobjEvo accepts (see :class:`qutip.QobjEvo` documentation).
        ``H`` can also be a superoperator (liouvillian) if some collapse
        operators are to be treated deterministically.

    ops_and_rates : list
        A ``list`` of tuples ``(L, Gamma)``, where the Lindblad operator ``L``
        is a :class:`qutip.Qobj` and the corresponding rate ``Gamma`` is
        callable. That is, ``Gamma(t)`` returns the rate at time ``t``, which
        is allowed to be negative. The Lindblad operators must be operators
        even if ``H`` is a superoperator.

    args : None / dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : SolverOptions, [optional]
        Options for the evolution.

    seed : int, SeedSequence, list, [optional]
        Seed for the random number generator. It can be a single seed used to
        spawn seeds for each trajectory or a list of seed, one for each
        trajectory. Seeds are saved in the result and can be reused with::

            seeds=prev_result.seeds
    """
    name = "nm_mcsolve"
    resultclass = NmmcResult
    solver_options = {
        **MCSolver.solver_options,
        "completeness_rtol": 1e-5,
        "completeness_atol": 1e-8,
    }

    # "solver" argument will be partially initialized in constructor
    trajectory_resultclass = NmmcTrajectoryResult

    def __init__(self, H, ops_and_rates, *args, options=None, **kwargs):
        # XXX Fix me -- the result class expects solver to be a string
        #     with the name of the solver, but here it is supplied as
        #     an object
        self.trajectory_resultclass = functools.partial(
            NmmcTrajectoryResult, solver=self,
        )

        self.ops_and_rates = [
            _parse_op_and_rate(op, rate)
            for op, rate in ops_and_rates
        ]

        self.options = options
        self._mu_c = None

        self._a_parameter, L = self._check_completeness(ops_and_rates)
        if L is not None:
            self.ops_and_rates.append((L, coefficient.const(0)))

        self._rate_shift = RateShiftCoefficient(
            np.array([f for _, f in self.ops_and_rates], dtype=Coefficient)
        )

        c_ops = self._compute_paired_c_ops()
        super().__init__(H, c_ops, *args, options=options, **kwargs)

    def _check_completeness(self, ops_and_rates):
        """
        Checks whether ``sum(Li.dag() * Li)`` is proportional to the identity
        operator. If not, creates an extra Lindblad operator so that it is.

        Returns the proportionality factor a, and the extra Lindblad operator
        (or None if no extra Lindblad operator is necessary).
        """
        op = sum((L.dag() * L) for L, _ in ops_and_rates)

        a_candidate = op.tr() / op.shape[0]
        with CoreOptions(rtol=self.options["completeness_rtol"],
                         atol=self.options["completeness_atol"]):
            if op == a_candidate * qeye(op.dims[0]):
                return np.real(a_candidate), None

        a = max(op.eigenenergies())
        L = (a * qeye(op.dims[0]) - op).sqrtm()  # new Lindblad operator
        return a, L

    def _compute_paired_c_ops(self):
        """
        Shift all given rate functions Gamma_i by the function rate_shift,
        creating the positive definite rate functions
            gamma_i = Gamma_i + rate_shift.
        Returns the collapse operators
            c_i = L_i * sqrt(gamma_i)
        as QobjEvo objects.
        """
        c_ops = []
        for i, (op, _) in enumerate(self.ops_and_rates):
            # Note that this cannot be done using a lambda expression, since
            # lambda expressions can't be pickled and pickling is required
            # for parallel execution.
            c_ops.append(QobjEvo([op, self._rate_shift.sqrt_shifted_rate(i)]))
        return c_ops

    def _continuous_martingale(self, t):
        """
        Continuous part of the martingale evolution. Checks the stored values
        in self._mu_c for the closest time t0 earlier than the given time t and
        starts integration from there.
        Returns the continuous part of the martingale at time t and stores it
        in self._mu_c.
        """
        if self._mu_c is None:
            raise RuntimeError("The .start() method must called first.")
        if t in self._mu_c:
            return self._mu_c[t]

        earlier_times = filter(lambda t0: t0 < t, self._mu_c.keys())
        try:
            t0 = max(earlier_times)
        except ValueError as exc:
            raise ValueError("Cannot integrate backwards in time.") from exc

        integral = scipy.integrate.quad(self._rate_shift.rate_shift, t0, t)[0]
        result = self._mu_c[t0] * np.exp(self._a_parameter * integral)
        self._mu_c[t] = result
        return result

    def _current_martingale(self):
        """
        Returns the value of the influence martingale along the current
        trajectory. The value of the martingale is the product of the
        continuous and the discrete contribution. The current time and the
        collapses that have happened are read out from the internal integrator.
        """
        t, *_ = self._integrator.get_state(copy=False)
        collapses = np.array(
            self._integrator.collapses,
            dtype=[
                ("time", np.float64),
                ("idx", np.int32),
            ]
        )
        return (
            self._continuous_martingale(t) *
            self._rate_shift.discrete_martingale(
                collapses["time"], collapses["idx"],
            )
        )

    # Override "run" and "start" to precompute
    #     continuous part of martingale evolution
    def run(self, state, tlist, *args, **kwargs):
        self._mu_c = {tlist[0]: 1}
        return super().run(state, tlist, *args, **kwargs)

    def start(self, state, t0, seed=None):
        self._mu_c = {t0: 1}
        return super().start(state, t0, seed=seed)

    def _argument(self, args):
        super()._argument(args)
        self._rate_shift = self._rate_shift.replace_arguments(args)

    # Override "step" to include the martingale mu in the state
    # Note that the returned state will be a density matrix with trace=mu
    def step(self, t, *, args=None, copy=True):
        state = super().step(t, args=args, copy=copy)
        if isket(state):
            state = ket2dm(state)
        return state * self._current_martingale()

    run.__doc__ = MultiTrajSolver.run.__doc__
    start.__doc__ = MultiTrajSolver.start.__doc__
    step.__doc__ = MultiTrajSolver.step.__doc__
