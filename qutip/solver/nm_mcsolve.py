__all__ = ['NonMarkovianMCSolver']

from functools import partial
import numpy as np
import scipy
from .mcsolve import MCSolver
from .result import McResult
from ..core import CoreOptions, QobjEvo, isket, ket2dm, qeye


class NonMarkovianMCSolver(MCSolver):
    r"""
    Monte-Carlo solver for master equation with negative rates.
    Based on the methods explained in arXiv:2209.08958 [quant-ph]
    """
    name = "nm_mcsolve"
    resultclass = McResult
    solver_options = {
        **MCSolver.solver_options,
        "completeness_rtol": 1e-5,
        "completeness_atol": 1e-8,
    }

    # ops_and_rates is a list of tuples (L_i, Gamma_i),
    #     where Gamma_i = Gamma_i(t) is callable
    def __init__(self, H, ops_and_rates, *args, options=None, **kwargs):
        self.ops_and_rates = list(ops_and_rates)
        self.options = options
        self._mu_c = None

        self._a_parameter, L = self._check_completeness(ops_and_rates)
        if L is not None:
            self.ops_and_rates.append((L, float))  # float() is zero

        c_ops = self._compute_paired_c_ops()
        super().__init__(H, c_ops, *args, options=options, **kwargs)

    # Check whether op = sum(Li.dag() * Li) is proportional to identity
    # If not, creates an extra Lindblad operator so that it is
    # Returns: * the proportionality factor a
    #          * the extra Lindblad operator (or None if not necessary)
    def _check_completeness(self, ops_and_rates):
        op = sum((L.dag() * L) for L, _ in ops_and_rates)

        a_candidate = op.tr() / op.shape[0]
        with CoreOptions(rtol=self.options["completeness_rtol"],
                         atol=self.options["completeness_atol"]):
            if op == a_candidate * qeye(op.dims[0]):
                return np.real(a_candidate), None

        a = max(op.eigenenergies())
        L = (a * qeye(op.dims[0]) - op).sqrtm()  # new Lindblad operator
        return a, L

    # Shifts all rate function by the function rate_shift
    # Returns c_i = L_i * sqrt(gamma_i) as QobjEvo objects
    def _compute_paired_c_ops(self):
        c_ops = []
        for f in self.ops_and_rates:
            sqrt_gamma = partial(self._sqrt_gamma, original_rate=f[1])
            c_ops.append(QobjEvo([f[0], sqrt_gamma]))
        return c_ops

    def _rate_shift(self, t):
        min_rate = min(f[1](t) for f in self.ops_and_rates)
        return 2 * abs(min(0, min_rate))

    def _sqrt_gamma(self, t, original_rate):
        return np.sqrt(original_rate(t) + self._rate_shift(t))

    # Continuous part of the martingale evolution
    # Checks self._mu_c for the closest time t0 earlier than the given time t
    #     and starts integration from there.
    # Returns the continuous part of the martingale at time t and stores it in
    #     self._mu_c
    def _continuous_martingale(self, t):
        if self._mu_c is None:
            raise RuntimeError("The `start` method must called first.")
        if t in self._mu_c:
            return self._mu_c[t]

        earlier_times = filter(lambda t0: t0 < t, self._mu_c.keys())
        try:
            t0 = max(earlier_times)
        except ValueError as exc:
            raise ValueError("Cannot integrate backwards in time.") from exc

        integral = scipy.integrate.quad(self._rate_shift, t0, t)[0]
        result = self._mu_c[t0] * np.exp(self._a_parameter * integral)
        self._mu_c[t] = result
        return result

    # Discrete part of the martingale evolution
    # collapses is a list of (t_k, i_k)
    def _discrete_martingale(self, collapses):
        o_r = self.ops_and_rates
        factors = [o_r[ik][1](tk) / (o_r[ik][1](tk) + self._rate_shift(tk))
                   for tk, ik in collapses]
        return np.prod(factors)

    # Override "run" and "start" to initialize continuous part
    #     of martingale evolution
    def run(self, state, tlist, *args, **kwargs):
        self._mu_c = {tlist[0]: 1}
        for t in tlist[1:]:
            self._continuous_martingale(t)  # precompute self._mu_c
        return super().run(state, tlist, *args, **kwargs)

    def start(self, state, t0, seed=None):
        self._mu_c = {t0: 1}
        return super().start(state, t0, seed=seed)

    # Override "_restore_state" to include the martingale in the state
    def _restore_state(self, data, t, *, copy=True):
        # find state |psi><psi|
        state = super()._restore_state(data, t, copy=copy)
        if isket(state):
            # influence martingale is for weighting density matrices, not kets!
            state = ket2dm(state)

        # find martingale mu
        collapses = self._integrator.collapses
        mu = self._continuous_martingale(t) *\
            self._discrete_martingale(collapses)

        # return weighted state
        return mu * state
