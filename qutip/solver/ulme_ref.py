"""
Slow reference implementations of the ULME jump operator and Lamb shift.

These are NOT part of the library. They are kept for the tests, to check the
production code (``qutip.solver.ulme``) against simpler, independent
implementations. Suggested location: ``qutip/tests/solver/``.

Summary
-------
jump_operator_quad
    Jump operator only. Trapezoidal rule over ``s in [-T, T]`` using
    ``Propagator``.
jump_operator_and_lamb_discrete
    Jump operator and Lamb shift. Discrete (Riemann) sums on a fixed grid.
lamb_shift_double_sum
    Lamb shift only. Oldest version, O(Nt^2) matrix products.
cont_t2w_fft, cont_w2t_fft
    Unused FFT helpers, kept here until they find a home.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from qutip import Propagator, QobjEvo


def _trapezoid(func, t, T, Nt):
    """Trapezoidal integral of ``func(s, t)`` for ``s`` in ``[-T, T]``."""
    ts = np.linspace(-T, T, Nt)
    out = func(ts[0], t) / 2
    out += func(ts[-1], t) / 2
    for s in ts[1:-1]:
        out += func(s, t)
    return out * (2 * T / (Nt - 1))


def jump_operator_quad(H, X, env, T=15, Nt=500):
    """
    Jump operator ``L(t) = int_{-T}^{T} ds g(-s) X_I(t + s)`` by the
    trapezoidal rule. ``H`` and ``X`` are QobjEvo.
    """
    U = Propagator(H)

    def integrand(s, t):
        Us = U(t + s, t)
        return env.jump_correlator(-s) * Us.dag() @ X(t + s) @ Us

    def L(t):
        return _trapezoid(integrand, t, T, Nt)

    if H.isconstant and X.isconstant:
        return QobjEvo(L(0))
    return QobjEvo(L)


class OP:
    def __init__(self, U, X, env, T, Nt):
        self.U1 = U[0]
        self.U2 = U[1]
        self.X = X
        self.g = env.jump_correlator
        self.ts = np.linspace(0, T, Nt)
        self.t = None
        self._L = None
        self._Lamd = None

    def compute(self, t):
        # TODO: auto detect T, Nt according to the jump_correlator convergence
        dt = self.ts[1] - self.ts[0]

        Xp = self.X(t)
        Xm = self.X(t)
        g = self.g(0)

        Lp = Xp * (g.conjugate() * 0.5)
        Lm = Xm * (g * 0.5)
        Ip = Xp * (g * 0.5)
        Im = Xm * (g.conjugate() * 0.5)
        Yp = Xp @ Lp * g
        Ym = Xm @ Lm * (g.conjugate() * -1)
        Ut = self.U1(t)
        Ut_inv = Ut.dag()

        for s in self.ts[1:]:
            Us = self.U1(s + t) @ Ut_inv
            Xp = Us.dag() @ self.X(s + t) @ Us
            Us = Ut @ self.U2(0, t - s)
            Xm = Us @ self.X(t - s) @ Us.dag()
            g = self.g(s)

            Lp += Xp * g.conjugate()
            Lm += Xm * g
            Ip += Xp * g
            Im += Xm * g.conjugate()
            Yp += Xp @ Lp * (g * 2)
            Ym += Xm @ Lm * (g.conjugate() * -2)

        self._L = (Lp + Lm) * dt
        self._Lamd = ((Ip + Im) @ (-Lp + Lm) + (Yp + Ym)) * (dt**2 * -0.5j)
        self.t = t

    def L(self, t):
        if t != self.t:
            self.compute(t)
        return self._L

    def Lamd(self, t):
        if t != self.t:
            self.compute(t)
        return self._Lamd


def jump_operator_and_lamb_discrete(H, X, env, T=15, Nt=300):
    """
    Jump operator and Lamb shift from discrete sums over ``Nt`` points in
    ``s in [0, T]``. Returns Qobj for constant systems, QobjEvo otherwise.
    """
    # Propagator memoization scheme is bad with tracking 2 evolutions at once
    U1 = Propagator(H, tol=1e-12, options={"method": "tsit5"})
    U2 = Propagator(H, tol=1e-12, options={"method": "tsit5"})
    op = OP([U1, U2], X, env, T, Nt)

    if H.isconstant and X.isconstant:
        return op.L(0), op.Lamd(0)
    return QobjEvo(op.L), QobjEvo(op.Lamd)


def lamb_shift_double_sum(H, X, env, T=15, Nt=300):
    """Lamb shift only, with an explicit double sum over the time grid."""
    U = Propagator(H, memoize=Nt + 1, tol=1e-12)

    class _Lamb:
        def __init__(self, U, X, env, T, Nt):
            self.U = U
            self.X = X
            self.g = env.jump_correlator
            self.ts = np.linspace(-T, T, Nt)

        def __call__(self, t):
            Xs = []
            gs = self.g(self.ts)
            for s in self.ts:
                Us = U(t, s + t)
                Xs.append(Us @ self.X(s + t) @ Us.dag())

            dt = self.ts[1] - self.ts[0]

            out = Xs[0] @ Xs[-1] * (gs[0] * gs[0] * -0.25)
            out += Xs[-1] @ Xs[0] * (gs[-1] * gs[-1] * 0.25)

            for i in range(1, Nt - 1):
                out -= Xs[0] @ Xs[i] * (gs[0] * gs[-i - 1] * 0.5)
                out += Xs[-1] @ Xs[i] * (gs[-1] * gs[-i - 1] * 0.5)
                out -= Xs[i] @ Xs[0] * (gs[i] * gs[-1] * 0.5)
                out += Xs[i] @ Xs[-1] * (gs[i] * gs[0] * 0.5)

            for i in range(1, Nt - 1):
                for j in range(i + 1, Nt - 1):
                    out -= Xs[i] @ Xs[j] * (gs[i] * gs[-j - 1])
                    out += Xs[j] @ Xs[i] * (gs[j] * gs[-i - 1])

            return out * (dt ** 2 * -0.5j)

    op = _Lamb(U, X, env, T, Nt)

    if H.isconstant and X.isconstant:
        return op(0)
    return QobjEvo(op)


def cont_t2w_fft(ft, t_max, Nt):
    dt = t_max * 2**(1 - Nt)
    N = 2**Nt
    ts = np.linspace(-t_max, t_max - dt, N)
    vals = ft(ts)
    ffts = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(vals)) * 2 * t_max / N
    )[::-1]
    ws = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(N, dt))
    return CubicSpline(ws[1:], ffts[:-1])


def cont_w2t_fft(fw, t_max, Nt):
    dt = t_max * 2**(1 - Nt)
    N = 2**Nt
    ws = np.linspace(-t_max, t_max - dt, N)
    vals = fw(ws)
    ffts = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(vals)) * t_max / np.pi
    )[::-1]
    ts = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(N, dt))
    return CubicSpline(ts[1:], ffts[:-1])
