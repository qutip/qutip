import numpy as np
import scipy
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline
import qutip as qt
import matplotlib.pyplot as plt
import functools
import itertools

def cont_t2w_fft(ft, t_max, Nt):
    dt = t_max * 2**(1 - Nt)
    N = 2**Nt
    ts = np.linspace(-t_max, t_max - dt, N)
    vals = ft(ts)
    ffts = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(vals)) * 2 * t_max / N)[::-1]
    ws = np.fft.fftshift(2 * np.pi  * np.fft.fftfreq(N, dt))
    return CubicSpline(ws[1:], ffts[:-1])


def cont_w2t_fft(fw, t_max, Nt):
    dt = t_max * 2**(1 - Nt)
    N = 2**Nt
    ws = np.linspace(-t_max, t_max - dt, N)
    vals = fw(ws)
    ffts = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(vals)) * t_max / np.pi)[::-1]
    ts = np.fft.fftshift(2 * np.pi  * np.fft.fftfreq(N, dt))
    return CubicSpline(ts[1:], ffts[:-1])


def integrate_1d_flat(op, t, T, Nt):
    ts = np.linspace(-T, T, Nt)
    out = op(ts[0], t) / 2
    out += op(ts[-1], t) / 2
    for s in ts[1:-1]:
        out += op(s, t)
    return out * (2 * T / (Nt - 1))


def _make_L_prop(H: qt.QobjEvo | qt.Qobj, X: qt.QobjEvo | qt.Qobj, g: callable, T, Nt):
    U = qt.Propagator(H)
    if isinstance(X, qt.QobjEvo):
        def op(s, t):
            return g(-s) * U(t, t+s) @ X(t+s) @ U(t+s, t)
    else:
        def op(s, t):
            return g(-s) * U(t, t+s) @ X @ U(t+s, t)

    def L(t):
        return integrate_1d_flat(op, t, T, Nt)

    if isinstance(H, qt.QobjEvo):
        return qt.QobjEvo(L)
    else:
        return L(0)


def _make_L_eigs(H: qt.Qobj, X: qt.Qobj, g: callable):
    vals, vecs = H.eigenstates(output_type="oper")
    X_H = (vecs.dag() @ X @ vecs).data
    L_H = qt.data.multiply(X_H, qt.data.Dense(gw(-np.subtract.outer(vals, vals))))
    L = vecs @ qt.Qobj(L_H, dims=a.dims) @ vecs.dag()
    return L


def _make_lambda_eigen(H, X, g):

    @functools.lru_cache
    def f(e1, e2, eps=1e-5):
        return integrate.quad(
            lambda w: g(w -e1) * g(w + e2),
            -30, 30, weight='cauchy', wvar=0
        )[0] * 2 * np.pi

    vals, vecs = H.eigenstates(output_type="oper")
    X_diag = (vecs.dag() @ X @ vecs).full()
    N = len(vals)
    fs = np.zeros((N, N, N), dtype=float)
    for i, j, k in itertools.product(range(N), repeat=3):
        fs[i, j, k] = f(vals[i] - vals[k], vals[j] - vals[k])
    LL = np.einsum("ijk,ik,kj->ij", fs, X_diag, X_diag)
    return vecs @ qt.Qobj(LL, dims=H.dims) @ vecs.dag()


def _make_lambda_prop(H: qt.QobjEvo, X: qt.QobjEvo, g: callable, T, Nt):
    U = qt.Propagator(H)
    if isinstance(X, qt.QobjEvo):
        def op(t, s, sp):
            if s == sp:
                return 0
            return U(t, s + t) @ X(s + t) @ U(s + t, sp + t) @ X(sp + t) @ U(sp + t, t) * (g(s) * g(-sp) * (-1 + 2 * (s > sp)))
    else:
        def op(t, s, sp):
            if s == sp:
                return 0
            return U(t, s + t) @ X @ U(s + t, sp + t) @ X @ U(sp + t, t) * (g(s) * g(-sp) * (-1 + 2 * (s > sp)))

    if isinstance(H,  qt.QobjEvo) or isinstance(X, qt.QobjEvo):
        return lambda t: integrate_2d_flat(op, t, T, Nt) * -0.5j

    return integrate_2d_flat(op, 0, T, Nt)(0) * -0.5j

def integrate_2d_flat(op, t, T, Nt):
    ts = np.linspace(-T, T, Nt)
    # 4 corners
    out = op(ts[0], ts[0], t) / 4
    out += op(ts[-1], ts[0], t) / 4
    out += op(ts[-1], ts[-1], t) / 4
    out += op(ts[0], ts[-1], t) / 4
    # 4 edges
    for s in ts[1:-1]:
        out += op(ts[0], s, t) / 2
        out += op(ts[-1], s, t) / 2
        out += op(s, ts[0], t) / 2
        out += op(s, ts[-1], t) / 2
    # fill
    for s in ts[1:-1]:
        for sp in ts[1:-1]:
            out += op(s, sp, t)
    return out * (2 * T / (Nt - 1))**2
