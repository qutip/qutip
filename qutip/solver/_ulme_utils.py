




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
        out = 0
        ts = np.linspace(-T, T, Nt)
        out = op(-ts[0], t) / 2
        out += op(ts[-1], t) / 2
        for s in ts[1:-1]:
            out += op(s, t)
        return out * (2 * T / (Nt - 1))

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
