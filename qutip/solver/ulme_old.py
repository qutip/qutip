





def _make_l_lambda(H, X, env, options=None):
    options = options or {}
    method = options.get("ULME_creation", None)
    if method is None:
        if H.isconstant and X.isconstant:
            method = "eigen"
        else:
            method = "prop"

    if not options["use_lamb_shift"]:
        if method == "eigen":
            return _make_L_eigs(H, X, env), 0
        elif method == "prop":
            return _make_L_prop(H, X, env, options), 0
        raise NotImplementedError(...)

    else:
        if method == "eigen":
            return _make_L_lambda_eigen(H, X, env)
        elif method == "prop":
            return _make_l_lambda_prop(H, X, env, options)
        elif method == "prev":
            return _make_l_lambda_prop_prev(H, X, env, options)
        raise NotImplementedError(...)




def _make_L_prop(
    H: QobjEvo,
    X: QobjEvo,
    env: BosonicEnvironment,
    options=None,
):
    options = options or {}
    T = options.get("T", 15)
    Nt = options.get("Nt", 500)
    Nt = options.get("Nt", 500)
    U = Propagator(H)

    def integrate_1d_flat(op, t, T, Nt):
        ts = np.linspace(-T, T, Nt)
        out = op(ts[0], t) / 2
        out += op(ts[-1], t) / 2
        for s in ts[1:-1]:
            out += op(s, t)
        return out * (2 * T / (Nt - 1))

    def op(s, t):
        Us = U(t+s, t)
        return env.jump_correlator(-s) * Us.dag() @ X(t+s) @ Us

    def L(t):
        return integrate_1d_flat(op, t, T, Nt)

    if H.isconstant and X.isconstant:
        return QobjEvo(L(0))
    else:
        return QobjEvo(L)






class ULOP_backup():
    def __init__(self, H, X, env, options={}):
        self.H = H
        self.X = X
        self.size = H.shape[0]
        self.options = options
        self.t = None
        self._L = None
        self._Lamd = None
        if options.get("prop_options", {}).get("new", False):
            self.integrator = qt.solver.integrator.IntegratorTsit5(self.derr2, {})
        else:
            self.integrator = qt.solver.integrator.IntegratorTsit5(self.derr, {})
        self.prepare_g(env.jump_correlator)

    def prepare_g(self, jc):
        """
        Create a spline for the jump_correlator.
        The jump_correlator is expected to decrease exponentially:
           jc(t) = f(t) * exp(-t*alpha)
        but we don't know the units so have to estimate the cutoff.
        """
        ts = np.logspace(-8, 3, 201)
        jcs = jc(ts)
        idx = np.where(np.abs(jcs) > self.options.get("tol", 1e-6))[0]

        if len(idx) == 0:
            t_max = 1000.
        else:
            t_max = ts[np.max(idx)]

        ts = np.linspace(0, t_max, 1001)
        jcs = jc(ts)
        self.g = qt.coefficient(jcs, tlist=ts)
        self.t_scale = t_max / 100

    @staticmethod
    def merge_states(list_state):
        N = len(list_state)
        if isinstance(list_state, qt.Qobj):
            state0 = [op.data for op in list_state]
        else:
            state0 = list_state
        state0 = [qt.core.data.column_stack(state) for state in state0]
        state = qt.core.data.dense.zeros(state0[0].shape[0], N, fortran=True)
        for i in range(N):
            state.as_ndarray()[:, i] = state0[i].to_array()[:, 0]
        return state

    @staticmethod
    def split_states(state, ncol):
        if isinstance(state, qt.Qobj):
            state = state.data
        # split = qt.core.data.split_columns(state, copy=False)
        out = []
        for op in state.as_ndarray().T:
            data = qt.data.dense.fast_from_numpy(op)
            out.append(qt.data.column_unstack_dense(data, ncol, inplace=True))
        return out

    def initial(self, t):
        Id = qt.data.dense.identity(self.size)
        zero = qt.core.data.dense.zeros(self.size, self.size, fortran=True)
        self._tmp = zero.copy()
        self._Xp = zero.copy()
        self._Xm = zero.copy()
        self._derr = self.merge_states([zero, zero, zero, zero, zero, zero])

        return self.merge_states([Id, Id, zero, zero, zero, zero])

    def derr(self, s, state):
        states = self.split_states(state, self.size)
        # derrivative = self.split_states(self._derr, self.size)
        Xp = states[0].adjoint() @ self.X._call(s + self.t) @ states[0]
        Xm = states[1].adjoint() @ self.X._call(-s + self.t) @ states[1]
        g = self.g(s)
        derr = [
            -1j * self.H._call(s + self.t) @ states[0],
            # FIXME: check the ordering for a time-dependent H. The physical
            # backward propagator P(s) = U(t-s, t) obeys dP/ds = 1j H(t-s) P
            # (left multiplication), with Xm = P^dag X P. Both forms agree
            # when H is constant.
            1j * states[1] @ self.H._call(self.t - s),
            Xp * g.conjugate(),
            Xm * g,
            Xp @ states[2] * (2 * g),
            Xm @ states[3] * (-2 * g.conjugate()),
        ]
        return self.merge_states(derr)

    def derr2(self, s, state):
        #states = self.split_states(state, self.size)
        states = split_dense(state, self.size)
        _derr = _data.dense.zeros(self.size**2, 6, fortran=True)
        #derrivative = self.split_states(_derr, self.size)
        derrivative = split_dense(_derr, self.size)
        g = self.g(s)

        # Inplace is needed for this to work

        #Propagator

        self.H.adjoint_rmatmul_data(self.t + s, states[0], out=derrivative[0], scale=1j)
        self.H.matmul_data(self.t - s, states[1], out=derrivative[1], scale=-1j)
        """
        self.H.matmul_data(self.t + s, states[0], out=derrivative[0], scale=-1j)
        self.H.adjoint_rmatmul_data(self.t - s, states[1], out=derrivative[1], scale=1j)
        """

        # diffusion correction terms

        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t + s, states[0], out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[0], out=derrivative[2], scale=g.conjugate())
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t - s, states[1], out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[1], out=derrivative[3], scale=g)
        """
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t + s, states[0].adjoint(), out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[0].adjoint(), out=derrivative[2], scale=g.conjugate())
        self._tmp = _data.imul_dense(self._tmp, 0)
        self._tmp = self.X.adjoint_rmatmul_data(self.t - s, states[1].adjoint(), out=self._tmp)
        _data.matmul_dag_dense(self._tmp, states[1].adjoint(), out=derrivative[3], scale=g)
        """

        # second order terms for lamb shift
        _data.matmul_dense(derrivative[2], states[2], out=derrivative[4], scale=(2 * g / g.conjugate()))
        _data.matmul_dense(derrivative[3], states[3], out=derrivative[5], scale=(-2 * g.conjugate() / g))

        return _derr

    def L(self, t):
        if t != self.t:
            self.compute(t)
        return qt.Qobj(self._L, dims=self.H._dims)

    def Lamd(self, t):
        if t != self.t:
            self.compute(t)
        return qt.Qobj(self._Lamd, dims=self.H._dims)

    def compute(self, t, tol = 1e-4):
        prev = self.initial(t)
        self.t = t
        self.integrator.set_state(0, prev)
        t_scale = self.t_scale / 10

        diff = tol + 1
        s = 0
        while diff > tol:
            s += t_scale
            _, state = self.integrator.integrate(s)
            diff = np.linalg.norm(
                state.to_array()[:, 2] - prev.to_array()[:, 2], 2
            )
            prev = state
        state = self.split_states(state, self.size)
        self._L = (state[2] + state[3])
        self._Lamd = ((state[2].adjoint() + state[3].adjoint()) @ (-state[2] + state[3]) + (state[-1] + state[-2])) * -0.5j



def _make_l_lambda_prop(H, X, env, options=None):
    op = ULOP(H, X, env, options)

    if H.isconstant and X.isconstant:
        return QobjEvo(op.L(0)), QobjEvo(op.Lamd(0))

    return QobjEvo(op.L), QobjEvo(op.Lamd)


# ------------- Development utility functions ------------

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
        Nt = len(self.ts)

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

        for i, s in enumerate(self.ts[1:]):
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


def _make_l_lambda_prop_prev(H, X, env, options=None):
    options = options or {}
    T = options.get("T", 15)
    Nt = options.get("Nt", 300)
    # Propagator memoization scheme is bad with tracking 2 evolutions at once...
    U1 = Propagator(H, tol=1e-12, options={"method": "tsit5"})
    U2 = Propagator(H, tol=1e-12, options={"method": "tsit5"})
    op = OP([U1, U2], X, env, T, Nt)

    if H.isconstant and X.isconstant:
        return op.L(0), op.Lamd(0)

    return QobjEvo(op.L), QobjEvo(op.Lamd)


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


def _make_lambda_prop_old(
    H: QobjEvo, X: QobjEvo, env: BosonicEnvironment, T=15, Nt=300
):
    U = Propagator(H, memoize=Nt+1, tol=1e-12)

    class OP:
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

            for i in range(1, Nt-1):
                out -= Xs[0] @ Xs[i] * (gs[0] * gs[-i - 1] * 0.5)
                out += Xs[-1] @ Xs[i] * (gs[-1] * gs[-i - 1] * 0.5)
                out -= Xs[i] @ Xs[0] * (gs[i] * gs[-1] * 0.5)
                out += Xs[i] @ Xs[-1] * (gs[i] * gs[0] * 0.5)

            for i in range(1, Nt-1):
                for j in range(i+1, Nt-1):
                    out -= Xs[i] @ Xs[j] * (gs[i] * gs[-j-1])
                    out += Xs[j] @ Xs[i] * (gs[j] * gs[-i-1])

            return out * (dt ** 2 * -0.5j)

    op = OP(U, X, env, T, Nt)

    if H.isconstant and X.isconstant:
        return op(0)

    return QobjEvo(op)
