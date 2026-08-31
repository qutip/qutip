import numpy as np
from itertools import product
from qutip.core import data as _data
from qutip import (qeye, destroy, QobjEvo, rand_ket, rand_herm, create, Qobj,
                   operator_to_vector, fock_dm)
import qutip.solver.sode._sode as _sode
import pytest
from qutip.solver.sode.ssystem import (
    TaylorStochasticSystem, StochasticOpenSystem, StochasticClosedSystem
)
from qutip.solver.sode._noise import _Noise, Wiener
from qutip.solver.stochastic import SMESolver, _StochasticRHS


class SimpleStochasticSystem(TaylorStochasticSystem):
    """
    Simple system that can be solver analytically.
    Used in tests.

        drift = -iH @ vec

        diffusion = c_i @ vec

    """
    def __init__(self, H, c_ops):
        self.L = -1j * H
        self.c_ops = c_ops

        self.num_diffusion = len(self.c_ops)
        self.dt = 1e-6

    def _drift(self, t, state):
        return self.L.matmul_data(t, state)

    def _diffusion(self, t, state):
        out = []
        for i in range(self.num_diffusion):
            out.append(self.c_ops[i].matmul_data(t, state))
        return out

    def _shift_i(self, _):
        return 0.

    def set_state(self, t, state):
        self.t = t
        self.state = state

    def a(self):
        return self.L.matmul_data(self.t, self.state)

    def bi(self, i):
        return self.c_ops[i].matmul_data(self.t, self.state)

    def Libj(self, i, j):
        bj = self.c_ops[i].matmul_data(self.t, self.state)
        return self.c_ops[j].matmul_data(self.t, bj)

    def Lia(self, i):
        bi = self.c_ops[i].matmul_data(self.t, self.state)
        return self.L.matmul_data(self.t, bi)

    def L0bi(self, i):
        # L0bi = abi' + dbi/dt + Sum_j bjbjbi"/2
        a = self.L.matmul_data(self.t, self.state)
        abi = self.c_ops[i].matmul_data(self.t, a)
        b = self.c_ops[i].matmul_data(self.t, self.state)
        bdt = self.c_ops[i].matmul_data(self.t + self.dt, self.state)
        return abi + (bdt - b) / self.dt

    def LiLjbk(self, i, j, k):
        bk = self.c_ops[k].matmul_data(self.t, self.state)
        Ljbk = self.c_ops[j].matmul_data(self.t, bk)
        return self.c_ops[i].matmul_data(self.t, Ljbk)

    def L0a(self):
        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        a = self.L.matmul_data(self.t, self.state)
        aa = self.L.matmul_data(self.t, a)
        adt = self.L.matmul_data(self.t + self.dt, self.state)
        return aa + (adt - a) / self.dt

    def analytic(self, t, W):
        """
        Analytic solution, H and all c_ops must commute.
        Error of order t**3
        """
        def _intergal(f, T):
            return (f(0) + 4 * f(T/2) + f(T)) / 6

        out = _intergal(self.L, t) * t
        for i in range(self.num_diffusion):
            out += _intergal(self.c_ops[i], t) * W[i]
            out -= 0.5 * _intergal(
                lambda t: self.c_ops[i](t) @ self.c_ops[i](t), t
            ) * t
        return out.expm().data


def get_error_order(system, state, method, plot=False, **kw):
    stepper = getattr(_sode, method)(system, **kw)
    num_runs = 10
    ts = 0.1 * (0.5) ** np.arange(16)
    # state = rand_ket(system.dims[0]).data
    err = np.zeros(len(ts), dtype=float)
    for _ in range(num_runs):
        noise = _Noise(ts[0], ts[-1], system.num_diffusion)
        for i, t in enumerate(ts):
            out = stepper.run(0, state.copy(), t, noise.dW(t), 1)
            target = system.analytic(t, noise.dw(t)[0]) @ state
            err[i] += _data.norm.l2(out - target)

    err /= num_runs
    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(ts, err)
    return np.polyfit(np.log(ts), np.log(err + 1e-20), 1)[0]


def _make_oper(kind, N):
    a = destroy(N)
    if kind == "qeye":
        out = qeye(N) * np.random.rand()
    elif kind == "create":
        out = a.dag() * np.random.rand()
    elif kind == "destroy":
        out = a * np.random.rand()
    elif kind == "destroy td":
        out = [a, lambda t: 1 + t/2]
    elif kind == "destroy2":
        out = a**2
    elif kind == "herm":
        out = rand_herm(N)
    elif kind == "herm td":
        out = [rand_herm(N), lambda t: -1 + t/2 + t**2]
    elif kind == "random":
        out = Qobj(np.random.randn(N, N) + 1j * np.random.rand(N, N))
    return QobjEvo(out)


@pytest.mark.parametrize(["method", "order", "kw"], [
    pytest.param("Euler", 0.5, {}, id="Euler"),
    pytest.param("Milstein", 1.0, {}, id="Milstein"),
    pytest.param("Milstein_imp", 1.0, {}, id="Milstein implicit"),
    pytest.param("Milstein_imp", 1.0, {"solve_method": "inv"},
                 id="Milstein implicit inv"),
    pytest.param("Platen", 1.0, {}, id="Platen"),
    pytest.param("PredCorr", 1.0, {}, id="PredCorr"),
    pytest.param("PredCorr", 1.0, {"alpha": 0.5}, id="PredCorr_0.5"),
    pytest.param("Taylor15", 1.5, {}, id="Taylor15"),
    pytest.param("Explicit15", 1.5, {}, id="Explicit15"),
    pytest.param("Taylor15_imp", 1.5, {}, id="Taylor15 implicit"),
    pytest.param("Taylor15_imp", 1.5, {"solve_method": "inv"},
                 id="Taylor15 implicit inv"),
])
@pytest.mark.parametrize(['H', 'sc_ops'], [
    pytest.param("qeye", ["destroy"], id='simple'),
    pytest.param("destroy", ["destroy"], id='destroy'),
    pytest.param("destroy", ["destroy td"], id='sc_ops td'),
    pytest.param("herm td", ["qeye"], id='H td'),
    pytest.param("qeye", ["qeye", "destroy", "destroy2"], id='3 sc_ops'),
])
def test_methods(H, sc_ops, method, order, kw):
    if kw == {"solve_method": "inv"} and ("td" in H or "td" in sc_ops[0]):
        pytest.skip("inverse method only available for constant cases.")
    N = 5
    H = _make_oper(H, N)
    sc_ops = [_make_oper(op, N) for op in sc_ops]
    system = SimpleStochasticSystem(H, sc_ops)
    state = rand_ket(N).data
    error_order = get_error_order(system, state, method, **kw)
    # The first error term of the method is dt**0.5 greater than the solver
    # order.
    assert (order + 0.25) < error_order


def get_error_order_integrator(
    integrator, ref_integrator, N_sc_ops, plot=False
):
    ts = np.logspace(-4, -1, 20)
    err = np.zeros(len(ts), dtype=float)
    state = operator_to_vector(fock_dm(5, 3, dtype="Dense")).data
    for i, t in enumerate(ts):
        integrator.options["dt"] = t
        ref_integrator.options["dt"] = t
        wiener = Wiener(0, t, np.random.default_rng(0), N_sc_ops)
        integrator.set_state(0., state, wiener)
        ref_integrator.set_state(0., state, wiener)
        out = integrator.integrate(t)[1]
        target = ref_integrator.integrate(t)[1]
        err[i] = _data.norm.l2(out - target)

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(ts, err)
    if np.all(err < 1e-12):
        # Exact match
        return np.inf
    return np.polyfit(np.log(ts), np.log(err + 1e-20), 1)[0]


@pytest.mark.parametrize(["method", "order"], [
    pytest.param("euler", 0.5, id="Euler"),
    pytest.param("milstein", 1.0, id="Milstein"),
    pytest.param("milstein_imp", 1.0, id="Milstein implicit"),
    pytest.param("platen", 1.0, id="Platen"),
    pytest.param("pred_corr", 1.0, id="PredCorr"),
    pytest.param("explicit1.5", 1.5, id="Explicit15"),
    pytest.param("taylor1.5_imp", 1.5, id="Taylor15 implicit"),
])
@pytest.mark.parametrize(['H', 'c_ops', 'sc_ops'], [
    pytest.param("qeye", [], ["destroy"], id='simple'),
    pytest.param("qeye", ["destroy"], ["destroy"], id='simple + collapse'),
    pytest.param("herm", ["destroy", "destroy2"], [], id='2 c_ops'),
    pytest.param("herm", [], ["destroy", "destroy2"], id='2 sc_ops'),
    pytest.param("herm", ["create", "destroy"], ["destroy", "destroy2"],
                 id='many terms'),
    pytest.param("herm", [], ["random"], id='random'),
    pytest.param("herm", ["random"], ["random"], id='complex'),
    pytest.param("herm td", ["random"], ["destroy"], id='H td'),
    pytest.param("herm", ["random"], ["destroy td"], id='sc_ops td'),
])
def test_open_integrator(method, order, H, c_ops, sc_ops):
    N = 5
    H = _make_oper(H, N)
    c_ops = [_make_oper(op, N) for op in c_ops]
    sc_ops = [_make_oper(op, N) for op in sc_ops]
    opt = {"dt": 0.01}

    rhs = _StochasticRHS(StochasticOpenSystem, H, sc_ops, c_ops, False)
    system = rhs(opt)
    ref_sode = SMESolver.avail_integrators()["taylor1.5"](system, opt)
    sode = SMESolver.avail_integrators()[method](system, opt)

    error_order = get_error_order_integrator(sode, ref_sode, len(sc_ops))
    assert (order + 0.25) < error_order


@pytest.mark.parametrize(["method", "order"], [
    pytest.param("euler", 0.5, id="Euler"),
    pytest.param("platen", 1.0, id="Platen"),
])
@pytest.mark.parametrize(['H', 'sc_ops'], [
    pytest.param("qeye", ["destroy"], id='simple'),
    pytest.param("herm", ["destroy", "destroy2"], id='2 sc_ops'),
    pytest.param("herm", ["random"], id='random'),
    pytest.param("herm td", ["destroy"], id='H td'),
    pytest.param("herm", ["destroy td"], id='sc_ops td'),
])
def test_closed_integrator(method, order, H, sc_ops):
    N = 5
    H = _make_oper(H, N)
    sc_ops = [_make_oper(op, N) for op in sc_ops]
    opt = {"dt": 0.01}

    rhs = _StochasticRHS(StochasticClosedSystem, H, sc_ops, (), False)
    system = rhs(opt)
    ref_sode = SMESolver.avail_integrators()["explicit1.5"](system, opt)
    sode = SMESolver.avail_integrators()[method](system, opt)

    error_order = get_error_order_integrator(sode, ref_sode, len(sc_ops))
    assert (order + 0.25) < error_order


@pytest.mark.parametrize(["method", "order"], [
    pytest.param("rouchon", 1.0, id="rouchon"),
])
@pytest.mark.parametrize(['H', 'c_ops', 'sc_ops'], [
    pytest.param("qeye", [], ["destroy"], id='simple'),
    pytest.param("qeye", ["destroy"], ["destroy"], id='simple + collapse'),
    pytest.param("herm", ["destroy", "destroy2"], [], id='2 c_ops'),
    pytest.param("herm", [], ["destroy", "destroy2"], id='2 sc_ops'),
    pytest.param("herm", ["create", "destroy"], ["destroy", "destroy2"],
                 id='many terms'),
    pytest.param("herm", [], ["random"], id='random'),
    pytest.param("herm", ["random"], ["random"], id='complex'),
    pytest.param("herm td", ["random"], ["destroy"], id='H td'),
    pytest.param("herm", ["random"], ["destroy td"], id='sc_ops td'),
])
def test_open_integrator_system_format(method, order, H, c_ops, sc_ops):
    N = 5
    H = _make_oper(H, N)
    c_ops = [_make_oper(op, N) for op in c_ops]
    sc_ops = [_make_oper(op, N) for op in sc_ops]
    opt = {"dt": 0.01}

    rhs = _StochasticRHS(StochasticOpenSystem, H, sc_ops, c_ops, False)
    ref_sode = SMESolver.avail_integrators()["taylor1.5"](rhs(opt), opt)
    sode = SMESolver.avail_integrators()[method](rhs, opt)

    error_order = get_error_order_integrator(sode, ref_sode, len(sc_ops))
    assert (order + 0.25) < error_order


@pytest.mark.parametrize(["method", "order"], [
    pytest.param("rouchon", 1.0, id="Rouchon"),
])
@pytest.mark.parametrize(['H', 'sc_ops'], [
    pytest.param("qeye", ["destroy"], id='simple'),
    pytest.param("herm", ["destroy", "destroy2"], id='2 sc_ops'),
    pytest.param("herm", ["random"], id='random'),
    pytest.param("herm td", ["destroy"], id='H td'),
    pytest.param("herm", ["destroy td"], id='sc_ops td'),
])
def test_closed_integrator_system_format(method, order, H, sc_ops):
    N = 5
    H = _make_oper(H, N)
    sc_ops = [_make_oper(op, N) for op in sc_ops]
    opt = {"dt": 0.01}

    rhs = _StochasticRHS(StochasticClosedSystem, H, sc_ops, (), False)
    ref_sode = SMESolver.avail_integrators()["explicit1.5"](rhs(opt), opt)
    sode = SMESolver.avail_integrators()[method](rhs, opt)

    error_order = get_error_order_integrator(sode, ref_sode, len(sc_ops))
    assert (order + 0.25) < error_order
