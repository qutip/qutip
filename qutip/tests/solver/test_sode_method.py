import numpy as np
from itertools import product
from qutip.core import data as _data
from qutip import (qeye, destroy, QobjEvo, rand_ket, rand_herm, create, Qobj,
                   operator_to_vector, fock_dm)
import qutip.solver.sode._sode as _sode
import pytest
from qutip.solver.sode.ssystem import (
    SimpleStochasticSystem, StochasticOpenSystem, StochasticClosedSystem
)
from qutip.solver.sode._noise import _Noise
from qutip.solver.stochastic import SMESolver, _StochasticRHS


def get_error_order(system, state, method, plot=False, **kw):
    stepper = getattr(_sode, method)(system, **kw)
    num_runs = 10
    ts = [
        0.000001, 0.000002, 0.000005, 0.00001, 0.00002,  0.00005,
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
    ]
    # state = rand_ket(system.dims[0]).data
    err = np.zeros(len(ts), dtype=float)
    for _ in range(num_runs):
        noise = _Noise(0.1, 0.000001, system.num_collapse)
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
    assert (order + 0.35) < error_order


def get_error_order_integrator(integrator, ref_integrator, state, plot=False):
    ts = [
        0.000001, 0.000002, 0.000005, 0.00001, 0.00002,  0.00005,
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
    ]
    # state = rand_ket(system.dims[0]).data
    err = np.zeros(len(ts), dtype=float)
    for i, t in enumerate(ts):
        integrator.options["dt"] = t
        ref_integrator.options["dt"] = t
        integrator.set_state(0., state, np.random.default_rng(0))
        ref_integrator.set_state(0., state, np.random.default_rng(0))
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
    pytest.param("rouchon", 1.0, id="rouchon"),
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

    rhs = _StochasticRHS(StochasticOpenSystem, H, sc_ops, c_ops, False)
    ref_sode = SMESolver.avail_integrators()["taylor1.5"](rhs, {"dt": 0.01})
    sode = SMESolver.avail_integrators()[method](rhs, {"dt": 0.01})
    state = operator_to_vector(fock_dm(5, 3, dtype="Dense")).data

    error_order = get_error_order_integrator(sode, ref_sode, state)
    assert (order + 0.35) < error_order


@pytest.mark.parametrize(["method", "order"], [
    pytest.param("euler", 0.5, id="Euler"),
    pytest.param("platen", 1.0, id="Platen"),
    pytest.param("rouchon", 1.0, id="Rouchon"),
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

    rhs = _StochasticRHS(StochasticClosedSystem, H, sc_ops, (), False)
    ref_sode = SMESolver.avail_integrators()["explicit1.5"](rhs, {"dt": 0.01})
    sode = SMESolver.avail_integrators()[method](rhs, {"dt": 0.01})
    state = operator_to_vector(fock_dm(5, 3, dtype="Dense")).data

    error_order = get_error_order_integrator(sode, ref_sode, state)
    assert (order + 0.35) < error_order
