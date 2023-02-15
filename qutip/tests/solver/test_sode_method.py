import numpy as np
from itertools import product
from qutip.core import data as _data
from qutip import qeye, destroy, QobjEvo, rand_ket
import qutip.solver.sode._sode as _sode
import pytest
from qutip.solver.sode.ssystem import SimpleStochasticSystem
from qutip.solver.sode.ito import MultiNoise


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
        noise = MultiNoise(0.1, 0.000001, system.num_collapse)
        for i, t in enumerate(ts):
            out = stepper.run(0, state.copy(), t, noise.dW(t), 1)
            target = system.analytic(t, noise.dw(t)[0]) @ state
            err[i] += _data.norm.l2(out - target)

    err /= num_runs
    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(ts, err)
    return np.polyfit(np.log(ts), np.log(err), 1)[0]


def _make_oper(kind, N):
    if kind == "qeye":
        out = qeye(N) * np.random.rand()
    elif kind == "destroy":
        out = destroy(N) * np.random.rand()
    elif kind == "destroy2":
        out = destroy(N)**2 * np.random.rand()
    return QobjEvo(out)


@pytest.mark.parametrize(["method", "order", "kw"], [
    pytest.param("Euler", 0.5, {}, id="Euler"),
    pytest.param("Milstein", 1.0, {}, id="Milstein"),
    pytest.param("Platen", 1.0, {}, id="Platen"),
    pytest.param("PredCorr", 1.0, {}, id="PredCorr"),
    pytest.param("PredCorr", 1.0, {"alpha": 0.5}, id="PredCorr_0.5"),
    pytest.param("Taylor15", 1.5, {}, id="Taylor15"),
    pytest.param("Explicit15", 1.5, {}, id="Explicit15"),
])
@pytest.mark.parametrize(['H', 'c_ops'], [
    pytest.param("qeye", ["destroy"], id='simple'),
    pytest.param("destroy", ["destroy"], id='simple'),
    pytest.param("qeye", ["qeye", "destroy", "destroy2"], id='2 c_ops'),
])
def test_methods(H, c_ops, method, order, kw):
    N = 5
    H = _make_oper(H, N)
    c_ops = [_make_oper(op, N) for op in c_ops]
    system = SimpleStochasticSystem(H, c_ops)
    state = rand_ket(N).data
    error_order = get_error_order(system, state, method, **kw)
    # The first error term of the method is dt**0.5 greater than the solver
    # order.
    assert (order + 0.35) < error_order
