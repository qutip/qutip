import pytest
import numpy as np
from qutip import (
    mesolve, liouvillian, QobjEvo, spre, spost, CoreOptions,
    destroy, coherent, qeye, fock_dm, num, basis
)
from qutip.solver.stochastic import smesolve, ssesolve, SMESolver, SSESolver
from qutip.core import data as _data


def f(t, w):
    return w * t

def _make_system(N, system):
    gamma = 0.25
    a = destroy(N)

    if system == "no sc_ops":
        H = a.dag() * a
        sc_ops = []

    if system == "simple":
        H = a.dag() * a
        sc_ops = [np.sqrt(gamma) * a]

    elif system == "2 c_ops":
        H = QobjEvo([a.dag() * a])
        sc_ops = [np.sqrt(gamma) * a, gamma * a * a]

    elif system == "H td":
        H = [[a.dag() * a, f]]
        sc_ops = [np.sqrt(gamma) * QobjEvo(a)]

    elif system == "complex":
        H = [a.dag() * a + a.dag() + a]
        sc_ops = [np.sqrt(gamma) * a, gamma * a * a]

    elif system == "c_ops td":
        H = [a.dag() * a]
        sc_ops = [[np.sqrt(gamma) * a, f]]

    return H, sc_ops


@pytest.mark.parametrize("system", [
    "no sc_ops", "simple", "2 c_ops", "H td", "complex", "c_ops td",
])
@pytest.mark.parametrize("heterodyne", [True, False])
def test_smesolve(heterodyne, system):
    tol = 0.05
    N = 4
    ntraj = 20

    H, sc_ops = _make_system(N, system)
    c_ops = [destroy(N)]
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 0.1, 21)
    res_ref = mesolve(H, psi0, times, c_ops + sc_ops,
                      e_ops=e_ops, args={"w": 2})

    options = {
        "store_measurement": False,
        "map": "serial",
    }

    res = smesolve(
        H, psi0, times, sc_ops=sc_ops, e_ops=e_ops, c_ops=c_ops,
        ntraj=ntraj, args={"w": 2}, options=options, heterodyne=heterodyne,
        seeds=1,
    )

    for idx in range(len(e_ops)):
        np.testing.assert_allclose(
            res.expect[idx], res_ref.expect[idx], rtol=tol, atol=tol
        )


@pytest.mark.parametrize("system", [
    "no sc_ops", "simple"
])
@pytest.mark.parametrize("heterodyne", [True, False])
@pytest.mark.parametrize("method", SMESolver.avail_integrators().keys())
def test_smesolve_methods(method, heterodyne, system):
    tol = 0.05
    N = 4
    ntraj = 20

    H, sc_ops = _make_system(N, system)
    c_ops = [destroy(N)]
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 0.1, 21)
    res_ref = mesolve(H, psi0, times, c_ops + sc_ops,
                      e_ops=e_ops, args={"w": 2})

    options = {
        "store_measurement": True,
        "map": "parallel",
        "method": method,
    }

    res = smesolve(
        H, psi0, times, sc_ops=sc_ops, e_ops=e_ops, c_ops=c_ops,
        ntraj=ntraj, args={"w": 2}, options=options, heterodyne=heterodyne,
        seeds=list(range(ntraj)),
    )

    for idx in range(len(e_ops)):
        np.testing.assert_allclose(
            res.expect[idx], res_ref.expect[idx], rtol=tol, atol=tol
        )

    assert len(res.measurement) == ntraj

    if heterodyne:
        assert all([
            dw.shape == (len(sc_ops), 2, len(times)-1)
            for dw in res.dW
        ])
        assert all([
            w.shape == (len(sc_ops), 2, len(times))
            for w in res.wiener_process
        ])
        assert all([
            m.shape == (len(sc_ops), 2, len(times)-1)
            for m in res.measurement
        ])
    else:
        assert all([
            dw.shape == (len(sc_ops), len(times)-1)
            for dw in res.dW
        ])
        assert all([
            w.shape == (len(sc_ops), len(times))
            for w in res.wiener_process
        ])
        assert all([
            m.shape == (len(sc_ops), len(times)-1)
            for m in res.measurement
        ])


@pytest.mark.parametrize("system", [
    "no sc_ops", "simple", "2 c_ops", "H td", "complex", "c_ops td",
])
@pytest.mark.parametrize("heterodyne", [True, False])
def test_ssesolve(heterodyne, system):
    tol = 0.1
    N = 4
    ntraj = 20

    H, sc_ops = _make_system(N, system)
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 0.1, 21)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops=e_ops, args={"w": 2})

    options = {
        "map": "serial",
    }

    res = ssesolve(
        H, psi0, times, sc_ops, e_ops=e_ops,
        ntraj=ntraj, args={"w": 2}, options=options, heterodyne=heterodyne,
        seeds=list(range(ntraj)),
    )

    for idx in range(len(e_ops)):
        np.testing.assert_allclose(
            res.expect[idx], res_ref.expect[idx], rtol=tol, atol=tol
        )

    assert res.measurement is None
    assert res.wiener_process is None
    assert res.dW is None


@pytest.mark.parametrize("system", [
    "no sc_ops", "simple"
])
@pytest.mark.parametrize("heterodyne", [True, False])
@pytest.mark.parametrize("method", SSESolver.avail_integrators().keys())
def test_ssesolve_method(method, heterodyne, system):
    "Stochastic: smesolve: homodyne, time-dependent H"
    tol = 0.1
    N = 4
    ntraj = 20

    H, sc_ops = _make_system(N, system)
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 0.1, 21)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops=e_ops, args={"w": 2})

    options = {
        "store_measurement": True,
        "map": "parallel",
        "method": method,
        "keep_runs_results": True,
    }

    res = ssesolve(
        H, psi0, times, sc_ops, e_ops=e_ops,
        ntraj=ntraj, args={"w": 2}, options=options, heterodyne=heterodyne,
        seeds=1,
    )

    for idx in range(len(e_ops)):
        np.testing.assert_allclose(
            res.average_expect[idx], res_ref.expect[idx], rtol=tol, atol=tol
        )

    assert len(res.measurement) == ntraj

    if heterodyne:
        assert all([
            dw.shape == (len(sc_ops), 2, len(times)-1)
            for dw in res.dW
        ])
        assert all([
            w.shape == (len(sc_ops), 2, len(times))
            for w in res.wiener_process
        ])
        assert all([
            m.shape == (len(sc_ops), 2, len(times)-1)
            for m in res.measurement
        ])
    else:
        assert all([
            dw.shape == (len(sc_ops), len(times)-1)
            for dw in res.dW
        ])
        assert all([
            w.shape == (len(sc_ops), len(times))
            for w in res.wiener_process
        ])
        assert all([
            m.shape == (len(sc_ops), len(times)-1)
            for m in res.measurement
        ])


def test_reuse_seeds():
    tol = 0.05
    N = 4
    ntraj = 5

    H, sc_ops = _make_system(N, "simple")
    c_ops = [destroy(N)]
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 0.1, 2)

    options = {
        "store_final_state": True,
        "map": "serial",
        "keep_runs_results": True,
        "store_measurement": True,
    }

    res = smesolve(
        H, psi0, times, sc_ops=sc_ops, e_ops=e_ops, c_ops=c_ops,
        ntraj=ntraj, args={"w": 2}, options=options,
    )

    res2 = smesolve(
        H, psi0, times, sc_ops=sc_ops, e_ops=e_ops, c_ops=c_ops,
        ntraj=ntraj, args={"w": 2}, options=options,
        seeds=res.seeds,
    )

    np.testing.assert_allclose(
        res.wiener_process, res2.wiener_process, atol=1e-14
    )

    np.testing.assert_allclose(res.expect, res2.expect, atol=1e-14)

    for out1, out2 in zip(res.final_state, res2.final_state):
        assert out1 == out2


@pytest.mark.parametrize("heterodyne", [True, False])
def test_measurements(heterodyne):
    N = 10
    ntraj = 1

    H = num(N)
    sc_ops = [destroy(N)]
    psi0 = basis(N, N-1)

    times = np.linspace(0, 1.0, 11)

    solver = SMESolver(H, sc_ops, heterodyne=heterodyne)

    solver.options["store_measurement"] = "start"
    res_start = solver.run(psi0, times, ntraj=ntraj, seeds=1)

    solver.options["store_measurement"] = "middle"
    res_middle = solver.run(psi0, times, ntraj=ntraj, seeds=1)

    solver.options["store_measurement"] = "end"
    res_end = solver.run(psi0, times, ntraj=ntraj, seeds=1)

    diff = np.sum(np.abs(res_end.measurement[0] - res_start.measurement[0]))
    assert diff > 0.1 # Each measurement should be different by ~dt
    np.testing.assert_allclose(
        res_middle.measurement[0] * 2,
        res_start.measurement[0] + res_end.measurement[0],
    )

    np.testing.assert_allclose(
        np.diff(res_start.wiener_process[0][0]), res_start.dW[0][0]
    )


@pytest.mark.parametrize("heterodyne", [True, False])
def test_m_ops(heterodyne):
    N = 10

    H = num(N)
    sc_ops = [destroy(N), qeye(N)]
    psi0 = basis(N, N-1)
    m_ops = [num(N), qeye(N)]
    if heterodyne:
        m_ops = m_ops * 2

    times = np.linspace(0, 1.0, 51)

    options = {"store_measurement": "end",}

    solver = SMESolver(H, sc_ops, heterodyne=heterodyne, options=options)
    solver.m_ops = m_ops
    solver.dW_factors = [0.] * len(m_ops)

    res = solver.run(psi0, times, e_ops=m_ops)
    # With dW_factors=0, measurements are computed as expectation values.
    if heterodyne:
        np.testing.assert_allclose(res.expect[0][1:], res.measurement[0][0][0])
        np.testing.assert_allclose(res.expect[1][1:], res.measurement[0][0][1])
    else:
        np.testing.assert_allclose(res.expect[0][1:], res.measurement[0][0])
        np.testing.assert_allclose(res.expect[1][1:], res.measurement[0][1])

    solver.dW_factors = [1.] * len(m_ops)
    # With dW_factors=0, measurements are computed as expectation values.
    res = solver.run(psi0, times, e_ops=m_ops)
    std = 1 / times[1]**0.5
    if heterodyne:
        noise = res.expect[0][1:] - res.measurement[0][0][0]
    else:
        noise = res.expect[0][1:] - res.measurement[0][0]
    assert np.mean(noise) == pytest.approx(0., abs=std / 50**0.5 * 5)
    assert np.std(noise) == pytest.approx(std, abs=std / 50**0.5 * 5)


def test_feedback():
    N = 10
    ntraj = 2

    def func(t, A, W):
        return (A - 6) * (A.real > 6.) * W(t)[0]

    H = num(10)
    sc_ops = [QobjEvo(
        [destroy(N), func],
        args={
            "A": SMESolver.ExpectFeedback(num(10)),
            "W": SMESolver.WienerFeedback()
        }
    )]
    psi0 = basis(N, N-3)

    times = np.linspace(0, 2, 101)
    options = {"map": "serial", "dt": 0.0005}

    solver = SMESolver(H, sc_ops=sc_ops, heterodyne=False, options=options)
    results = solver.run(psi0, times, e_ops=[num(N)], ntraj=ntraj)

    # If this was deterministic, it should never go under `6`.
    # We add a tolerance ~dt due to the stochatic part.
    assert np.all(results.expect[0] > 6. - 0.001)
    assert np.all(results.expect[0][-20:] < 6.7)


def test_deprecation_warnings():
    with pytest.warns(FutureWarning, match=r'map_func'):
        ssesolve(qeye(2), basis(2), [0, 0.01], [qeye(2)], map_func=None)

    with pytest.warns(FutureWarning, match=r'progress_bar'):
        ssesolve(qeye(2), basis(2), [0, 0.01], [qeye(2)], progress_bar=None)

    with pytest.warns(FutureWarning, match=r'nsubsteps'):
        ssesolve(qeye(2), basis(2), [0, 0.01], [qeye(2)], nsubsteps=None)

    with pytest.warns(FutureWarning, match=r'map_func'):
        ssesolve(qeye(2), basis(2), [0, 0.01], [qeye(2)], map_func=None)

    with pytest.warns(FutureWarning, match=r'store_all_expect'):
        ssesolve(qeye(2), basis(2), [0, 0.01], [qeye(2)], store_all_expect=1)

    with pytest.warns(FutureWarning, match=r'store_measurement'):
        ssesolve(qeye(2), basis(2), [0, 0.01], [qeye(2)], store_measurement=1)

    with pytest.raises(TypeError) as err:
        ssesolve(qeye(2), basis(2), [0, 0.01], [qeye(2)], m_ops=1)
    assert '"m_ops" and "dW_factors"' in str(err.value)


@pytest.mark.parametrize("method", ["euler", "rouchon"])
def test_small_step_warnings(method):
    with pytest.warns(RuntimeWarning, match=r'under minimum'):
        smesolve(
            qeye(2), basis(2), [0, 0.0000001], [qeye(2)],
            options={"method": method}
        )


@pytest.mark.parametrize("method", ["euler", "platen"])
@pytest.mark.parametrize("heterodyne", [True, False])
def test_run_from_experiment_close(method, heterodyne):
    N = 5

    H = num(N)
    a = destroy(N)
    sc_ops = [a, a @ a + (a @ a).dag()]
    psi0 = basis(N, N-1)
    tlist = np.linspace(0, 0.1, 501)
    options = {
        "store_measurement": "start",
        "dt": tlist[1],
        "store_states": True,
        "method": method,
    }
    solver = SSESolver(H, sc_ops, heterodyne, options=options)
    res_forward = solver.run(psi0, tlist, 1, e_ops=[H])
    res_backward = solver.run_from_experiment(
        psi0, tlist, res_forward.dW[0], e_ops=[H]
    )
    res_measure = solver.run_from_experiment(
        psi0, tlist, res_forward.measurement[0], e_ops=[H], measurement=True
    )

    np.testing.assert_allclose(
        res_backward.measurement, res_forward.measurement[0], atol=1e-10
    )
    np.testing.assert_allclose(
        res_measure.measurement, res_forward.measurement[0], atol=1e-10
    )

    np.testing.assert_allclose(res_backward.dW, res_forward.dW[0], atol=1e-10)
    np.testing.assert_allclose(res_measure.dW, res_forward.dW[0], atol=1e-10)

    np.testing.assert_allclose(
        res_backward.expect, res_forward.expect, atol=1e-10
    )
    np.testing.assert_allclose(
        res_measure.expect, res_forward.expect, atol=1e-10
    )


@pytest.mark.parametrize(
    "method", ["euler", "milstein", "platen", "pred_corr"]
)
@pytest.mark.parametrize("heterodyne", [True, False])
def test_run_from_experiment_open(method, heterodyne):
    N = 10

    H = num(N)
    a = destroy(N)
    sc_ops = [a, a.dag() * 0.1]
    psi0 = basis(N, N-1)
    tlist = np.linspace(0, 1, 251)
    options = {
        "store_measurement": "start",
        "dt": tlist[1],
        "store_states": True,
        "method": method,
    }
    solver = SMESolver(H, sc_ops, heterodyne, options=options)
    res_forward = solver.run(psi0, tlist, 1, e_ops=[H])
    res_backward = solver.run_from_experiment(
        psi0, tlist, res_forward.dW[0], e_ops=[H]
    )
    res_measure = solver.run_from_experiment(
        psi0, tlist, res_forward.measurement[0], e_ops=[H], measurement=True
    )

    np.testing.assert_allclose(
        res_backward.measurement, res_forward.measurement[0], atol=1e-10
    )
    np.testing.assert_allclose(
        res_measure.measurement, res_forward.measurement[0], atol=1e-10
    )

    np.testing.assert_allclose(res_backward.dW, res_forward.dW[0], atol=1e-10)
    np.testing.assert_allclose(res_measure.dW, res_forward.dW[0], atol=1e-10)

    np.testing.assert_allclose(
        res_backward.expect, res_forward.expect, atol=1e-10
    )
    np.testing.assert_allclose(
        res_measure.expect, res_forward.expect, atol=1e-10
    )


@pytest.mark.parametrize("store_measurement", [True, False])
@pytest.mark.parametrize("keep_runs_results", [True, False])
def test_merge_results(store_measurement, keep_runs_results):
    # Running smesolve with mixed ICs should be the same as running smesolve
    # multiple times and merging the results afterwards
    initial_state1 = basis([2, 2], [1, 0])
    initial_state2 = basis([2, 2], [1, 1])
    H = qeye([2, 2])
    L = destroy(2) & qeye(2)
    tlist = np.linspace(0, 1, 11)
    e_ops = [num(2) & qeye(2), qeye(2) & num(2)]

    options = {
        "store_measurement": True,
        "keep_runs_results": True,
        "store_states": True,
    }
    solver = SMESolver(H, [L], True, options=options)
    result1 = solver.run(initial_state1, tlist, 5, e_ops=e_ops)

    options = {
        "store_measurement": store_measurement,
        "keep_runs_results": keep_runs_results,
        "store_states": True,
    }
    solver = SMESolver(H, [L], True, options=options)
    result2 = solver.run(initial_state2, tlist, 10, e_ops=e_ops)

    result_merged = result1 + result2
    assert len(result_merged.seeds) == 15
    if store_measurement:
        with CoreOptions(atol=2e-8):
            # In the numpy 1.X test, this fail with the defautl atol=1e-12
            # One operation is computed in single precision?
            assert (
                result_merged.average_states[0] ==
                (initial_state1.proj() + 2 * initial_state2.proj()).unit()
            )
    np.testing.assert_allclose(result_merged.average_expect[0][0], 1)
    np.testing.assert_allclose(result_merged.average_expect[1], 2/3)

    if store_measurement:
        assert len(result_merged.measurement) == 15
        assert len(result_merged.dW) == 15
        assert all(
            dw.shape == result_merged.dW[0].shape
            for dw in result_merged.dW
        )
        assert len(result_merged.wiener_process) == 15
        assert all(
            w.shape == result_merged.wiener_process[0].shape
            for w in result_merged.wiener_process
        )


@pytest.mark.parametrize("open", [True, False])
@pytest.mark.parametrize("heterodyne", [True, False])
def test_step(open, heterodyne):
    state0 = basis(5, 3)
    kw = {}
    if open:
        SolverCls = SMESolver
        state0 = state0.proj()
    else:
        SolverCls = SSESolver

    solver = SolverCls(
        num(5),
        sc_ops=[destroy(5), destroy(5)**2 / 10],
        heterodyne=heterodyne,
        options={"dt": 0.001},
        **kw
    )
    solver.start(state0, t0=0)
    state1 = solver.step(0.01)
    assert state1.dims == state0.dims
    assert state1.norm() == pytest.approx(1, abs=0.01)
    state2, dW = solver.step(0.02, wiener_increment=True)
    assert state2.dims == state0.dims
    assert state2.norm() == pytest.approx(1, abs=0.01)
    if heterodyne:
        assert dW.shape == (2, 2)
        assert abs(dW[0, 0]) < 0.5 # 5 sigmas
    else:
        assert dW.shape == (2,)
        assert abs(dW[0]) < 0.5 # 5 sigmas
