import pytest
import numpy as np
from qutip import (
    mesolve, liouvillian, QobjEvo, spre, spost,
    destroy, coherent, qeye, fock_dm, num, basis
)
from qutip.solver.stochastic import smesolve, ssesolve, SMESolver, SSESolver
from qutip.core import data as _data


def f(t, w):
    return w * t

def _make_system(N, system):
    gamma = 0.25
    a = destroy(N)

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
    "simple", "2 c_ops", "H td", "complex", "c_ops td",
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
    res_ref = mesolve(H, psi0, times, c_ops + sc_ops, e_ops, args={"w": 2})

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


@pytest.mark.parametrize("heterodyne", [True, False])
@pytest.mark.parametrize("method", SMESolver.avail_integrators().keys())
def test_smesolve_methods(method, heterodyne):
    tol = 0.05
    N = 4
    ntraj = 20
    system = "simple"

    H, sc_ops = _make_system(N, system)
    c_ops = [destroy(N)]
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 0.1, 21)
    res_ref = mesolve(H, psi0, times, c_ops + sc_ops, e_ops, args={"w": 2})

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
    "simple", "2 c_ops", "H td", "complex", "c_ops td",
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
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops, args={"w": 2})

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


@pytest.mark.parametrize("heterodyne", [True, False])
@pytest.mark.parametrize("method", SSESolver.avail_integrators().keys())
def test_ssesolve_method(method, heterodyne):
    "Stochastic: smesolve: homodyne, time-dependent H"
    tol = 0.1
    N = 4
    ntraj = 20
    system = "simple"

    H, sc_ops = _make_system(N, system)
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 0.1, 21)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops, args={"w": 2})

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
def test_m_ops(heterodyne):
    N = 10
    ntraj = 1

    H = num(N)
    sc_ops = [destroy(N), qeye(N)]
    psi0 = basis(N, N-1)
    m_ops = [num(N), qeye(N)]
    if heterodyne:
        m_ops = m_ops * 2

    times = np.linspace(0, 1.0, 51)

    options = {"store_measurement": True,}

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
    tol = 0.05
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

    times = np.linspace(0, 10, 101)
    options = {"map": "serial", "dt": 0.001}

    solver = SMESolver(H, sc_ops=sc_ops, heterodyne=False, options=options)
    results = solver.run(psi0, times, e_ops=[num(N)], ntraj=ntraj)

    assert np.all(results.expect[0] > 6.-1e-6)
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
