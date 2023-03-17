import pytest
import numpy as np
from qutip import (
    mesolve, liouvillian, QobjEvo, spre, spost,
    destroy, coherent, qeye, fock_dm, num
)
from qutip.solver.stochastic import smesolve, ssesolve
from qutip.core import data as _data


def f(t, a):
    return a * t

def _make_system(N, system):
    gamma = 0.25
    a = destroy(N)

    if system == "simple":
        H = [a.dag() * a]
        sc_ops = [np.sqrt(gamma) * a]

    elif system == "2 c_ops":
        H = [a.dag() * a]
        sc_ops = [np.sqrt(gamma) * a, gamma * a * a]

    elif system == "H td":
        H = [[a.dag() * a, f]]
        sc_ops = [np.sqrt(gamma) * a]

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
@pytest.mark.parametrize("method", ["milstein", "platen", "rouchon"])
def test_smesolve(method, heterodyne, system):
    "Stochastic: smesolve: homodyne, time-dependent H"
    tol = 0.05
    N = 4
    ntraj = 20

    H, sc_ops = _make_system(N, system)
    c_ops = [destroy(N)]
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 1.0, 21)
    res_ref = mesolve(H, psi0, times, c_ops + sc_ops, e_ops, args={"a": 2})

    options = {
        "store_measurement": True,
        "map": "serial",
        "method": method,
    }

    res = smesolve(
        H, psi0, times, sc_ops=sc_ops, e_ops=e_ops, c_ops=c_ops,
        ntraj=ntraj, args={"a": 2}, options=options, heterodyne=heterodyne,
    )

    for idx in range(len(e_ops)):
        np.testing.assert_allclose(
            res.expect[idx], res_ref.expect[idx], rtol=tol, atol=tol
        )

    assert len(res.measurement) == ntraj

    if heterodyne:
        assert all([
            m.shape == (len(sc_ops), 2, len(times)-1)
            for m in res.measurement
        ])
    else:
        assert all([
            m.shape == (len(sc_ops), len(times)-1)
            for m in res.measurement
        ])


@pytest.mark.parametrize("system", [
    "simple", "2 c_ops", "H td", "complex", "c_ops td",
])
@pytest.mark.parametrize("heterodyne", [True, False])
@pytest.mark.parametrize("method", ["platen", "rouchon"])
def test_ssesolve(method, heterodyne, system):
    "Stochastic: smesolve: homodyne, time-dependent H"
    tol = 0.1
    N = 4
    ntraj = 20

    H, sc_ops = _make_system(N, system)
    psi0 = coherent(N, 0.5)
    a = destroy(N)
    e_ops = [a.dag() * a, a + a.dag(), (-1j)*(a - a.dag())]

    times = np.linspace(0, 1.0, 21)
    res_ref = mesolve(H, psi0, times, sc_ops, e_ops, args={"a": 2})

    options = {
        "store_measurement": True,
        "map": "serial",
        "method": method,
    }

    res = ssesolve(
        H, psi0, times, sc_ops, e_ops,
        ntraj=ntraj, args={"a": 2}, options=options, heterodyne=heterodyne,
    )

    for idx in range(len(e_ops)):
        np.testing.assert_allclose(
            res.expect[idx], res_ref.expect[idx], rtol=tol, atol=tol
        )

    assert len(res.measurement) == ntraj

    if heterodyne:
        assert all([
            m.shape == (len(sc_ops), 2, len(times)-1)
            for m in res.measurement
        ])
    else:
        assert all([
            m.shape == (len(sc_ops), len(times)-1)
            for m in res.measurement
        ])
