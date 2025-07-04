import numpy as np
from qutip import (
    qeye, num, destroy, create, QobjEvo, Qobj,
    basis, rand_herm, fock_dm, liouvillian, operator_to_vector
)
from qutip.solver.sode.ssystem import *
from qutip.solver.sode.ssystem import SimpleStochasticSystem, StochasticClosedSystem
import qutip.core.data as _data
import pytest
from itertools import product


def L0(system, f):
    def _func(t, rho, dt=1e-6):
        n = rho.shape[0]
        f0 = f(t, rho)

        out = (f(t+dt, rho) - f0) / dt

        jac = np.zeros((n, n), dtype=complex)
        for i in range(n):
            dxi = basis(n, i).data
            jac[:, i] = (f(t, rho + dt * dxi) - f0).to_array().flatten() / dt
        out = out + _data.Dense(jac) @ system.drift(t, rho)

        for i, j in product(range(n), repeat=2):
            dxi = basis(n, i).data
            dxj = basis(n, j).data
            sec = f(t, (rho + dxi * dt + dxj * dt))
            sec = sec - f(t, (rho + dxj * dt))
            sec = sec - f(t, (rho + dxi * dt))
            sec = sec + f0
            sec = sec / dt / dt * 0.5
            for k in range(system.num_collapse):
                out = out + (
                    sec
                    * _data.inner(dxi, system.diffusion(t, rho)[k])
                    * _data.inner(dxj, system.diffusion(t, rho)[k])
                )
        return out
    return _func


def L(system, ii, f):
    def _func(t, rho, dt=1e-6):
        n = rho.shape[0]
        jac = np.zeros((n, n), dtype=complex)
        f0 = f(t, rho)
        for i in range(n):
            dxi = basis(n, i).data
            jac[:, i] = (f(t, (rho + dt * dxi)) - f0).to_array().flatten()
        return _data.Dense(jac) @ system.diffusion(t, rho)[ii] / dt
    return _func


def LL(system, ii, jj, f):
    # Can be implemented as 2 calls of ``L``, but would use 2 ``dt`` which
    # cannot be different.
    def _func(t, rho, dt=1e-6):
        f0 = f(t, rho)
        bi = system.diffusion(t, rho)[ii]
        bj = system.diffusion(t, rho)[jj]
        out = rho *0.
        n = rho.shape[0]

        for i, j in product(range(n), repeat=2):
            dxi = basis(n, i, dtype="Dense").data
            dxj = basis(n, j, dtype="Dense").data
            sec = f(t, (rho + dxi * dt + dxj * dt))
            sec = sec - f(t, (rho + dxj * dt))
            sec = sec - f(t, (rho + dxi * dt))
            sec = sec + f0
            sec = sec / dt / dt

            out = out + (
                sec * _data.inner(dxi, bi) * _data.inner(dxj, bj)
            )
            df = (f(t, (rho + dxj * dt)) - f0) / dt
            db = (
                system.diffusion(t, (rho + dxi * dt))[jj]
                - system.diffusion(t, rho)[jj]
            ) / dt

            out = out + (
                df * _data.inner(dxi, bi) * _data.inner(dxj, db)
            )

        return out
    return _func


def _check_equivalence(f, target, args):
    """
    Check that the error is proportional to `dt`.
    """
    dts = np.logspace(-4, -1, 7)
    errors_dt = np.array([
        _data.norm.l2(f(*args, dt=dt) - target)
        for dt in dts
    ])
    if np.all(errors_dt < 1e-6):
        return

    power = np.polyfit(np.log(dts), np.log(errors_dt + 1e-16), 1)[0]
    # Sometime the dt term is cancelled and the dt**2 term is dominant
    assert power > 0.9


def _run_derr_check(solver, state):
    """
    Compare each derrivatives to the finite differences equivalent.
    """
    t = 0
    N = solver.num_collapse
    a = solver.drift
    solver.set_state(t, state)

    assert _data.norm.l2(solver.drift(t, state) - solver.a()) < 1e-6
    for i in range(N):
        b = lambda *args: solver.diffusion(*args)[i]
        assert b(t, state) == solver.bi(i)
        for j in range(N):
            _check_equivalence(
                L(solver, j, b), solver.Libj(j, i), (t, state)
            )

    _check_equivalence(L0(solver, a), solver.L0a(), (t, state))

    for i in range(N):
        b = lambda *args: solver.diffusion(*args)[i]
        _check_equivalence(L0(solver, b), solver.L0bi(i), (t, state))
        _check_equivalence(L(solver, i, a), solver.Lia(i), (t, state))

        for j in range(i, N):
            for k in range(j, N):
                _check_equivalence(
                    LL(solver, k, j, b), solver.LiLjbk(k, j, i), (t, state)
                )


def _make_oper(kind, N):
    if kind == "qeye":
        out = qeye(N)
    elif kind == "destroy":
        out = destroy(N)
    elif kind == "destroy2":
        out = destroy(N)**2
    elif kind == "tridiag":
        out = destroy(N) + num(N) + create(N)
    elif kind == "td":
        out = [num(N), [destroy(N) + create(N), lambda t: 1 + t]]
    elif kind == "rand":
        out = rand_herm(N)
    return QobjEvo(out)


@pytest.mark.parametrize(['H', 'sc_ops'], [
    pytest.param("qeye", ["destroy"], id='simple'),
    pytest.param("tridiag", ["destroy"], id='simple'),
    pytest.param("qeye", ["destroy", "destroy2"], id='2 c_ops'),
    pytest.param("td", ["destroy"], id='H td'),
    pytest.param("qeye", ["td"], id='c_ops td'),
    pytest.param("rand", ["rand"], id='random'),
])
@pytest.mark.parametrize('heterodyne', [False, True])
def test_open_system_derr(H, sc_ops, heterodyne):
    N = 5
    H = _make_oper(H, N)
    sc_ops = [_make_oper(op, N) for op in sc_ops]
    if heterodyne:
        new_sc_ops = []
        for c_op in sc_ops:
            new_sc_ops.append(c_op / np.sqrt(2))
            new_sc_ops.append(c_op * (-1j / np.sqrt(2)))
        sc_ops = new_sc_ops

    system = StochasticOpenSystem(H, sc_ops)
    state = operator_to_vector(fock_dm(N, N-2, dtype="Dense")).data
    _run_derr_check(system, state)
