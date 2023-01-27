import numpy as np
from itertools import product
from qutip.core import data as _data
import qutip

def make_1_step(H, psi, sc_ops, solver, dt, dw=False):
    ssolver = qutip.smesolve(H, psi, [0,dt], sc_ops=c_ops, _dry_run=1, solver=solver, nsubsteps=1)
    state = qutip.operator_to_vector(ssolver.sso.state0).full().flatten()
    func = getattr(ssolver, solver)
    out = np.zeros_like(state)
    if dw is False:
        dw = np.array([1.] * 2 * len(sc_ops)) * dt**0.5
    func(0, dt, dw, state, out)
    return qutip.Qobj( qutip.unstack_columns(out))


def L0(system, f):
    def _func(t, rho, dt=1e-6):
        n = rho.shape[0]
        f0 = f(t, rho)

        out = (f(t+dt, rho) - f0) / dt

        jac = np.zeros((n, n), dtype=complex)
        for i in range(n):
            dxi = qutip.basis(n, i).data
            jac[:, i] = (f(t, rho + dt * dxi) - f0).to_array().flatten() / dt
        out = out + qutip.data.Dense(jac) @ system.drift(t, rho)

        for i, j in product(range(n), repeat=2):
            dxi = qutip.basis(n, i).data
            dxj = qutip.basis(n, j).data
            sec = f(t, (rho + dxi * dt + dxj * dt))
            sec = sec - f(t, (rho + dxj * dt))
            sec = sec - f(t, (rho + dxi * dt))
            sec = sec + f0
            sec = sec / dt / dt * 0.5
            for k in range(system.num_collapse):
                out = out + (
                    sec
                    * qutip.data.inner(dxi, system.diffusion(t, rho)[k])
                    * qutip.data.inner(dxj, system.diffusion(t, rho)[k])
                )
        return out
    return _func


def L(system, ii, f):
    def _func(t, rho, dt=1e-6):
        n = rho.shape[0]
        jac = np.zeros((n, n), dtype=complex)
        f0 = f(t, rho)
        for i in range(n):
            dxi = qutip.basis(n, i).data
            jac[:, i] = (f(t, (rho + dt * dxi)) - f0).to_array().flatten()
        return qutip.data.Dense(jac) @ system.diffusion(t, rho)[ii] / dt
    return _func


def LL(system, ii, jj, f):
    # Can be implemented as 2 calls of ``L``, but this use 2 ``dt`` which
    # cannot be different.
    def _func(t, rho, dt=1e-6):
        f0 = f(t, rho)
        bi = system.diffusion(t, rho)[ii]
        bj = system.diffusion(t, rho)[jj]
        out = rho *0.
        n = rho.shape[0]

        for i, j in product(range(n), repeat=2):
            dxi = qutip.basis(n, i, dtype="Dense").data
            dxj = qutip.basis(n, j, dtype="Dense").data
            sec = f(t, (rho + dxi * dt + dxj * dt))
            sec = sec - f(t, (rho + dxj * dt))
            sec = sec - f(t, (rho + dxi * dt))
            sec = sec + f0
            sec = sec / dt / dt

            out = out + (
                sec * qutip.data.inner(dxi, bi) * qutip.data.inner(dxj, bj)
            )
            df = (f(t, (rho + dxj * dt)) - f0) / dt
            db = (system.diffusion(t, (rho + dxi * dt))[jj] - system.diffusion(t, rho)[jj] ) / dt

            out = out + (
                df * qutip.data.inner(dxi, bi) * qutip.data.inner(dxj, db)
            )

        return out
    return _func


def _check_equivalence(f, target, args):
    dts = np.logspace(-5, -1, 9)
    errors_dt = [
        _data.norm.l2(f(*args, dt=dt) - target) / dt
        for dt in dts
    ]
    poly = np.polyfit(dts, errors_dt, 1)
    return -1 < poly[0] < 1


def run_derr_check(solver, t, state):
    N = solver.num_collapse
    a = solver.drift
    solver.set_state(t, state)

    assert solver.drift(t, state) == solver.a()
    assert _check_equivalence(L0(solver, a), solver.L0a(), (t, state))

    for i in range(N):
        b = lambda *args: solver.diffusion(*args)[i]
        assert b(t, state) == solver.bi(i)
        assert _check_equivalence(L0(solver, b), solver.L0bi(i), (t, state))
        assert _check_equivalence(L(solver, i, a), solver.Lia(i), (t, state))

        for j in range(N):
            assert _check_equivalence(
                L(solver, j, b), solver.Libj(j, i), (t, state)
            )

            for k in range(N):
                assert _check_equivalence(
                    LL(solver, k, j, b), solver.LiLjbk(k, j, i), (t, state)
                )
