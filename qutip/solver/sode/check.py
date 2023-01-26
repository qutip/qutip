import numpy as np

def make_1_step(H, psi, sc_ops, solver, dt, dw=False):
    ssolver = qt.smesolve(H, psi, [0,dt], sc_ops=c_ops, _dry_run=1, solver=solver, nsubsteps=1)
    state = qt.operator_to_vector(ssolver.sso.state0).full().flatten()
    func = getattr(ssolver, solver)
    out = np.zeros_like(state)
    if dw is False:
        dw = np.array([1.] * 2 * len(sc_ops)) * dt**0.5
    func(0, dt, dw, state, out)
    return qt.Qobj( qt.unstack_columns(out))


def L0(system, f):
    def _func(t, rho, dt=1e-6):
        n = rho.shape[0]
        f0 = f(t, rho.data)

        out = (f(t+dt, rho) - f0) / dt

        jac = np.zeros((n, n), dtype=complex)
        for i in range(n):
            dxi = qt.basis(n, i).data
            jac[:, i] = (f(t, rho + dt * dxi) - f0).to_array().flatten() / dt
        out = out + qt.data.Dense(jac) @ system.drift(t, rho)

        for i, j in product(range(n), repeat=2):
            dxi = qt.basis(n, i).data
            dxj = qt.basis(n, j).data
            sec = f(t, (rho + dxi * dt + dxj * dt))
            sec = sec - f(t, (rho + dxj * dt))
            sec = sec - f(t, (rho + dxi * dt))
            sec = sec + f0
            sec = sec / dt / dt * 0.5
            for k in range(system.num_collapse):
                out = out + (
                    sec
                    * qt.data.inner(dxi, system.diffusion(t, rho)[k])
                    * qt.data.inner(dxj, system.diffusion(t, rho)[k])
                )
        return out
    return _func


def Li(system, ii, f):
    def _func(t, rho, dt=1e-6):
        n = rho.shape[0]
        jac = np.zeros((n, n), dtype=complex)
        f0 = f(t, rho)
        for i in range(n):
            dxi = qt.basis(n, i).data
            jac[:, i] = (f(t, (rho + dt * dxi)) - f0).to_array().flatten()
        return qt.data.Dense(jac) @ system.diffusion(t, rho)[ii] / dt
    return _func


def _check_equivalence(f, target, shape):
    ...
