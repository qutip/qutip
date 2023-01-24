import numpy

def make_1_step(H, psi, sc_ops, solver, dt, dw=False):
    ssolver = qt.smesolve(H, psi, [0,dt], sc_ops=c_ops, _dry_run=1, solver=solver, nsubsteps=1)
    state = qt.operator_to_vector(ssolver.sso.state0).full().flatten()
    func = getattr(ssolver, solver)
    out = np.zeros_like(state)
    if dw is False:
        dw = np.array([1.] * 2 * len(sc_ops)) * dt**0.5
    func(0, dt, dw, state, out)
    return qt.Qobj( qt.unstack_columns(out))
