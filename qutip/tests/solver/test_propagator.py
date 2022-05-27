import numpy as np
from qutip import (destroy, propagator, Propagator, propagator_steadystate,
                   steadystate, tensor, qeye, basis, QobjEvo, sesolve)
from qutip.solver.options import SolverOptions


def testPropHOB():
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H, 1)
    U2 = (-1j * H).expm()
    assert (U - U2).norm('max') < 1e-4


def testPropObj():
    opt = SolverOptions(method='dop853')
    a = destroy(5)
    H = a.dag()*a
    U = Propagator(H, [a], options=opt, memoize=5, tol=1e-5)
    # Few call to fill the stored propagators.
    U(0.5), U(0.25), U(0.75), U(1), U(-1), U(-.5)
    assert len(U.times) == 5
    assert (U(1) - propagator(H, 1, [a])).norm('max') < 1e-4
    assert (U(0.5) - propagator(H, 0.5, [a])).norm('max') < 1e-4
    assert (U(1.5, 0.5) - propagator(H, 1, [a])).norm('max') < 1e-4
    # Within tol, should use the precomupted value at U(0.5)
    assert (U(0.5) - U(0.5 + 1e-6)).norm('max') < 1e-10


def func(t):
    return np.cos(t)


def testPropHOTd():
    "Propagator: func td format"
    a = destroy(5)
    H = a.dag()*a
    Htd = [H, [H, func]]
    U = propagator(Htd, 1)
    ts = np.linspace(0, 1, 101)
    U2 = (-1j * H * np.trapz(1 + func(ts), ts)).expm()
    assert (U - U2).norm('max') < 1e-4


def testPropObjTd():
    a = destroy(5)
    H = a.dag()*a
    U = Propagator([H, [H, "w*t"]], [a], args={'w': 1})
    assert (
        U(1) - propagator([H, [H, "w*t"]], 1, [a], args={'w': 1})
    ).norm('max') < 1e-4
    assert (
        U(0.5, w=2) - propagator([H, [H, "w*t"]], 0.5, [a], args={'w': 2})
    ).norm('max') < 1e-4
    assert (
        U(1.5, 0.5, w=1.5)
        - propagator([H, [H, "w*t"]], [0.5, 1.5], [a], args={'w': 1.5})[1]
    ).norm('max') < 1e-4


def testPropHOSteady():
    "Propagator: steady state"
    a = destroy(5)
    H = a.dag()*a
    c_op_list = []
    kappa = 0.1
    n_th = 2
    rate = kappa * (1 + n_th)
    c_op_list.append(np.sqrt(rate) * a)
    rate = kappa * n_th
    c_op_list.append(np.sqrt(rate) * a.dag())
    U = propagator(H, 2*np.pi, c_op_list)
    rho_prop = propagator_steadystate(U)
    rho_ss = steadystate(H, c_op_list)
    assert (rho_prop - rho_ss).norm('max') < 1e-4


def testPropHDims():
    "Propagator: preserve H dims (unitary_mode='single', parallel=False)"
    H = tensor([qeye(2), qeye(2)])
    U = propagator(H, 1)
    assert U.dims == H.dims


def testPropEvo():
    a = destroy(5)
    H = a.dag()*a
    U = Propagator([H, [a + a.dag(), "w*t"]], args={'w': 1})
    psi = QobjEvo(U) @ basis(5, 4)
    tlist = np.linspace(0, 1, 6)
    psi_expected = sesolve(
        [H, [a + a.dag(), "w*t"]], basis(5,4), tlist=tlist, args={'w': 1}
    ).states
    for t, psi_t in zip(tlist, psi_expected):
        assert abs(psi(t).overlap(psi_t)) > 1-1e-6
