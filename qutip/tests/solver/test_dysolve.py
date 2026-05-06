from qutip.solver.dysolve import Dysolve, dysolve_propagator, dysolve
from qutip.solver import propagator, sesolve
from qutip.solver.cy.dysolve import cy_compute_integrals
from qutip import (
    sigmax,
    sigmay,
    sigmaz,
    qeye,
    qeye_like,
    tensor,
    enr_destroy,
    destroy,
    CoreOptions,
    rand_herm,
    coefficient,
    Coefficient,
    QobjEvo,
    Qobj,
    num,
)
from scipy.special import factorial
import numpy as np
import pytest


@pytest.mark.parametrize("eff_omega", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("dt", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
@pytest.mark.parametrize(
    "ws, answer",
    [
        # First part of tuple is "ws", second part is "answer"
        (np.array([0.0]), lambda _, dt: dt),
        (np.array([1e-12]), lambda _, dt: dt),
        (
            lambda eff_omega: np.array([eff_omega]),
            lambda eff_omega, dt: (-1j / eff_omega)
            * (np.exp(1j * eff_omega * dt) - 1),
        ),
        (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            lambda _, dt: (dt**5) / factorial(5),
        ),
        (
            np.array([1e-12, 1e-12, 1e-12]),
            lambda _, dt: (dt**3) / factorial(3),
        ),
        (
            lambda eff_omega: np.array([eff_omega, 0.0]),
            lambda eff_omega, dt: (-1j / eff_omega)
            * ((-1j / eff_omega) * (np.exp(1j * eff_omega * dt) - 1) - dt),
        ),
        (
            lambda eff_omega: np.array([0.0, eff_omega]),
            lambda eff_omega, dt: (-1j * dt / eff_omega)
            * np.exp(1j * eff_omega * dt)
            - ((1j / eff_omega) ** 2) * (np.exp(1j * eff_omega * dt) - 1),
        ),
    ],
)
def test_integrals_1(eff_omega, dt, ws, answer):
    if callable(ws):
        ws = ws(eff_omega)
    if callable(answer):
        answer = answer(eff_omega, dt)

    integrals = cy_compute_integrals(ws, dt)

    assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("eff_omega_1", [-25.0, -5.0, -0.5, 0.5, 5.0, 25.0])
@pytest.mark.parametrize("eff_omega_2", [-25.0, -5.0, -0.5, 0.5, 5.0, 25.0])
@pytest.mark.parametrize("dt", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
def test_integrals_2(eff_omega_1, eff_omega_2, dt):
    ws = np.array([eff_omega_1, eff_omega_2])
    integrals = cy_compute_integrals(ws, dt)

    if eff_omega_1 + eff_omega_2 == 0:
        answer = (-1j * dt / eff_omega_1) + (
            np.exp(1j * eff_omega_2 * dt) - 1
        ) / (eff_omega_1 * eff_omega_2)
    else:
        exp_1 = np.exp(1j * (eff_omega_1 + eff_omega_2) * dt)
        exp_2 = np.exp(1j * eff_omega_2 * dt)
        answer = -(exp_1 - 1) / (eff_omega_1 * (eff_omega_1 + eff_omega_2)) + (
            exp_2 - 1
        ) / (eff_omega_1 * eff_omega_2)

    assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("H_0", [
    sigmay(),
    tensor(sigmax(), sigmaz()),
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay()),
])
@pytest.mark.parametrize("t_i, t_f", [
    (0, 0.1),
    (0, -1),
    (-0.5, 0.5),
    (1, -1),
])
def test_zeroth_order(H_0, t_i, t_f):
    # self.X and self.omega don't matter
    dysolve = Dysolve(H_0, [], options={"max_order": 0})
    U = dysolve.propagator(t_f, t_i)

    exp = (-1j * H_0 * (t_f - t_i)).expm()

    with CoreOptions(atol=1e-10, rtol=1e-10):
        assert U == exp


def _drive2QobjEvo(drive):
    if not isinstance(drive, dysolve.Drive):
        drive = dysolve.Drive(*drive)
    oper, w, form, coeff = drive
    funcs = {
        "cos": lambda t: np.cos(w * t),
        "sin": lambda t: np.sin(w * t),
        "exp": lambda t: np.exp(1j * w * t),
    }
    drive_func = coefficient(funcs[form])
    if isinstance(coeff, Coefficient):
        drive_func = drive_func * coeff
    elif coeff is not None:
        oper = oper * coeff
    return QobjEvo([oper, drive_func])


@pytest.mark.parametrize("H_0", [sigmax(), sigmaz()])
@pytest.mark.parametrize("X", [sigmay(), sigmaz()])
@pytest.mark.parametrize("t", [
    -0.1, -0.075, -0.025, 0, 0.075, 0.15,
    [0, 0.25, 0.5], [0, -0.25, -0.5], [-0.1, 0, 0.1]
])
@pytest.mark.parametrize("omega", [0, 1, 2, 10])
def test_2x2_propagators(H_0, X, t, omega):
    # Dysolve
    options = {"max_order": 3, "max_dt": 0.05}
    drive = (X, omega)
    U = dysolve_propagator(H_0, [drive], t, options=options)

    H = H_0 + _drive2QobjEvo(drive)
    prop = propagator(H, t, options={"atol": 1e-10, "rtol": 1e-8})

    with CoreOptions(atol=1e-10, rtol=1e-6):
        assert U == prop

    if isinstance(U, Qobj):
        assert H_0._dims == U._dims


@pytest.mark.parametrize("H_0", [
    tensor(sigmaz(), qeye(2)),
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay()),
])
@pytest.mark.parametrize("X", [
    tensor(sigmaz(), qeye(2)),
    tensor(sigmaz(), sigmax()) + tensor(sigmay(), qeye(2)),
])
@pytest.mark.parametrize("omega", [1, 2, 10])
@pytest.mark.parametrize("t_f", [1, -1])
def test_4x4_propagators(H_0, X, omega, t_f):
    options = {"max_order": 3, "max_dt": 0.01}
    drive = (X, omega)
    U = dysolve_propagator(H_0, [drive], t_f, options=options)

    H = H_0 + _drive2QobjEvo(drive)
    prop = propagator(H, t_f, options={"atol": 1e-10, "rtol": 1e-8})

    with CoreOptions(atol=1e-10, rtol=1e-5):
        assert U == prop

    assert H_0._dims == U._dims


@pytest.mark.parametrize("omega", [0, 0.5, 1, 2, 100])
@pytest.mark.parametrize("t_f", [1, -1])
def test_enr_propagators(omega, t_f):
    a, b = enr_destroy([2, 2], 1)
    X = (a + a.dag()) @ (b + b.dag())
    H_0 = (a.dag() @ a) + (b.dag() @ b)
    test_4x4_propagators(H_0, X, omega, t_f)


@pytest.fixture(params=[
    pytest.param(lambda: num(10), id='diag'),
    pytest.param(lambda: rand_herm(10, density=0.7), id='non-diag'),
])
def H0(request):
    return request.param


@pytest.mark.parametrize("formats", [
    ["cos"], ["sin"], ["sin", "cos"], ["exp", "exp"]
])
def test_drive_formats(H0, formats):
    if len(formats) == 1:
        drives = [(rand_herm(10), 100, formats[0])]
    elif formats[0] != "exp":
        drives = [
            (rand_herm(10), 100, formats[0]),
            (rand_herm(10), -200, formats[1]),
        ]
    else:
        op = rand_herm(10)
        drives = [(op, 100, "exp"), (op, -100, "exp")]

    H = H0()
    for drive in drives:
        H += _drive2QobjEvo(drive)

    dy_instance = Dysolve(H0(), drives)

    for t in [-0.1, 0.1]:
        prop = propagator(H, t, options={"atol": 1e-10, "rtol": 1e-8})
        dy_prop = dy_instance.propagator(t)
        assert dy_prop == prop


def test_envelopes(H0):
    args = {"A": 1.}
    a = destroy(10)
    coeff = coefficient(lambda t, A: A*t, args=args)
    drives = [dysolve.Drive(a + a.dag(), 100, envelope=coeff)]
    dy_instance = Dysolve(H0(), drives, options={"step_size": 0.001})

    H = H0()
    for drive in drives:
        H += _drive2QobjEvo(drive)

    prop = propagator(H, 0.1, options={"atol": 1e-10, "rtol": 1e-8})
    dy_prop = dy_instance.propagator(0.1)
    assert dy_prop == prop

    args = {"A": 5.}
    prop = propagator(H, 0.1, args=args, options={"atol": 1e-10, "rtol": 1e-8})
    dy_prop = dy_instance.propagator(0.1, args=args)
    assert dy_prop == prop
