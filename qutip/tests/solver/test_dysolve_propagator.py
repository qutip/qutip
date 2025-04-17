from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip.solver import propagator
from qutip.solver.cy.dysolve import cy_compute_integrals
from qutip import sigmax, sigmay, sigmaz, qeye, qeye_like, tensor, CoreOptions
from scipy.special import factorial
import numpy as np
import pytest


@pytest.fixture(scope='module')
def empty_instance():
    return DysolvePropagator.__new__(DysolvePropagator)


@pytest.mark.parametrize("eff_omega", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("dt", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("ws, answer", [
    # First part of tuple is "ws", second part is "answer"
    (
        np.array([0.0]),
        lambda _, dt: dt
    ),
    (
        np.array([1e-12]),
        lambda _, dt: dt
    ),
    (
        lambda eff_omega: np.array([eff_omega]),
        lambda eff_omega, dt: (-1j/eff_omega) * (np.exp(1j*eff_omega*dt) - 1)
    ),
    (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        lambda _, dt: (dt**5) / factorial(5)
    ),
    (
        np.array([1e-12, 1e-12, 1e-12]),
        lambda _, dt: (dt**3) / factorial(3)
    ),
    (
        lambda eff_omega: np.array([eff_omega, 0.0]),
        lambda eff_omega, dt: (-1j/eff_omega) * (
            (-1j/eff_omega) * (np.exp(1j*eff_omega*dt) - 1) - dt
        )
    ),
    (
        lambda eff_omega: np.array([0.0, eff_omega]),
        lambda eff_omega, dt: (-1j*dt/eff_omega) * np.exp(1j*eff_omega*dt) -
        ((1j/eff_omega)**2) * (np.exp(1j*eff_omega*dt)-1))
])
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
        answer = (-1j*dt/eff_omega_1) + \
            (np.exp(1j*eff_omega_2*dt)-1)/(eff_omega_1*eff_omega_2)
    else:
        exp_1 = np.exp(1j*(eff_omega_1+eff_omega_2)*dt)
        exp_2 = np.exp(1j*eff_omega_2*dt)
        answer = -(exp_1-1)/(eff_omega_1*(eff_omega_1+eff_omega_2)) + \
            (exp_2-1)/(eff_omega_1*eff_omega_2)

    assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("max_order, X, answer", [
    (
        1,
        sigmay(),
        np.array([0, -1j, 1j, 0])
    ),
    (
        2,
        sigmay(),
        np.array([0, 0, 1, 0, 0, 1, 0, 0])
    ),
    (
        3,
        sigmay(),
        np.array([0, 0, 0, 0, 0, -1j, 0, 0, 0, 0, 1j, 0, 0, 0, 0, 0])
    ),
    (
        1,
        sigmax(),
        np.array([0, 1, 1, 0])
    ),
    (
        2,
        sigmax(),
        np.array([0, 0, 1, 0, 0, 1, 0, 0])
    ),
    (
        3,
        sigmax(),
        np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    ),
    (
        1,
        sigmaz(),
        np.array([1, 0, 0, -1])
    ),
    (
        2,
        sigmaz(),
        np.array([1, 0, 0, 0, 0, 0, 0, 1])
    ),
    (
        3,
        sigmaz(),
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    ),
    (
        1,
        qeye(2),
        np.array([1, 0, 0, 1])
    ),
    (
        2,
        qeye(2),
        np.array([1, 0, 0, 0, 0, 0, 0, 1])
    ),
    (
        3,
        qeye(2),
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    ),
    (
        1,
        qeye(3),
        np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    ),
    (
        2,
        qeye(3),
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1])
    )
])
def test_matrix_elements(empty_instance, max_order, X, answer):
    dysolve = empty_instance
    dysolve._X = X
    # The basis shouldn't matter
    dysolve._basis = qeye_like(X)
    current_matrix_elements = None
    dysolve._elems = dysolve._X.transform(dysolve._basis).full().flatten()

    for _ in range(1, max_order + 1):
        current_matrix_elements = dysolve._update_matrix_elements(
            current_matrix_elements
        )
    assert np.array_equal(current_matrix_elements, answer)


@pytest.mark.parametrize("H_0", [
    sigmaz(), sigmay(), sigmaz(), qeye(2), tensor(sigmax(), sigmaz()),
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay())
])
@pytest.mark.parametrize("t_i, t_f", [
    (0, 0.1), (0, 0.5), (0, 1), (0, 10), (0, -1),
    (-0.1, 0.1), (-0.5, 0.5), (-1, 1), (-10, 10), (1, -1)
])
def test_zeroth_order(H_0, t_i, t_f):
    # self.X and self.omega don't matter
    dysolve = DysolvePropagator(
        H_0, qeye_like(H_0), 0, options={'max_order': 0}
    )
    U = dysolve(t_f, t_i)

    exp = (-1j*H_0*(t_f - t_i)).expm()

    with CoreOptions(atol=1e-10, rtol=1e-10):
        assert U == exp


@pytest.mark.parametrize("H_0", [sigmax(), sigmay(), sigmaz()])
@pytest.mark.parametrize("X", [sigmax(), sigmay(), sigmaz()])
@pytest.mark.parametrize("t", [-0.15, -0.1, 0, 0.1, 0.15])
@pytest.mark.parametrize("omega", [0, 1, 10])
def test_2x2_propagators_single_time(H_0, X, t, omega):
    # Dysolve
    options = {'max_order': 3, 'max_dt': 0.05}
    U = dysolve_propagator(H_0, X, omega, t, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    prop = propagator(
        H, t, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-6):
        assert U == prop


@pytest.mark.parametrize("H_0", [sigmay(), sigmaz()])
@pytest.mark.parametrize("X", [sigmay(), sigmaz()])
@pytest.mark.parametrize("ts", [
    [0, 0.25, 0.5],
    [0, -0.25, -0.5],
    [-0.1, 0, 0.1]
])
@pytest.mark.parametrize("omega", [0, 10])
def test_2x2_propagators_list_times(H_0, X, ts, omega):
    options = {'max_order': 3, 'max_dt': 0.01}
    Us = dysolve_propagator(H_0, X, omega, ts, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    props = propagator(
        H, ts, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-6):
        assert Us == props


@pytest.mark.parametrize("H_0", [
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay()),
    tensor(sigmaz(), qeye(2))
])
@pytest.mark.parametrize("X", [
    tensor(qeye(2), sigmaz()),
    tensor(sigmaz(), sigmax()) + tensor(sigmay(), qeye(2))
])
@pytest.mark.parametrize("omega", [
    5, 10
])
@pytest.mark.parametrize("t_f", [
    1, -1
])
def test_4x4_propagators_single_time(H_0, X, omega, t_f):
    options = {'max_order': 3, 'max_dt': 0.01}
    U = dysolve_propagator(H_0, X, omega, t_f, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    prop = propagator(
        H, t_f, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-5):
        assert U == prop


@pytest.mark.parametrize("H_0", [
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay()),
    tensor(sigmaz(), qeye(2))
])
@pytest.mark.parametrize("X", [
    tensor(qeye(2), sigmaz()),
    tensor(sigmaz(), sigmax()) + tensor(sigmay(), qeye(2))
])
@pytest.mark.parametrize("omega", [
    0, 10
])
@pytest.mark.parametrize("ts", [
    [0, 0.25, 0.5],
    [0, -0.25, -0.5],
    [-0.1, 0, 0.1]
])
def test_4x4_propagators_list_times(H_0, X, omega, ts):
    options = {'max_order': 3, 'max_dt': 0.01}
    Us = dysolve_propagator(H_0, X, omega, ts, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    props = propagator(
        H, ts, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-6):
        assert Us == props


@pytest.mark.parametrize("H_0, X", [
    (
        sigmaz(), sigmax(),
    ),
    (
        tensor(sigmaz(), sigmaz()), tensor(sigmax(), sigmax()),
    ),
    (
        tensor(sigmaz(), sigmaz(), sigmaz()),
        tensor(sigmax(), sigmax(), sigmax())
    ),
    (
        tensor(sigmaz(), sigmaz(), sigmaz(), sigmaz()),
        tensor(sigmax(), sigmax(), sigmax(), sigmax())
    )
])
def test_dims(H_0, X):
    dysolve = DysolvePropagator(
        H_0, X, 1, {'max_order': 0}
    )
    U = dysolve(0.001)
    assert (dysolve._H_0.dims == dysolve._X.dims == U.dims)
