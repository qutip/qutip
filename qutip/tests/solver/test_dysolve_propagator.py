from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip.solver import propagator
from qutip import Qobj, sigmax, sigmay, sigmaz, qeye, qeye_like, tensor, CoreOptions
from scipy.special import factorial
import numpy as np
import itertools
import pytest


@pytest.fixture(scope='module')
def empty_instance():
    return DysolvePropagator.__new__(DysolvePropagator)


@pytest.mark.parametrize("eff_omega", [-10, -1, -0.1, 0.1, 1, 10])
@pytest.mark.parametrize("dt", [-10, -1, -0.1, 0.1, 1, 10])
@pytest.mark.parametrize("ws, answer", [
    # First part of tuple is "ws", second part is "answer"
    (
        np.array([]),
        1
    ),
    (
        np.array([0]),
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
        np.array([0, 0, 0, 0, 0]),
        lambda _, dt: (dt**5) / factorial(5)
    ),
    (
        np.array([1e-12, 1e-12, 1e-12]),
        lambda _, dt: (dt**3) / factorial(3)
    ),
    (
        lambda eff_omega: np.array([eff_omega, 0]),
        lambda eff_omega, dt: (-1j/eff_omega) * (
            (-1j/eff_omega) * (np.exp(1j*eff_omega*dt) - 1) - dt
        )
    ),
    (
        lambda eff_omega: np.array([0, eff_omega]),
        lambda eff_omega, dt: (-1j*dt/eff_omega) * np.exp(1j*eff_omega*dt) -
        ((1j/eff_omega)**2) * (np.exp(1j*eff_omega*dt)-1))
])
def test_integrals_1(empty_instance, eff_omega, dt, ws, answer):
    # Create instance only with the required data
    dysolve = empty_instance
    dysolve.a_tol = 1e-10
    dysolve.dt = dt

    if callable(ws):
        ws = ws(eff_omega)
    if callable(answer):
        answer = answer(eff_omega, dt)

    integrals = dysolve._compute_integrals(ws)

    assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("eff_omega_1", [-25, -5, -0.5, 0.5, 5, 25])
@pytest.mark.parametrize("eff_omega_2", [-25, -5, -0.5, 0.5, 5, 25])
@pytest.mark.parametrize("dt", [-10, -1, -0.1, 0.1, 1, 10])
def test_integrals_2(empty_instance, eff_omega_1, eff_omega_2, dt):
    # Create instance only with the required data
    dysolve = empty_instance
    dysolve.a_tol = 1e-10
    dysolve.dt = dt

    ws = [eff_omega_1, eff_omega_2]
    integrals = dysolve._compute_integrals(ws)

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
        np.array([[0], [-1j], [1j], [0]])
    ),
    (
        2,
        sigmay(),
        np.array([[0], [0], [1], [0], [0], [1], [0], [0]])
    ),
    (
        3,
        sigmay(),
        np.array([[0], [0], [0], [0], [0], [-1j], [0], [0],
                  [0], [0], [1j], [0], [0], [0], [0], [0]])
    ),
    (
        1,
        sigmax(),
        np.array([[0], [1], [1], [0]])
    ),
    (
        2,
        sigmax(),
        np.array([[0], [0], [1], [0], [0], [1], [0], [0]])
    ),
    (
        3,
        sigmax(),
        np.array([[0], [0], [0], [0], [0], [1], [0], [0],
                  [0], [0], [1], [0], [0], [0], [0], [0]])
    ),
    (
        1,
        sigmaz(),
        np.array([[1], [0], [0], [-1]])
    ),
    (
        2,
        sigmaz(),
        np.array([[1], [0], [0], [0], [0], [0], [0], [1]])
    ),
    (
        3,
        sigmaz(),
        np.array([[1], [0], [0], [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0], [0], [0], [-1]])
    ),
    (
        1,
        qeye(2),
        np.array([[1], [0], [0], [1]])
    ),
    (
        2,
        qeye(2),
        np.array([[1], [0], [0], [0], [0], [0], [0], [1]])
    ),
    (
        3,
        qeye(2),
        np.array([[1], [0], [0], [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0], [0], [0], [1]])
    ),
    (
        1,
        qeye(3),
        np.array([[1], [0], [0], [0], [1], [0], [0], [0], [1]])
    ),
    (
        2,
        qeye(3),
        np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0],
                  [0], [0], [0], [0], [1], [0], [0], [0], [0],
                  [0], [0], [0], [0], [0], [0], [0], [0], [1]])
    )
])
def test_matrix_elements_2(empty_instance, max_order, X, answer):
    dysolve = empty_instance
    dysolve.X = X
    current_matrix_elements = None

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
    dysolve(t_f, t_i)
    U = dysolve.U

    _, basis = H_0.eigenstates()
    exp = (-1j*H_0*(t_f - t_i)).expm().transform(basis)

    with CoreOptions(atol=1e-10, rtol=1e-10):
        assert U == exp


# def test_number_of_propagators():
#     # Single time
#     dysolve_1 = dysolve_propagator(
#         sigmaz(), sigmax(), 1, 0.1, {'max_order': 1, "a_tol": 1e-8}
#     )
#     assert isinstance(dysolve_1, Qobj)

#     # Multiple times
#     times = [0, 0.1, 0.2, 0.3]
#     dysolve_2 = dysolve_propagator(sigmaz(), sigmax(), 1, times)
#     assert len(dysolve_2) == len(times)


# def test_dims():
#     # Dimensions of H_0, X, and propagators shoud be the same
#     H_0s = [sigmaz(), tensor(sigmaz(), sigmaz()),
#             tensor(sigmaz(), sigmaz(), sigmaz()),
#             tensor(sigmaz(), sigmaz(), sigmaz(), sigmaz())]
#     Xs = [sigmax(), tensor(sigmax(), sigmax()),
#           tensor(sigmax(), sigmax(), sigmax()),
#           tensor(sigmax(), sigmax(), sigmax(), sigmax())]

#     for H_0, X in zip(H_0s, Xs):
#         dysolve_1 = DysolvePropagator(
#             H_0, X, 1, {'max_order': 1, 'a_tol': 1e-8}
#         )
#         U_1 = dysolve_1(0, 1)
#         assert (dysolve_1.H_0.dims == dysolve_1.X.dims == U_1.dims)


# #@pytest.mark.xfail()
# def test_2x2_propagators_single_time():
#     # Data
#     omega = 10
#     t = 0.1
#     H_0s = [qeye(2), sigmax(), sigmay(), sigmaz()]
#     Xs = [qeye(2), sigmax(), sigmay(), sigmaz()]

#     for i in range(len(H_0s)):
#         for j in range(len(Xs)):
#             H_0, X = H_0s[i], Xs[j]
#             # Dysolve
#             U = dysolve_propagator(H_0, X, omega, t, {'max_orer':0})

#             # Qutip.solver.propagator
#             def H1_coeff(t, omega):
#                 return np.cos(omega * t)

#             basis = H_0.eigenstates()[1]

#             H = [H_0, [X, H1_coeff]]
#             args = {'omega': omega}
#             prop = propagator(
#                 H, t, args=args, options={"atol": 1e-10, "rtol": 1e-8}
#             )

#             with CoreOptions(atol=1e-8, rtol=1e-8):
#                 assert U == prop, (i,j)
