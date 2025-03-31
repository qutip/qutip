from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip.solver import propagator
from qutip import Qobj, sigmax, sigmay, sigmaz, qeye, tensor, CoreOptions
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


# @pytest.mark.parametrize("X", [
#     sigmax(), sigmay(), sigmaz(), qeye(2),
#     sigmax() + sigmay(), sigmaz() + qeye(2),
#     tensor(sigmax(), sigmaz()), tensor(sigmay(), sigmaz()),
#     tensor(sigmax(), sigmaz()) + tensor(sigmay(), sigmaz()),
# ])
# @pytest.mark.parametrize("max_order, answer", [
#     (
#         1,
#         lambda indices, X: np.tile(X.full(), 1).reshape((indices.shape[0], 1))
#     ),
#     (
#         2,
#         lambda indices, X: np.tile(
#             np.tile(X.full(), 1).reshape((X.shape[0]**2, 1)), X.shape[0]
#         ).reshape((indices.shape[0], 1)) *
#         np.tile(X.full(), 1).reshape(
#             (X.shape[0]**2, 1)
#         ).repeat(X.shape[0]).reshape((indices.shape[0], 1))
#     )
# ])
# def test_matrix_elements_1(empty_instance, X, max_order, answer):
#     dysolve = empty_instance
#     dysolve.X = X
#     current_matrix_elements = None

#     for n in range(1, max_order + 1):
#         indices = np.array(
#             list(
#                 itertools.product(
#                     range(X.shape[0]), repeat=n + 1
#                 )
#             )
#         )
#         current_matrix_elements = dysolve._update_matrix_elements(
#             current_matrix_elements
#         )

#     answer = answer(indices, X)
#     assert np.array_equal(current_matrix_elements, answer)


@pytest.mark.parametrize("max_order, answer", [
    (
        1,
        np.array([[0], [-1j], [1j], [0]])
    ),
    (
        2,
        np.array([[0], [0], [1], [0], [0], [1], [0], [0]])
    ),
    (
        3,
        np.array([[0], [0], [0], [0], [0], [-1j], [0], [0],
                  [0], [0], [1j], [0], [0], [0], [0], [0],])
    )
])
def test_matrix_elements_2(empty_instance, max_order, answer):
    dysolve = empty_instance
    dysolve.X = sigmay()
    current_matrix_elements = None

    for n in range(1, max_order + 1):
        current_matrix_elements = dysolve._update_matrix_elements(
            current_matrix_elements
        )
    assert np.array_equal(current_matrix_elements, answer)


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


# def test_prop():
#     #Data
#     omega = 1
#     t = 0.1
#     H_0s = [qeye(2)]
#     Xs = [qeye(2)]

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
