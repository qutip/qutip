from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip.solver import propagator
from qutip import Qobj, sigmax, sigmay, sigmaz, qeye, tensor, CoreOptions
from scipy.special import factorial
import numpy as np
import pytest


def test_integrals_1():
    # In order to have an instance to call the function
    # Data inside doesn't matter and is not used in the calculations.
    dysolve = DysolvePropagator(
        sigmaz(), sigmax(), 1
    )

    omegas = [-10, -1, -0.1, 0.1, 1, 10]
    dts = [-10, -1, -0.1, 0.1, 1, 10]

    for dt in dts:
        dysolve.dt = dt
        for omega in omegas:
            list_ws = [
                np.array([]),
                np.array([0]),
                np.array([omega]),
                np.array([0, 0, 0, 0, 0]),
                np.array([omega, 0]),
                np.array([0, omega])
            ]

            answers = [
                1,
                dt,
                (-1j / omega) * (np.exp(1j * omega * dt) - 1),
                (dt**5) / factorial(5),
                (-1j / omega) * ((-1j / omega)*(np.exp(1j*omega*dt)-1) - dt),
                (-1j*dt/omega) * np.exp(1j*omega*dt) -
                ((1j/omega)**2) * (np.exp(1j*omega*dt)-1)
            ]

            for ws, answer in zip(list_ws, answers):
                integrals = dysolve._compute_integrals(ws)
                assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)


def test_integrals_2():
    # In order to have an instance to call the function
    # Data inside doesn't matter and is not used in the calculations.
    dysolve = DysolvePropagator(
        sigmaz(), sigmax(), 1
    )

    omegas_1 = [-25, -5, -0.5, 0.5, 5, 25]
    omegas_2 = [-25, -5, -0.5, 0.5, 5, 25]
    dts = [-10, -1, -0.1, 0.1, 1, 10]

    for dt in dts:
        dysolve.dt = dt
        for omega_1 in omegas_1:
            for omega_2 in omegas_2:
                ws = [omega_1, omega_2]
                integrals = dysolve._compute_integrals(ws)

                if omega_1 + omega_2 == 0:
                    answer = (-1j*dt/omega_1) + \
                        (np.exp(1j*omega_2*dt)-1)/(omega_1*omega_2)
                else:
                    exp_1 = np.exp(1j*(omega_1+omega_2)*dt)
                    exp_2 = np.exp(1j*omega_2*dt)
                    answer = -(exp_1-1)/(omega_1*(omega_1+omega_2)) + \
                        (exp_2-1)/(omega_1*omega_2)

                assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)

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
#             U = dysolve_propagator(H_0, X, omega, t, {'max_orer':5})

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
