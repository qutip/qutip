from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip.solver import propagator
from qutip import sigmax, sigmaz, qeye, tensor, CoreOptions
import numpy as np


# def test_number_of_propagators():
#     # Should have the same number of propagators as len(dysolve.times)
#     t_i = 0
#     t_f = 1
#     dts = [0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1]

#     for dt in dts:
#         dysolve = DysolvePropagator(
#             sigmaz(), sigmax(), 1, {'max_order': 1, 'a_tol': 1e-8}
#         )
#         Us = dysolve(t_i, t_f, dt)
#         assert (len(dysolve.times) == len(Us))


# def test_dims():
#     # Dimensions of H_0, X, and propagators shoud be the same
#     H_0s = [sigmaz(), tensor(sigmaz(), sigmaz()),
#             tensor(sigmaz(), sigmaz(), sigmaz()),
#             tensor(sigmaz(), sigmaz(), sigmaz(), sigmaz())]
#     Xs = [sigmax(), tensor(sigmax(), sigmax()),
#           tensor(sigmax(), sigmax(), sigmax()),
#           tensor(sigmax(), sigmax(), sigmax(), sigmax())]

#     for H_0, X in zip(H_0s, Xs):
#         dysolve = DysolvePropagator(
#             H_0, X, 1, {'max_order': 1, 'a_tol': 1e-8}
#         )
#         Us = dysolve(0, 1, 0.1)
#         for U in Us:
#             assert (dysolve.H_0.dims == dysolve.X.dims == U.dims)

# def test_2x2_propagators_single_time():
#     # Data
#     omega = 1
#     t = 0.1
#     H_0 = sigmaz()
#     X = sigmax()

#     # Dysolve
#     U = dysolve_propagator(H_0, X, omega, t)

#     # Qutip.solver.propagator
#     def H1_coeff(t, omega):
#         return np.cos(omega * t)

#     H = [H_0, [X, H1_coeff]]
#     args = {'omega': omega}
#     prop = propagator(
#         H, t, args=args, options={"atol": 1e-10, "rtol": 1e-8}
#     )

#     with CoreOptions(atol=1e-8, rtol=1e-8):
#         assert U == prop

# def test_4x4_propagators():
#     max_order = 5
#     omega = 10
#     t_i = 0
#     t_f = 1
#     dt = 0.01
#     H_0 = tensor(sigmaz(), qeye(2))
#     X = tensor(sigmax(), qeye(2))

#     # Dysolve
#     dysolve, propagators = dysolve_propagator(
#         max_order, H_0, X, omega, t_i, t_f, dt
#     )

#     # Qutip.solver.propagator
#     def H1_coeff(t, omega):
#         return np.cos(omega * t)

#     H = [H_0, [X, H1_coeff]]
#     args = {'omega': omega}
#     prop = propagator(
#         H, dysolve.times, args=args, options={"atol": 1e-10, "rtol": 1e-8}
#     )[1:]

#     with CoreOptions(atol=1e-8, rtol=1e-8):
#         for i in range(len(propagators)-1):
#             # .conj() for them to be in the same basis
#             assert (propagators[i] == prop[i].conj())


# def test_8x8_propagators():
#     max_order = 4
#     omega = 100
#     t_i = 0
#     t_f = 1
#     dt = 0.01
#     H_0 = tensor(sigmaz(), qeye(2), qeye(2))
#     X = tensor(sigmax(), qeye(2), qeye(2))

#     # Dysolve
#     dysolve, propagators = dysolve_propagator(
#         max_order, H_0, X, omega, t_i, t_f, dt
#     )

#     # Qutip.solver.propagator
#     def H1_coeff(t, omega):
#         return np.cos(omega * t)

#     H = [H_0, [X, H1_coeff]]
#     args = {'omega': omega}
#     prop = propagator(
#         H, dysolve.times, args=args, options={"atol": 1e-10, "rtol": 1e-8}
#     )[1:]

#     with CoreOptions(atol=1e-8, rtol=1e-8):
#         for i in range(len(propagators)-1):
#             # .conj() for them to be in the same basis
#             assert (propagators[i] == prop[i].conj())
