from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip.solver import propagator
from qutip import sigmax, sigmaz, tensor, CoreOptions
import numpy as np


def test_number_of_propagators():
    # Should have the same number of propagators as len(dysolve.times)
    t_i = 0
    t_f = 1
    dts = [0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1]
    with CoreOptions():
        for dt in dts:
            dysolve = DysolvePropagator(
                1, sigmaz(), sigmax(), 1
            )
            dysolve(t_i, t_f, dt)
            assert (len(dysolve.times) == len(dysolve.Us))


def test_dims():
    # Dimensions of H_0, X, and propagators shoud be the same
    H_0s = [sigmaz(), tensor(sigmaz(), sigmaz()),
            tensor(sigmaz(), sigmaz(), sigmaz()),
            tensor(sigmaz(), sigmaz(), sigmaz(), sigmaz())]
    Xs = [sigmax(), tensor(sigmax(), sigmax()),
          tensor(sigmax(), sigmax(), sigmax()),
          tensor(sigmax(), sigmax(), sigmax(), sigmax())]

    with CoreOptions():
        for H_0, X in zip(H_0s, Xs):
            dysolve = DysolvePropagator(
                1, H_0, X, 1
            )
            dysolve(0, 1, 0.1)
            for U in dysolve.Us:
                assert (dysolve.H_0.dims == dysolve.X.dims == U.dims)


def test_2x2_propagators():
    # Data
    max_order = 5
    omega = 1
    t_i = 0
    t_f = 1
    dt = 0.1
    H_0 = sigmaz()
    X = sigmax()

    # Dysolve
    dysolve, propagators = dysolve_propagator(
        max_order, H_0, X, omega, t_i, t_f, dt
    )

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    prop = propagator(
        H, dysolve.times, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )[1:]

    with CoreOptions(atol=1e-8, rtol=1e-8):
        for i in range(len(propagators)-1):
            # .conj() for them to be in the same basis
            assert (propagators[i] == prop[i].conj())
