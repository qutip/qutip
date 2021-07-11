import numpy as np
import qutip


def test_unitary_evolution_two_level_system():
    delta = 1.0 * 2 * np.pi
    eps0 = 1.0 * 2 * np.pi
    A = 0.5 * 2 * np.pi
    omega = np.sqrt(delta**2 + eps0**2)
    T = 2*np.pi / omega
    tlist = np.linspace(0, 2*T, 101)
    psi0 = qutip.rand_ket(2)
    H0 = -0.5*eps0*qutip.sigmaz() - 0.5*delta*qutip.sigmax()
    H1 = 0.5 * A * qutip.sigmax()
    args = {'w': omega}
    H = [H0, [H1, lambda t, args: np.sin(args['w'] * t)]]
    e_ops = [qutip.num(2)]

    trial = qutip.fsesolve(H, psi0, tlist, e_ops, T, args).expect[0]
    expected = qutip.mesolve(H, psi0, tlist, [], e_ops, args).expect[0]

    np.testing.assert_allclose(trial, expected, atol=1e-4)
