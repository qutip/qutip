import pytest
import numpy as np
import qutip
from qutip.solver.brmesolve import brmesolve
from qutip.core.environment import DrudeLorentzEnvironment


def pauli_spin_operators():
    return [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]


_simple_qubit_gamma = 0.25
coeff = qutip.coefficient(lambda t, w: _simple_qubit_gamma * (w >= 0),
                          args={'w': 0})
_m_c_op = np.sqrt(_simple_qubit_gamma) * qutip.sigmam()
_z_c_op = np.sqrt(_simple_qubit_gamma) * qutip.sigmaz()
_x_a_op = [qutip.sigmax(), coeff]


@pytest.mark.parametrize("me_c_ops, brme_c_ops, brme_a_ops", [
    pytest.param([_m_c_op], [], [_x_a_op], id="me collapse-br coupling"),
    pytest.param([_m_c_op], [_m_c_op], [], id="me collapse-br collapse"),
    pytest.param([_m_c_op, _z_c_op], [_z_c_op], [_x_a_op],
                 id="me collapse-br collapse-br coupling"),
])
def test_simple_qubit_system(me_c_ops, brme_c_ops, brme_a_ops):
    """
    Test that the BR solver handles collapse and coupling operators correctly
    relative to the standard ME solver.
    """
    delta = 0.0
    epsilon = 0.5 * 2 * np.pi
    e_ops = pauli_spin_operators()
    H = delta * 0.5 * qutip.sigmax() + epsilon * 0.5 * qutip.sigmaz()
    psi0 = (2 * qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    times = np.linspace(0, 10, 100)
    me = qutip.mesolve(H, psi0, times, c_ops=me_c_ops, e_ops=e_ops)
    opt = {"tensor_type": "dense"}
    brme = brmesolve(
        H, psi0, times,
        a_ops=brme_a_ops, c_ops=brme_c_ops,
        e_ops=e_ops, options=opt
    )
    for me_expectation, brme_expectation in zip(me.expect, brme.expect):
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=1e-2)


def _harmonic_oscillator_spectrum_frequency(n_th, w0, kappa):
    if n_th == 0:
        return lambda w: kappa * (w >= 0)

    w_th = w0 / np.log(1 + 1/n_th)

    def f(t, w):
        scale = np.exp(w / w_th) if w < 0 else 1
        return (n_th + 1) * kappa * scale
    return f


def _harmonic_oscillator_c_ops(n_th, kappa, dimension):
    a = qutip.destroy(dimension)
    if n_th == 0:
        return [np.sqrt(kappa) * a]
    return [np.sqrt(kappa * (n_th+1)) * a, np.sqrt(kappa * n_th) * a.dag()]


@pytest.mark.parametrize("n_th", [0, 0.15])
def test_harmonic_oscillator(n_th):
    N = 10
    w0 = 1.0 * 2*np.pi
    g = 0.05 * w0
    kappa = 0.15
    S_w = _harmonic_oscillator_spectrum_frequency(n_th, w0, kappa)

    a = qutip.destroy(N)
    H = w0*a.dag()*a + g*(a+a.dag())
    psi0 = (qutip.basis(N, 4) + qutip.basis(N, 2) + qutip.basis(N, 0)).unit()
    psi0 = qutip.ket2dm(psi0)
    times = np.linspace(0, 20, 1000)

    c_ops = _harmonic_oscillator_c_ops(n_th, kappa, N)
    a_ops = [[a + a.dag(), S_w]]
    e_ops = [a.dag()*a, a+a.dag()]

    me = qutip.mesolve(H, psi0, times, c_ops, e_ops=e_ops)
    brme = brmesolve(H, psi0, times, a_ops, e_ops=e_ops)
    for me_expectation, brme_expectation in zip(me.expect, brme.expect):
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=1e-2)

    num = qutip.num(N)
    me_num = qutip.expect(num, me.states)
    brme_num = qutip.expect(num, brme.states)
    np.testing.assert_allclose(me_num, brme_num, atol=1e-2)


def test_jaynes_cummings_zero_temperature_spectral_callable():
    """
    brmesolve: Jaynes-Cummings model, zero temperature
    """
    N = 10
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
    sp = qutip.tensor(qutip.qeye(N), qutip.sigmap())
    psi0 = qutip.ket2dm(qutip.tensor(qutip.basis(N, 1), qutip.basis(2, 0)))
    kappa = 0.05
    a_ops = [(a + a.dag(), lambda w: kappa * (w >= 0))]
    e_ops = [a.dag()*a, sp.dag()*sp]

    w0 = 1.0 * 2*np.pi
    g = 0.05 * 2*np.pi
    times = np.linspace(0, 2 * 2*np.pi / g, 1000)

    c_ops = [np.sqrt(kappa) * a]
    H = w0*a.dag()*a + w0*sp.dag()*sp + g*(a+a.dag())*(sp+sp.dag())

    me = qutip.mesolve(H, psi0, times, c_ops, e_ops=e_ops)
    brme = brmesolve(H, psi0, times, a_ops, e_ops=e_ops)
    for me_expectation, brme_expectation in zip(me.expect, brme.expect):
        # Accept 5% error.
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=5e-2)


def test_tensor_system():
    """
    brmesolve: Check for #572 bug.
    """
    w1, w2, w3 = 1, 2, 3
    gamma2, gamma3 = 0.1, 0.1
    id2 = qutip.qeye(2)

    # Hamiltonian for three uncoupled qubits
    H = (w1/2. * qutip.tensor(qutip.sigmaz(), id2, id2)
         + w2/2. * qutip.tensor(id2, qutip.sigmaz(), id2)
         + w3/2. * qutip.tensor(id2, id2, qutip.sigmaz()))

    # White noise
    S2 = qutip.coefficient(lambda t, w: gamma2, args={'w': 0})
    S3 = qutip.coefficient(lambda t, w: gamma3, args={'w': 0})

    qubit_2_x = qutip.tensor(id2, qutip.sigmax(), id2)
    qubit_3_x = qutip.tensor(id2, id2, qutip.sigmax())

    # Initial state : first qubit is excited
    grnd2 = qutip.sigmam() * qutip.sigmap()  # 2x2 ground
    exc2 = qutip.sigmap() * qutip.sigmam()   # 2x2 excited state
    ini = qutip.tensor(exc2, grnd2, grnd2)   # Full system

    # Projector on the excited state of qubit 1
    proj_up1 = qutip.tensor(exc2, id2, id2)
    times = np.linspace(0, 10./gamma3, 1000)

    sol = brmesolve(H, ini, times, [[qubit_2_x, S2], [qubit_3_x, S3]],
                    e_ops=[proj_up1]).expect[0]

    np.testing.assert_allclose(sol, np.ones_like(times))


def test_solver_accepts_list_hamiltonian():
    """
    brmesolve: input list of Qobj
    """
    delta = 0.0 * 2 * np.pi
    epsilon = 0.5 * 2 * np.pi
    gamma = 0.25
    c_ops = [np.sqrt(gamma) * qutip.sigmam()]
    e_ops = pauli_spin_operators()
    H = [delta * 0.5 * qutip.sigmax(), epsilon * 0.5 * qutip.sigmaz()]
    psi0 = (2 * qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    times = np.linspace(0, 10, 100)
    me = qutip.mesolve(H, psi0, times, c_ops=c_ops, e_ops=e_ops).expect
    brme = brmesolve(H, psi0, times, [], e_ops=e_ops, c_ops=c_ops).expect
    for me_expectation, brme_expectation in zip(me, brme):
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=1e-8)


def test_jaynes_cummings_zero_temperature_spectral_str():
    N = 10
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
    sp = qutip.tensor(qutip.qeye(N), qutip.sigmap())
    psi0 = qutip.ket2dm(qutip.tensor(qutip.basis(N, 1), qutip.basis(2, 0)))
    kappa = 0.05
    a_ops = [[(a + a.dag()), "{kappa} * (w >= 0)".format(kappa=kappa)]]
    e_ops = [a.dag()*a, sp.dag()*sp]

    w0 = 1.0 * 2*np.pi
    g = 0.05 * 2*np.pi
    times = np.linspace(0, 2 * 2*np.pi / g, 1000)

    c_ops = [np.sqrt(kappa) * a]
    H = w0*a.dag()*a + w0*sp.dag()*sp + g*(a+a.dag())*(sp+sp.dag())

    me = qutip.mesolve(H, psi0, times, c_ops, e_ops=e_ops)
    brme = brmesolve(H, psi0, times, a_ops, e_ops=e_ops)
    for me_expectation, brme_expectation in zip(me.expect, brme.expect):
        # Accept 5% error.
        np.testing.assert_allclose(me_expectation, brme_expectation, atol=5e-2)


def _mixed_string(kappa, _):
    return "{kappa} * exp(-t) * (w >= 0)".format(kappa=kappa), "1"


def _separate_strings(kappa, _):
    return ("{kappa} * (w >= 0)".format(kappa=kappa), "exp(-t/2)")


def _string_w_interpolating_t(kappa, times):
    spline = qutip.coefficient(np.exp(-times/2), tlist=times)
    return ("{kappa} * (w >= 0)".format(kappa=kappa), spline)


@pytest.mark.slow
@pytest.mark.parametrize("time_dependence_tuple", [
    _mixed_string,
    _separate_strings,
    _string_w_interpolating_t,
])
def test_time_dependence_tuples(time_dependence_tuple):
    N = 10
    a = qutip.destroy(N)
    H = a.dag()*a
    psi0 = qutip.basis(N, 9)
    times = np.linspace(0, 10, 100)
    kappa = 0.2
    spectra, coeff = time_dependence_tuple(kappa, times)
    a_ops = [[qutip.QobjEvo([a + a.dag(), coeff]), spectra]]
    exact = 9 * np.exp(-kappa * (1 - np.exp(-times)))
    brme = brmesolve(H, psi0, times, a_ops, e_ops=[a.dag()*a])
    assert np.mean(np.abs(brme.expect[0] - exact) / exact) < 1e-5


def test_time_dependent_spline_in_c_ops():
    N = 10
    a = qutip.destroy(N)
    H = a.dag()*a
    psi0 = qutip.basis(N, 9)
    times = np.linspace(0, 10, 100)
    kappa = 0.2
    exact = 9 * np.exp(-2 * kappa * (1 - np.exp(-times)))
    spectra, coeff = _string_w_interpolating_t(kappa, times)
    a_ops = [[qutip.QobjEvo([a + a.dag(), coeff]), spectra]]
    collapse_points = np.sqrt(kappa) * np.exp(-0.5*times)
    c_ops = [[a, qutip.coefficient(collapse_points, tlist=times)]]
    brme = brmesolve(H, psi0, times, a_ops, e_ops=[a.dag()*a], c_ops=c_ops)
    assert np.mean(np.abs(brme.expect[0] - exact) / exact) < 1e-5


@pytest.mark.slow
def test_nonhermitian_e_ops():
    N = 5
    a = qutip.destroy(N)
    coefficient = np.random.random() + 1j*np.random.random()
    H = a.dag()*a + coefficient*a + np.conj(coefficient)*a.dag()
    H_brme = [[H, '1']]
    psi0 = qutip.basis(N, 2)
    times = np.linspace(0, 10, 10)
    me = qutip.mesolve(H, psi0, times, c_ops=[], e_ops=[a]).expect[0]
    brme = brmesolve(H_brme, psi0, times, a_ops=[], e_ops=[a]).expect[0]
    np.testing.assert_allclose(me, brme, atol=1e-4)


@pytest.mark.slow
def test_result_states():
    N = 5
    a = qutip.destroy(N)
    coefficient = np.random.random() + 1j*np.random.random()
    H = a.dag()*a + coefficient*a + np.conj(coefficient)*a.dag()
    H_brme = [[H, '1']]
    psi0 = qutip.fock_dm(N, 2)
    times = np.linspace(0, 10, 10)
    me = qutip.mesolve(H, psi0, times).states
    brme = brmesolve(H_brme, psi0, times).states
    assert max(np.abs((me_state - brme_state).full()).max()
               for me_state, brme_state in zip(me, brme)) < 1e-5


@pytest.mark.slow
def test_hamiltonian_taking_arguments():
    N = 10
    w0 = 1.0 * 2*np.pi
    g = 0.75 * 2*np.pi
    kappa = 0.05
    a = qutip.tensor(qutip.destroy(N), qutip.qeye(2))
    sp = qutip.tensor(qutip.qeye(N), qutip.sigmap())
    psi0 = qutip.tensor(qutip.basis(N, 1), qutip.basis(2, 0))
    psi0 = qutip.ket2dm(psi0)
    times = np.linspace(0, 5 * 2*np.pi / g, 1000)

    a_ops = [[(a + a.dag()), "{kappa}*(w > 0)".format(kappa=kappa)]]
    e_ops = [a.dag()*a, sp.dag()*sp]

    H = w0*a.dag()*a + w0*sp.dag()*sp + g*(a+a.dag())*(sp+sp.dag())
    args = {'ii': 1}

    no_args = brmesolve(H, psi0, times, a_ops, e_ops=e_ops)
    args = brmesolve([[H, 'ii']], psi0, times, a_ops, e_ops=e_ops, args=args)
    for arg, no_arg in zip(args.expect, no_args.expect):
        np.testing.assert_allclose(arg, no_arg, atol=1e-10)


def test_feedback():
    N = 10
    tol = 1e-4
    psi0 = qutip.basis(N, 7)
    a = qutip.destroy(N)
    H = qutip.QobjEvo(qutip.num(N))
    a_op = (
        qutip.QobjEvo(a + a.dag()),
        qutip.coefficient("(A.real - 4)*(w > 0)", args={"A": 7.+0j, "w": 0.})
    )
    solver = qutip.BRSolver(H, [a_op])
    result = solver.run(
        psi0, np.linspace(0, 3, 31), e_ops=[qutip.num(N)],
        args={"A": qutip.BRSolver.ExpectFeedback(qutip.num(N))}
    )
    assert np.all(result.expect[0] > 4. - tol)


@pytest.mark.parametrize("lam,gamma,beta", [(0.05, 1, 1), (0.1, 5, 2)])
def test_accept_environment(lam, gamma, beta):
    DL = (
        "2 * pi * 2.0 * {lam} / (pi * {gamma} * {beta}) if (w==0) "
        "else 2 * pi * (2.0 * {lam} * {gamma} * w / (pi * (w**2 + {gamma}**2))) "
        "* ((1 / (exp(w * {beta}) - 1)) + 1)"
    ).format(gamma=gamma, beta=beta, lam=lam)
    H = 0.5 * qutip.sigmax()
    psi0 = (2 * qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    times = np.linspace(0, 10, 100)
    resultBR_str = brmesolve(
        H, psi0, times,
        a_ops=[[qutip.sigmaz(), DL]],
        e_ops=[qutip.sigmaz()]
    )
    env = DrudeLorentzEnvironment(T=1/beta, lam=lam, gamma=gamma)
    resultBR_env = brmesolve(
        H, psi0, times,
        a_ops=[[qutip.sigmaz(), env]],
        e_ops=[qutip.sigmaz()]
    )
    assert np.allclose(resultBR_env.expect[0], resultBR_str.expect[0])
