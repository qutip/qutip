import pytest
import functools
from itertools import product
import numpy as np
from scipy.integrate import trapezoid
import qutip

pytestmark = [pytest.mark.usefixtures("in_temporary_directory")]

_equivalence_dimension = 15
_equivalence_fock = qutip.fock(_equivalence_dimension, 1)
_equivalence_coherent = qutip.coherent_dm(_equivalence_dimension, 2)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(["solver", "start"], [
    pytest.param("es", _equivalence_coherent, id="es"),
    pytest.param("es", None, id="es-steady state"),
])
def test_correlation_solver_equivalence(solver, start):
    """
    Test that all of the correlation solvers give the same results for a given
    system.
    """
    a = qutip.destroy(_equivalence_dimension)
    x = ( a + a.dag() ) / np.sqrt(2)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2
    c_ops = [np.sqrt(G1 * (n_th+1)) * a,
             np.sqrt(G1 * n_th) * a.dag()]
    times = np.linspace(0, 3, 61)
    # Massively relax the tolerance for the Monte-Carlo approach to avoid a
    # long simulation time.
    tol = 0.25 if solver == "mc" else 1e-4
    # We use the master equation version as a base, but it doesn't actually
    # matter - if all the tests fail, it implies that the "me" solver might be
    # broken, whereas if only one fails, then it implies that only that one is
    # broken.  We test that all solvers are equivalent by transitive equality
    # to the "me" solver.
    base = qutip.correlation_2op_2t(H, start, None, times, c_ops,
                                    a.dag(), a, solver="me")
    cmp = qutip.correlation_2op_2t(H, start, None, times, c_ops,
                                   a.dag(), a, solver=solver)
    np.testing.assert_allclose(base, cmp, atol=tol)


def _spectrum_wrapper(solver):
    frequencies = 2*np.pi * np.linspace(0.5, 1.5, 101)
    @functools.wraps(qutip.spectrum)
    def out(H, c_ops, a, b):
        return (qutip.spectrum(H, frequencies, c_ops, a, b, solver=solver),
                frequencies)
    return out


def _spectrum_fft(H, c_ops, a, b):
    times = np.linspace(0, 100, 2500)
    correlation = qutip.correlation_2op_1t(H, None, times, c_ops, a, b)
    frequencies, spectrum = qutip.spectrum_correlation_fft(times, correlation)
    return spectrum, frequencies


@pytest.mark.parametrize("spectrum", [
    pytest.param(_spectrum_fft, id="fft"),
    pytest.param(_spectrum_wrapper("es"), id="es"),
    pytest.param(_spectrum_wrapper("pi"), id="pi"),
    pytest.param(_spectrum_wrapper("solve"), id="solve"),
])
def test_spectrum_solver_equivalence_to_es(spectrum):
    """Test equivalence of the spectrum solvers to the base "es" method."""
    # Jaynes--Cummings model.
    dimension = 4
    wc = wa = 1.0 * 2*np.pi
    g = 0.1 * 2*np.pi
    kappa = 0.75
    gamma = 0.25
    n_th = 0.01

    a = qutip.tensor(qutip.destroy(dimension), qutip.qeye(2))
    sm = qutip.tensor(qutip.qeye(dimension), qutip.sigmam())
    H = wc*a.dag()*a + wa*sm*sm.dag() + g*(a.dag()*sm.dag() + a*sm)
    c_ops = [np.sqrt(kappa * (n_th+1)) * a,
             np.sqrt(kappa * n_th) * a.dag(),
             np.sqrt(gamma) * sm.dag()]

    test, frequencies = spectrum(H, c_ops, a.dag(), a)
    base = qutip.spectrum(H, frequencies, c_ops, a.dag(), a, solver="es")
    np.testing.assert_allclose(base, test, atol=1e-3)


def _trapz_2d(z, xy):
    """2D trapezium-method integration assuming a square grid."""
    dx = xy[1] - xy[0]
    return dx*dx * trapezoid(trapezoid(z, axis=0))


def _n_correlation(times, n):
    """
    Numerical integration of the correlation function given an array of
    expectation values.
    """
    return np.array([[n[t] * n[t+tau] for tau in range(times.shape[0])]
                     for t in range(times.shape[0])])


def _coefficient_function(t, args):
    t_off, tp = args['t_off'], args['tp']
    return np.exp(-(t-t_off)*(t-t_off) / (2*tp*tp))


_coefficient_string = "exp(-(t-t_off)**2 / (2 * tp*tp))"


def _h_qobj_function(t, args):
    return args['H0'] * _coefficient_function(t, args)


# 2LS and 3LS stand for two- and three-level system respectively.

_2ls_args = {'H0': 2*qutip.sigmax(), 't_off': 1, 'tp': 0.5}
_2ls_times = np.linspace(0, 5, 51)
_3ls_args = {'t_off': 2, 'tp': 1}
_3ls_times = np.linspace(0, 6, 20)


def _2ls_g2_0(H, c_ops):
    sp = qutip.sigmap()
    start = qutip.basis(2, 0)
    times = _2ls_times
    H = qutip.QobjEvo(H, args=_2ls_args, tlist=times)
    correlation = qutip.correlation_3op_2t(H, start, times, times, [sp],
                                           sp.dag(), sp.dag()*sp, sp,
                                           args=_2ls_args)
    n_expectation = qutip.mesolve(H, start, times, [sp] + c_ops,
                                  e_ops=[qutip.num(2)],
                                  args=_2ls_args).expect[0]
    integral_correlation = _trapz_2d(np.real(correlation), times)
    integral_n_expectation = trapezoid(n_expectation, times)
    # Factor of two from negative time correlations.
    return 2 * integral_correlation / integral_n_expectation**2


@pytest.fixture(params=[
    pytest.param(_coefficient_string, id="string"),
    pytest.param(_coefficient_function(_2ls_times, _2ls_args), id="numpy"),
    pytest.param(_coefficient_function, id="function"),
])
def dependence_2ls(request):
    return request.param


class TestTimeDependence:
    """
    Test correlations with time-dependent operators using a two-level system
    (2LS) or a three-level system (3LS).
    """
    def test_varying_coefficient_hamiltonian_2ls(self, dependence_2ls):
        H = [[_2ls_args['H0'], dependence_2ls]]
        assert abs(_2ls_g2_0(H, []) - 0.575) < 1e-2

    def test_hamiltonian_from_function_2ls(self):
        H = _h_qobj_function
        assert abs(_2ls_g2_0(H, []) - 0.575) < 1e-2

    @pytest.mark.slow
    def test_varying_coefficient_hamiltonian_c_ops_2ls(self, dependence_2ls):
        H = [[_2ls_args['H0'], dependence_2ls]]
        c_ops = [[2*qutip.sigmam()*qutip.sigmap(), dependence_2ls]]
        assert abs(_2ls_g2_0(H, c_ops) - 0.824) < 1e-2

    @pytest.mark.slow
    @pytest.mark.parametrize("dependence_3ls", [
        pytest.param(_coefficient_string, id="string"),
        pytest.param(_coefficient_function(_3ls_times, _3ls_args), id="numpy"),
        pytest.param(_coefficient_function, id="function"),
    ])
    def test_coefficient_c_ops_3ls(self, dependence_3ls):
        # Calculate zero-delay HOM cross-correlation for incoherently pumped
        # three-level system, g2_ab[0] with gamma = 1.
        dimension = 3
        H = qutip.qzero(dimension)
        start = qutip.basis(dimension, 2)
        times = _3ls_times
        project_0_1 = qutip.projection(dimension, 0, 1)
        project_1_2 = qutip.projection(dimension, 1, 2)
        population_1 = qutip.projection(dimension, 1, 1)
        # Define the pi pulse to be when 99% of the population is transferred.
        rabi = np.sqrt(-np.log(0.01) / (_3ls_args['tp']*np.sqrt(np.pi)))
        coeff_3ls = qutip.coefficient(dependence_3ls, tlist=times,
                                      args=_3ls_args)
        c_ops = [project_0_1, [rabi*project_1_2, coeff_3ls]]
        forwards = qutip.correlation_2op_2t(H, start, times, times, c_ops,
                                            project_0_1.dag(), project_0_1,
                                            args=_3ls_args)
        backwards = qutip.correlation_2op_2t(H, start, times, times, c_ops,
                                             project_0_1.dag(), project_0_1,
                                             args=_3ls_args, reverse=True)
        times2 = np.concatenate([times, times[1:] + times[-1]])
        n_expect = qutip.mesolve(H, start, times2, c_ops, args=_3ls_args,
                                 e_ops=[population_1]).expect[0]
        correlation_ab = -forwards*backwards + _n_correlation(times, n_expect)
        g2_ab_0 = _trapz_2d(np.real(correlation_ab), times)
        assert abs(g2_ab_0 - 0.185) < 1e-2


def _step(t):
    return np.arctan(t)/np.pi + 0.5


def test_hamiltonian_order_unimportant():
    # Testing for regression on issue 1048.
    sp = qutip.sigmap()
    H = [[qutip.sigmax(), lambda t: _step(t-2)],
         [qutip.qeye(2), lambda t: _step(-(t-2))]]
    start = qutip.basis(2, 0)
    times = np.linspace(0, 5, 6)
    forwards = qutip.correlation_2op_2t(H, start, times, times, [sp],
                                        sp.dag(), sp)
    backwards = qutip.correlation_2op_2t(H[::-1], start, times, times, [sp],
                                         sp.dag(), sp)
    np.testing.assert_allclose(forwards, backwards, atol=1e-6)


@pytest.mark.parametrize(['solver', 'state'], [
    pytest.param('me', _equivalence_fock, id="me-ket"),
    pytest.param('me', _equivalence_coherent, id="me-dm"),
    pytest.param('me', None, id="me-steady"),
    pytest.param('es', _equivalence_fock, id="es-ket"),
    pytest.param('es', _equivalence_coherent, id="es-dm"),
    pytest.param('es', None, id="es-steady"),
])
@pytest.mark.parametrize("is_e_op_hermitian", [True, False],
                         ids=["hermitian", "nonhermitian"])
@pytest.mark.parametrize("w", [1, 2])
@pytest.mark.parametrize("gamma", [1, 10])
def test_correlation_2op_1t_known_cases(solver,
                                        state,
                                        is_e_op_hermitian,
                                        w,
                                        gamma,
                                       ):
    """This test compares the output correlation_2op_1 solution to an analytical
    solution."""

    a = qutip.destroy(_equivalence_dimension)
    x = (a + a.dag())/np.sqrt(2)

    H = w * a.dag() * a

    a_op = x if is_e_op_hermitian else a
    b_op = x if is_e_op_hermitian else a.dag()
    c_ops = [np.sqrt(gamma) * a]

    times = np.linspace(0, 1, 30)

    # Handle the case state==None when computing expt values
    rho0 = state if state else qutip.steadystate(H, c_ops)
    if is_e_op_hermitian:
        # Analitycal solution for x,x as operators.
        base = 0
        base += qutip.expect(a*x, rho0)*np.exp(-1j*w*times - gamma*times/2)
        base += qutip.expect(a.dag()*x, rho0)*np.exp(1j*w*times - gamma*times/2)
        base /= np.sqrt(2)
    else:
        # Analitycal solution for a,adag as operators.
        base = qutip.expect(a*a.dag(), rho0)*np.exp(-1j*w*times - gamma*times/2)

    cmp = qutip.correlation_2op_1t(H, state, times, c_ops, a_op, b_op, solver=solver)

    np.testing.assert_allclose(base, cmp, atol=0.25 if solver == 'mc' else 2e-5)


def test_correlation_timedependant_op():
    num = qutip.num(2)
    a = qutip.destroy(2)
    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    times = np.arange(4)
    # switch between sx and sz at t=1.5
    A_op = qutip.QobjEvo([[sx, lambda t: t<=1.5], [sz, lambda t: t>1.5]])

    cmp_sx = qutip.correlation_2op_1t(num, None, times, [a], sx, sx)
    cmp_sz = qutip.correlation_2op_1t(num, None, times, [a], sz, sx)
    cmp_switch = qutip.correlation_2op_1t(num, None, times, [a], A_op, sx)
    np.testing.assert_allclose(cmp_sx[:2], cmp_switch[:2])
    np.testing.assert_allclose(cmp_sz[-2:], cmp_switch[-2:])


def test_alternative_solver():
    from qutip.solver.mesolve import MESolver
    from qutip.solver.brmesolve import BRSolver

    H = qutip.num(5)
    a = qutip.destroy(5)
    a_ops = [(a+a.dag(), qutip.coefficient(lambda _, w: w>0, args={"w":0}))]

    br = BRSolver(H, a_ops)
    me = MESolver(H, [a])
    times = np.arange(4)

    br_corr = qutip.correlation_3op(br, qutip.basis(5), [0], times, a, a.dag())
    me_corr = qutip.correlation_3op(me, qutip.basis(5), [0], times, a, a.dag())

    np.testing.assert_allclose(br_corr, me_corr)


def test_G1():
    H = qutip.Qobj([[0,1], [1,0]])
    psi0 = qutip.basis(2)
    taus = np.linspace(0, 1, 11)
    scale = 2
    a_op = qutip.sigmaz() * scale
    g1, G1 = qutip.coherence_function_g1(H, psi0, taus, [], a_op)
    expected = np.array([np.cos(t)**2 - np.sin(t)**2 for t in taus])
    np.testing.assert_allclose(g1, expected, rtol=2e-5)
    np.testing.assert_allclose(G1, expected * scale**2, rtol=2e-5)


def test_G2():
    N = 10
    H = qutip.rand_dm(N)
    psi0 = qutip.rand_ket(N)
    taus = np.linspace(0, 1, 11)
    scale = 2
    a_op = qutip.rand_unitary(N) * scale
    g1, G1 = qutip.coherence_function_g2(H, psi0, taus, [], a_op)
    expected = np.ones(11)
    np.testing.assert_allclose(g1, expected, rtol=2e-5)
    np.testing.assert_allclose(G1, expected * scale**4, rtol=2e-5)
