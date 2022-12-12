import pytest
import functools
from itertools import product
import numpy as np
import qutip

pytestmark = [pytest.mark.usefixtures("in_temporary_directory")]

_equivalence_dimension = 20
_equivalence_fock = qutip.fock(_equivalence_dimension, 1)
_equivalence_coherent = qutip.coherent_dm(_equivalence_dimension, 2)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(["solver", "start", "legacy"], [
    pytest.param("es", _equivalence_coherent, False, id="es"),
    pytest.param("es", _equivalence_coherent, True, id="es-legacy"),
    pytest.param("es", None, False, id="es-steady state"),
    pytest.param("es", None, True, id="es-steady state-legacy"),
    pytest.param("mc", _equivalence_fock, False, id="mc",
                 marks=pytest.mark.slow),
])
def test_correlation_solver_equivalence(solver, start, legacy):
    """
    Test that all of the correlation solvers give the same results for a given
    system.
    """
    a = qutip.destroy(_equivalence_dimension)
    x = ( a +a.dag() )/np.sqrt(2)
    H = a.dag() * a
    G1 = 0.75
    n_th = 2
    c_ops = [np.sqrt(G1 * (n_th+1)) * a,
             np.sqrt(G1 * n_th) * a.dag()]
    times = np.linspace(0, 5, 101)
    # Massively relax the tolerance for the Monte-Carlo approach to avoid a
    # long simulation time.
    tol = 0.25 if solver == "mc" else 1e-4
    correlation = (qutip.correlation if legacy
                   else qutip.correlation_2op_2t)
    # We use the master equation version as a base, but it doesn't actually
    # matter - if all the tests fail, it implies that the "me" solver might be
    # broken, whereas if only one fails, then it implies that only that one is
    # broken.  We test that all solvers are equivalent by transitive equality
    # to the "me" solver.
    base = correlation(H, start, None, times, c_ops, a.dag(), a, solver="me")
    cmp = correlation(H, start, None, times, c_ops, a.dag(), a, solver=solver)
    np.testing.assert_allclose(base, cmp, atol=tol)


def _spectrum_wrapper(spectrum):
    frequencies = 2*np.pi * np.linspace(0.5, 1.5, 101)
    @functools.wraps(spectrum)
    def out(H, c_ops, a, b):
        return spectrum(H, frequencies, c_ops, a, b), frequencies
    return out


def _spectrum_fft(H, c_ops, a, b):
    times = np.linspace(0, 100, 2500)
    correlation = qutip.correlation_ss(H, times, c_ops, a, b)
    frequencies, spectrum = qutip.spectrum_correlation_fft(times, correlation)
    return spectrum, frequencies


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("spectrum", [
    pytest.param(_spectrum_fft, id="fft"),
    pytest.param(_spectrum_wrapper(qutip.spectrum_ss), id="es-legacy"),
    pytest.param(_spectrum_wrapper(qutip.spectrum_pi), id="pi-legacy"),
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
    return dx*dx * np.trapz(np.trapz(z, axis=0))


def _n_correlation(times, n_expectation):
    """
    Numerical integration of the correlation function given an array of
    expectation values.
    """
    interp = qutip.Cubic_Spline(times[0], times[-1], n_expectation)
    n = interp(np.concatenate([times, times[1:] + times[-1]]))
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
    correlation = qutip.correlation_3op_2t(H, start, times, times, [sp],
                                           sp.dag(), sp.dag()*sp, sp,
                                           args=_2ls_args)
    n_expectation = qutip.mesolve(H, start, times, [sp] + c_ops,
                                  e_ops=[qutip.num(2)],
                                  args=_2ls_args).expect[0]
    integral_correlation = _trapz_2d(np.real(correlation), times)
    integral_n_expectation = np.trapz(n_expectation, times)
    # Factor of two from negative time correlations.
    return 2 * integral_correlation / integral_n_expectation**2


@pytest.fixture(params=[
    pytest.param(_coefficient_string, id="string"),
    pytest.param(_coefficient_function(_2ls_times, _2ls_args), id="numpy"),
    pytest.param(_coefficient_function, id="function"),
])
def dependence_2ls(request):
    return request.param


def coeff(t, _):
    return np.sqrt(8*t)


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
        c_ops = [project_0_1, [rabi*project_1_2, dependence_3ls]]
        forwards = qutip.correlation_2op_2t(H, start, times, times, c_ops,
                                            project_0_1.dag(), project_0_1,
                                            args=_3ls_args)
        backwards = qutip.correlation_2op_2t(H, start, times, times, c_ops,
                                             project_0_1.dag(), project_0_1,
                                             args=_3ls_args, reverse=True)
        n_expect = qutip.mesolve(H, start, times, c_ops, args=_3ls_args,
                                 e_ops=[population_1]).expect[0]
        correlation_ab = -forwards*backwards + _n_correlation(times, n_expect)
        g2_ab_0 = _trapz_2d(np.real(correlation_ab), times)
        assert abs(g2_ab_0 - 0.185) < 1e-2

    @pytest.mark.parametrize("solver", ["me", "mc"])
    def test_correlation_c_ops_td(self, solver):
        t1 = 0.2
        t2 = 0.5
        tlist = [0, 0.2, 0.5]
        H = qutip.qzero(2)
        psi0 = qutip.basis(2, 1)
        cops = [[qutip.destroy(2), coeff]]
        a = qutip.destroy(2)
        ad = a.dag()
        options = qutip.Options(nsteps=1e5)
        result = qutip.correlation_2op_2t(
            H, psi0, [0, t1], [0, t2-t1], cops, ad, a, solver=solver
        )[1, 1].real
        expected = np.exp(-(4.0 * t1 ** 2 + 4.0 * t2 ** 2) / 2)
        atol = 2e-6 if solver == "me" else 0.2 # 4 sigma
        np.testing.assert_allclose(result, expected, atol=atol)


def _step(t):
    return np.arctan(t)/np.pi + 0.5


def test_hamiltonian_order_unimportant():
    # Testing for regression on issue 1048.
    sp = qutip.sigmap()
    H = [[qutip.sigmax(), lambda t, _: _step(t-2)],
         [qutip.qeye(2), lambda t, _: _step(-(t-2))]]
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
    pytest.param('mc', _equivalence_fock, id="mc-ket",
                 marks=[pytest.mark.slow]),
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
