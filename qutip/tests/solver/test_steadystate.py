import numpy as np
import scipy
import pytest
import qutip
import warnings
from packaging import version as pac_version


@pytest.mark.parametrize(['method', 'kwargs'], [
    pytest.param('direct', {}, id="direct"),
    pytest.param('direct', {'solver':'mkl'}, id="direct_mkl",
                 marks=pytest.mark.skipif(not qutip.settings.has_mkl,
                                          reason='MKL extensions not found.')),
    pytest.param('direct', {'sparse':False}, id="direct_dense"),
    pytest.param('direct', {'use_rcm':True}, id="direct_rcm"),
    pytest.param('direct', {'use_wbm':True}, id="direct_wbm"),
    pytest.param('eigen', {}, id="eigen"),
    pytest.param('eigen', {'use_rcm':True},  id="eigen_rcm"),
    pytest.param('svd', {}, id="svd"),
    pytest.param('power', {'power_tol':1e-5}, id="power"),
    pytest.param('power', {'power_tol':1e-5, 'solver':'mkl'}, id="power_mkl",
                 marks=pytest.mark.skipif(not qutip.settings.has_mkl,
                                          reason='MKL extensions not found.')),
    pytest.param('power-gmres', {'tol':1e-1, 'atol': 1e-12}, id="power-gmres"),
    pytest.param('power-gmres', {'tol':1e-1, 'use_rcm':True, 'use_wbm':True, 'atol': 1e-12},
                 id="power-gmres_perm"),
    pytest.param('power-bicgstab', {"atol": 1e-10, "tol": 1e-10, 'use_precond':1}, id="power-bicgstab"),
    pytest.param('iterative-gmres', {'tol': 1e-12, 'atol': 1e-12}, id="iterative-gmres"),
    pytest.param('iterative-gmres', {'use_rcm':True, 'use_wbm':True, 'tol': 1e-12, 'atol': 1e-12},
                 id="iterative-gmres_perm"),
    pytest.param('iterative-bicgstab', {'atol': 1e-12, "tol": 1e-10},
                 id="iterative-bicgstab"),
])
@pytest.mark.parametrize("dtype", ["dense", "dia", "csr"])
@pytest.mark.filterwarnings("ignore:Only CSR matrices:RuntimeWarning")
def test_qubit(method, kwargs, dtype):
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sz = qutip.sigmaz().to(dtype)
    sm = qutip.destroy(2, dtype=dtype)

    if (
        pac_version.parse(scipy.__version__) >= pac_version.parse("1.12")
        and "tol" in kwargs
    ):
        # From scipy 1.12, the tol keyword is renamed to rtol
        kwargs["rtol"] = kwargs.pop("tol")

    H = 0.5 * 2 * np.pi * sz
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 20)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):
        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * sm.dag())
        rho_ss = qutip.steadystate(H, c_op_list, method=method, **kwargs)
        p_ss[idx] = qutip.expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = np.exp(-1.0 / wth_vec) / (1 + np.exp(-1.0 / wth_vec))
    np.testing.assert_allclose(p_ss_analytic, p_ss, atol=1e-5)


@pytest.mark.parametrize(['method', 'kwargs'], [
    pytest.param('direct', {}, id="direct"),
    pytest.param('direct', {'solver': 'mkl'}, id="direct_mkl",
                 marks=pytest.mark.skipif(not qutip.settings.has_mkl,
                                          reason='MKL extensions not found.')),
    pytest.param('direct', {'sparse': False}, id="direct_dense"),
    pytest.param('direct', {'use_rcm': True}, id="direct_rcm"),
    pytest.param('direct', {'use_wbm': True}, id="direct_wbm"),
    pytest.param('eigen', {}, id="eigen"),
    pytest.param('eigen', {'use_rcm': True},  id="eigen_rcm"),
    pytest.param('svd', {}, id="svd"),
])
def test_exact_solution_for_simple_methods(method, kwargs):
    # this tests that simple methods correctly determine the steadystate
    # with high accuracy for a small Liouvillian requiring correct weighting.
    H = qutip.identity(2)
    c_ops = [qutip.sigmam(), 1e-5 * qutip.sigmap()]
    rho_ss = qutip.steadystate(H, c_ops, method=method, **kwargs)
    expected_rho_ss = np.array([
        [1.e-10+0.j, 0.e+00-0.j],
        [0.e+00-0.j, 1.e+00+0.j],
    ])
    np.testing.assert_allclose(expected_rho_ss, rho_ss.full(), atol=1e-14)
    assert rho_ss.tr() == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize(['method', 'kwargs'], [
    pytest.param('direct', {}, id="direct"),
    pytest.param('direct', {'sparse':False}, id="direct_dense"),
    pytest.param('eigen', {'sparse':False}, id="eigen"),
    pytest.param('power', {'power_tol':1e-5}, id="power"),
    pytest.param('iterative-lgmres', {'tol': 1e-7, 'atol': 1e-7}, id="iterative-lgmres"),
    pytest.param('iterative-gmres', {'tol': 1e-7, 'atol': 1e-7}, id="iterative-gmres"),
    pytest.param('iterative-bicgstab', {'tol': 1e-7, 'atol': 1e-7}, id="iterative-bicgstab"),
])
def test_ho(method, kwargs):
    # thermal steadystate of an oscillator: compare numerics with analytical
    # formula
    if (
        pac_version.parse(scipy.__version__) >= pac_version.parse("1.12")
        and "tol" in kwargs
    ):
        # From scipy 1.12, the tol keyword is renamed to rtol
        kwargs["rtol"] = kwargs.pop("tol")

    a = qutip.destroy(30)
    H = 0.5 * 2 * np.pi * a.dag() * a
    gamma1 = 0.05

    wth_vec = np.linspace(0.1, 3, 11)
    p_ss = np.zeros(np.shape(wth_vec))

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (np.exp(1.0 / wth) - 1)  # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(np.sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(np.sqrt(rate) * a.dag())
        rho_ss = qutip.steadystate(H, c_op_list, method=method, **kwargs)
        p_ss[idx] = np.real(qutip.expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (np.exp(1.0 / wth_vec) - 1)
    np.testing.assert_allclose(p_ss_analytic, p_ss, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(['method', 'kwargs'], [
    pytest.param('direct', {}, id="direct"),
    pytest.param('direct', {'sparse':False}, id="direct_dense"),
    pytest.param('eigen', {}, id="eigen"),
    pytest.param('svd', {}, id="svd"),
    pytest.param('power', {}, id="power"),
    pytest.param('power-gmres', {"atol": 1e-5, 'tol':1e-1, 'use_precond':1},
                 id="power-gmres"),
    pytest.param('power', {"solver": "bicgstab", "atol": 1e-6, "tol": 1e-6, 'use_precond':1},
                 id="power-bicgstab"),
    pytest.param('iterative-gmres', {"atol": 1e-10, "tol": 1e-10}, id="iterative-gmres"),
    pytest.param('iterative-bicgstab', {"atol": 1e-10, "tol": 1e-10}, id="iterative-bicgstab"),
])
def test_driven_cavity(method, kwargs):
    if (
        pac_version.parse(scipy.__version__) >= pac_version.parse("1.12")
        and "tol" in kwargs
    ):
        # From scipy 1.12, the tol keyword is renamed to rtol
        kwargs["rtol"] = kwargs.pop("tol")

    N = 30
    Omega = 0.01 * 2 * np.pi
    Gamma = 0.05

    a = qutip.destroy(N)
    H = Omega * (a.dag() + a)
    c_ops = [np.sqrt(Gamma) * a]
    rho_ss = qutip.steadystate(H, c_ops, method=method, **kwargs).full()
    rho_ss_analytic = qutip.coherent_dm(N, -1.0j * (Omega)/(Gamma/2)).full()

    np.testing.assert_allclose(
        rho_ss, rho_ss_analytic, atol=1e-4
    )
    assert rho_ss.trace() == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize(['method', 'kwargs'], [
    pytest.param('solve', {}, id="dense_direct"),
    pytest.param('numpy', {}, id="dense_numpy"),
    pytest.param('spilu', {},  id="spilu"),
])
def test_pseudo_inverse(method, kwargs):
    N = 4
    a = qutip.destroy(N)
    H = (a.dag() + a)
    L = qutip.liouvillian(H, [a])
    rho = qutip.steadystate(L)
    Lpinv = qutip.pseudo_inverse(L, rho, method=method, **kwargs)
    np.testing.assert_allclose((L * Lpinv * L).full(), L.full(), atol=5e-14)
    np.testing.assert_allclose(
        (Lpinv * L * Lpinv).full(), Lpinv.full(),  atol=3e-14
    )
    assert rho.tr() == pytest.approx(1, abs=3e-15)


@pytest.mark.parametrize('sparse', [True, False])
def test_steadystate_floquet(sparse):
    """
    Test the steadystate solution for a periodically
    driven system.
    """
    N_c = 20

    a = qutip.destroy(N_c)
    a_d = a.dag()
    X_c = a + a_d

    w_c = 1

    A_l = 0.001
    w_l = w_c
    gam = 0.01

    H = w_c * a_d * a

    H_t = [H, [X_c, lambda t, args: args["A_l"] * np.cos(args["w_l"] * t)]]

    psi0 = qutip.fock(N_c, 0)

    args = {"A_l": A_l, "w_l": w_l}

    c_ops = []
    c_ops.append(np.sqrt(gam) * a)

    t_l = np.linspace(0, 20 / gam, 2000)

    expect_me = qutip.mesolve(H_t, psi0, t_l,
                        c_ops, [a_d * a], args=args).expect[0]

    rho_ss = qutip.steadystate_floquet(H, c_ops,
                                       A_l * X_c, w_l, n_it=3, sparse=sparse)
    expect_ss = qutip.expect(a_d * a, rho_ss)

    np.testing.assert_allclose(expect_me[-20:], expect_ss, atol=1e-3)
    assert rho_ss.tr() == pytest.approx(1, abs=1e-15)


def test_bad_options_steadystate():
    N = 4
    a = qutip.destroy(N)
    H = (a.dag() + a)
    c_ops = [a]
    with pytest.raises(ValueError):
        qutip.steadystate(H, c_ops, method='not a method')
    with pytest.raises(TypeError):
        qutip.steadystate(H, c_ops, method='direct', bad_opt=True)
    with pytest.raises(ValueError):
        qutip.steadystate(H, c_ops, method='direct', solver='Error')


def test_bad_options_pseudo_inverse():
    N = 4
    a = qutip.destroy(N)
    H = (a.dag() + a)
    L = qutip.liouvillian(H, [a])
    with pytest.raises(TypeError):
        qutip.pseudo_inverse(L, method='splu', bad_opt=True)
    with pytest.raises(ValueError):
        qutip.pseudo_inverse(L, method='not a method', sparse=False)
    with pytest.raises(ValueError):
        qutip.pseudo_inverse(L, method='not a method')


def test_bad_system():
    N = 4
    a = qutip.destroy(N)
    H = (a.dag() + a)
    with pytest.raises(TypeError):
        qutip.steadystate(H, [], method='direct')
    with pytest.raises(TypeError):
        qutip.steadystate(qutip.basis(N, N-1), [], method='direct')
