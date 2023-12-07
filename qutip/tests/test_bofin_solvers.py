"""
Tests for qutip.nonmarkov.bofin_solvers.
"""

import numpy as np
import pytest
from numpy.linalg import eigvalsh
from scipy.integrate import quad
from scipy.sparse import csr_matrix

from qutip import (
    basis, destroy, expect, liouvillian, sigmax, sigmaz,
    tensor, Qobj, QobjEvo, Options,
)
from qutip.fastsparse import fast_csr_matrix
from qutip.nonmarkov.bofin_baths import (
    BathExponent,
    Bath,
    BosonicBath,
    DrudeLorentzBath,
    DrudeLorentzPadeBath,
    UnderDampedBath,
    FermionicBath,
    LorentzianBath,
    LorentzianPadeBath,
)
from qutip.nonmarkov.bofin_solvers import (
    HierarchyADOs,
    HierarchyADOsState,
    HEOMSolver,
    HSolverDL,
    _GatherHEOMRHS,
)
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar


class TestHierarchyADOs:
    def mk_exponents(self, dims):
        return [
            BathExponent("I", dim, Q=None, ck=1.0, vk=2.0) for dim in dims
        ]

    def test_create(self):
        exponents = self.mk_exponents([2, 3])
        ados = HierarchyADOs(exponents, max_depth=2)

        assert ados.exponents == exponents
        assert ados.max_depth == 2

        assert ados.dims == [2, 3]
        assert ados.vk == [2.0, 2.0]
        assert ados.ck == [1.0, 1.0]
        assert ados.ck2 == [None, None]
        assert ados.sigma_bar_k_offset == [None, None]

        assert ados.labels == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
        ]

    def test_state_idx(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), max_depth=2)
        assert ados.idx((0, 0)) == 0
        assert ados.idx((0, 1)) == 1
        assert ados.idx((0, 2)) == 2
        assert ados.idx((1, 0)) == 3
        assert ados.idx((1, 1)) == 4

    def test_next(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), max_depth=2)
        assert ados.next((0, 0), 0) == (1, 0)
        assert ados.next((0, 0), 1) == (0, 1)
        assert ados.next((1, 0), 0) is None
        assert ados.next((1, 0), 1) == (1, 1)
        assert ados.next((1, 1), 1) is None

    def test_prev(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), max_depth=2)
        assert ados.prev((0, 0), 0) is None
        assert ados.prev((0, 0), 1) is None
        assert ados.prev((1, 0), 0) == (0, 0)
        assert ados.prev((0, 1), 1) == (0, 0)
        assert ados.prev((1, 1), 1) == (1, 0)
        assert ados.prev((0, 2), 1) == (0, 1)

    def test_exps(self):
        ados = HierarchyADOs(self.mk_exponents([3, 3, 2]), max_depth=4)
        assert ados.exps((0, 0, 0)) == ()
        assert ados.exps((1, 0, 0)) == (ados.exponents[0],)
        assert ados.exps((2, 0, 0)) == (
            ados.exponents[0], ados.exponents[0],
        )
        assert ados.exps((1, 2, 1)) == (
            ados.exponents[0],
            ados.exponents[1], ados.exponents[1],
            ados.exponents[2],
        )

    def test_filter_by_nothing(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), max_depth=2)
        assert ados.filter() == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
        ]

    def test_filter_by_level(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), max_depth=2)
        assert ados.filter(level=0) == [
            (0, 0),
        ]
        assert ados.filter(level=1) == [
            (0, 1),
            (1, 0),
        ]
        assert ados.filter(level=2) == [
            (0, 2),
            (1, 1),
        ]
        assert ados.filter(level=3) == []

    def test_filter_by_exponents(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), max_depth=2)
        assert ados.filter(dims=[]) == [
            (0, 0),
        ]
        assert ados.filter(dims=[2]) == [
            (1, 0),
        ]
        assert ados.filter(level=1, dims=[2]) == [
            (1, 0),
        ]
        assert ados.filter(dims=[3]) == [
            (0, 1),
        ]
        assert ados.filter(dims=[2, 3]) == [
            (1, 1),
        ]
        assert ados.filter(level=2, dims=[2, 3]) == [
            (1, 1),
        ]
        assert ados.filter(dims=[3, 3]) == [
            (0, 2),
        ]
        assert ados.filter(types=["I"]) == [
            (0, 1),
            (1, 0),
        ]
        assert ados.filter(types=["I", "I"]) == [
            (0, 2),
            (1, 1),
        ]

        with pytest.raises(ValueError) as err:
            ados.filter(types=[], dims=[2])
        assert str(err.value) == (
            "The tags, dims and types filters must all be the same length."
        )

        with pytest.raises(ValueError) as err:
            ados.filter(dims=[2, 2, 2])
        assert str(err.value) == (
            "The maximum depth for the hierarchy is 2 but 3 levels of"
            " excitation filters were given."
        )

        with pytest.raises(ValueError) as err:
            ados.filter(level=0, dims=[2])
        assert str(err.value) == (
            "The level parameter is 0 but 1 levels of excitation filters"
            " were given."
        )


class TestHierarchyADOsState:
    def mk_ados(self, bath_dims, max_depth):
        exponents = [
            BathExponent("I", dim, Q=None, ck=1.0, vk=2.0) for dim in bath_dims
        ]
        ados = HierarchyADOs(exponents, max_depth=max_depth)
        return ados

    def mk_rho_and_soln(self, ados, rho_dims):
        n_ados = len(ados.labels)
        ado_soln = np.random.rand(n_ados, *[np.prod(d) for d in rho_dims])
        rho = Qobj(ado_soln[0, :], dims=rho_dims)
        return rho, ado_soln

    def test_create(self):
        ados = self.mk_ados([2, 3], max_depth=2)
        rho, ado_soln = self.mk_rho_and_soln(ados, [[2], [2]])
        ado_state = HierarchyADOsState(rho, ados, ado_soln)
        assert ado_state.rho == rho
        assert ado_state.labels == ados.labels
        assert ado_state.exponents == ados.exponents
        assert ado_state.idx((0, 0)) == ados.idx((0, 0))
        assert ado_state.idx((0, 1)) == ados.idx((0, 1))

    def test_extract(self):
        ados = self.mk_ados([2, 3], max_depth=2)
        rho, ado_soln = self.mk_rho_and_soln(ados, [[2], [2]])
        ado_state = HierarchyADOsState(rho, ados, ado_soln)
        ado_state.extract((0, 0)) == rho
        ado_state.extract(0) == rho
        ado_state.extract((0, 1)) == Qobj(ado_soln[1, :], dims=rho.dims)
        ado_state.extract(1) == Qobj(ado_soln[1, :], dims=rho.dims)


class DrudeLorentzPureDephasingModel:
    """ Analytic Drude-Lorentz pure-dephasing model for testing the HEOM solver.
    """
    def __init__(self, lam, gamma, T, Nk):
        self.lam = lam
        self.gamma = gamma
        self.T = T
        self.Nk = Nk
        # we add a very weak system hamiltonian here to avoid having
        # singular system that causes problems for the scipy.sparse.linalg
        # superLU solver used in spsolve.
        self.H = Qobj(1e-5 * np.ones((2, 2)))
        self.Q = sigmaz()

    def rho(self):
        """ Initial state. """
        return 0.5 * Qobj(np.ones((2, 2)))

    def state_results(self, states):
        projector = basis(2, 0) * basis(2, 1).dag()
        return expect(states, projector)

    def analytic_results(self, tlist):
        lam, gamma, T = self.lam, self.gamma, self.T
        lam_c = lam / np.pi

        def _integrand(omega, t):
            J = 2 * lam_c * omega * gamma / (omega**2 + gamma**2)
            return (-4 * J * (1 - np.cos(omega*t))
                    / (np.tanh(0.5*omega / T) * omega**2))

        # Calculate the analytical results by numerical integration
        return [
            0.5 * np.exp(quad(_integrand, 0, np.inf, args=(t,), limit=5000)[0])
            for t in tlist
        ]

    def bath_coefficients(self):
        """ Correlation function expansion coefficients for the Drude-Lorentz bath.
        """
        lam, gamma, T = self.lam, self.gamma, self.T
        Nk = self.Nk
        ck_real = [lam * gamma * (1 / np.tan(gamma / (2 * T)))]
        ck_real.extend([
            (4 * lam * gamma * T * 2 * np.pi * k * T /
                ((2 * np.pi * k * T)**2 - gamma**2))
            for k in range(1, Nk + 1)
        ])
        vk_real = [gamma]
        vk_real.extend([2 * np.pi * k * T for k in range(1, Nk + 1)])
        ck_imag = [lam * gamma * (-1.0)]
        vk_imag = [gamma]
        return ck_real, vk_real, ck_imag, vk_imag


class UnderdampedPureDephasingModel:
    """ Analytic Drude-Lorentz pure-dephasing model for testing the HEOM solver.
    """
    def __init__(self, lam,  gamma, w0, T, Nk):
        self.lam = lam
        self.gamma = gamma
        self.w0 = w0
        self.T = T
        self.Nk = Nk
        # we add a very weak system hamiltonian here to avoid having
        # singular system that causes problems for the scipy.sparse.linalg
        # superLU solver used in spsolve.
        self.H = Qobj(1e-5 * np.ones((2, 2)))
        self.Q = sigmaz()

    def rho(self):
        """ Initial state. """
        return 0.5 * Qobj(np.ones((2, 2)))

    def state_results(self, states):
        projector = basis(2, 0) * basis(2, 1).dag()
        return expect(states, projector)

    def analytic_results(self, tlist):
        lam, gamma, w0, T = self.lam, self.gamma, self.w0, self.T
        lam_c = lam**2 / np.pi

        def _integrand(omega, t):
            Jw = (
                lam_c * omega * gamma /
                ((w0**2 - omega**2)**2 + gamma**2 * omega**2)
            )
            return (-4 * Jw * (1 - np.cos(omega*t))
                    / (np.tanh(0.5*omega / T) * omega**2))

        return [
            0.5 * np.exp(quad(_integrand, 0, np.inf, args=(t,), limit=5000)[0])
            for t in tlist
        ]


class DiscreteLevelCurrentModel:
    """ Analytic discrete level current model for testing the HEOM solver
        with a fermionic bath.
    """
    def __init__(self, gamma, W, T, lmax):
        self.gamma = gamma
        self.W = W
        self.T = T
        self.lmax = lmax  # Pade cut-off
        self.beta = 1. / T

        # single fermion
        self.e1 = 1.
        d1 = destroy(2)
        self.H = self.e1 * d1.dag() * d1
        self.Q = d1

        # bias
        self.theta = 2.

    def rho(self):
        """ Initial state. """
        return 0.5 * Qobj(np.ones((2, 2)))

    def state_current(self, ado_state):
        level_1_aux = [
            (ado_state.extract(label), ado_state.exps(label)[0])
            for label in ado_state.filter(level=1)
        ]

        def exp_sign(exp):
            return 1 if exp.type == exp.types["+"] else -1

        def exp_op(exp):
            return exp.Q if exp.type == exp.types["+"] else exp.Q.dag()

        # right hand modes are the first k modes in ck/vk_plus and ck/vk_minus
        # and thus the first 2 * k exponents
        k = self.lmax + 1
        return 1.0j * sum(
            exp_sign(exp) * (exp_op(exp) * aux).tr()
            for aux, exp in level_1_aux[:2 * k]
        )

    def analytic_current(self):
        Gamma, W, beta, e1 = self.gamma, self.W, self.beta, self.e1
        mu_l = self.theta / 2.
        mu_r = - self.theta / 2.

        def f(x):
            return 1 / (np.exp(x) + 1.)

        def Gamma_w(w, mu):
            return Gamma * W**2 / ((w-mu)**2 + W**2)

        def lamshift(w, mu):
            return (w-mu)*Gamma_w(w, mu)/(2*W)

        def integrand(w):
            return (
                ((2 / (np.pi)) * Gamma_w(w, mu_l) * Gamma_w(w, mu_r) *
                    (f(beta * (w - mu_l)) - f(beta * (w - mu_r)))) /
                ((Gamma_w(w, mu_l) + Gamma_w(w, mu_r))**2 + 4 *
                    (w - e1 - lamshift(w, mu_l) - lamshift(w, mu_r))**2)
            )

        def real_func(x):
            return np.real(integrand(x))

        def imag_func(x):
            return np.imag(integrand(x))

        # These integral bounds should be checked to be wide enough if the
        # parameters are changed
        a = -2
        b = 2
        real_integral = quad(real_func, a, b)
        imag_integral = quad(imag_func, a, b)

        return real_integral[0] + 1.0j * imag_integral[0]

    def bath_coefficients(self):
        """ Correlation function expansion coefficients for the fermionic bath.
        """
        Gamma, W, beta, lmax = self.gamma, self.W, self.beta, self.lmax
        mu_l = self.theta / 2.
        mu_r = - self.theta / 2.

        def deltafun(j, k):
            return 1. if j == k else 0.

        Alpha = np.zeros((2 * lmax, 2 * lmax))
        for j in range(2*lmax):
            for k in range(2*lmax):
                Alpha[j][k] = (
                    (deltafun(j, k + 1) + deltafun(j, k - 1))
                    / np.sqrt((2 * (j + 1) - 1) * (2 * (k + 1) - 1))
                )

        eigvalsA = eigvalsh(Alpha)

        eps = []
        for val in eigvalsA[0:lmax]:
            eps.append(-2 / val)

        AlphaP = np.zeros((2 * lmax - 1, 2 * lmax - 1))
        for j in range(2 * lmax - 1):
            for k in range(2 * lmax - 1):
                AlphaP[j][k] = (
                    (deltafun(j, k + 1) + deltafun(j, k - 1))
                    / np.sqrt((2 * (j + 1) + 1) * (2 * (k + 1) + 1))
                )

        eigvalsAP = eigvalsh(AlphaP)

        chi = []
        for val in eigvalsAP[0:lmax - 1]:
            chi.append(-2/val)

        eta_list = [
            0.5 * lmax * (2 * (lmax + 1) - 1) * (
                np.prod([chi[k]**2 - eps[j]**2 for k in range(lmax - 1)]) /
                np.prod([
                    eps[k]**2 - eps[j]**2 + deltafun(j, k) for k in range(lmax)
                ])
            )
            for j in range(lmax)
        ]

        kappa = [0] + eta_list
        epsilon = [0] + eps

        def f_approx(x):
            f = 0.5
            for ll in range(1, lmax + 1):
                f = f - 2 * kappa[ll] * x / (x**2 + epsilon[ll]**2)
            return f

        def C(sigma, mu):
            eta_0 = 0.5 * Gamma * W * f_approx(1.0j * beta * W)
            gamma_0 = W - sigma*1.0j*mu
            eta_list = [eta_0]
            gamma_list = [gamma_0]
            if lmax > 0:
                for ll in range(1, lmax + 1):
                    eta_list.append(
                        -1.0j * (kappa[ll] / beta) * Gamma * W**2
                        / (-(epsilon[ll]**2 / beta**2) + W**2)
                    )
                    gamma_list.append(epsilon[ll]/beta - sigma*1.0j*mu)
            return eta_list, gamma_list

        etapL, gampL = C(1.0, mu_l)
        etamL, gammL = C(-1.0, mu_l)

        etapR, gampR = C(1.0, mu_r)
        etamR, gammR = C(-1.0, mu_r)

        ck_plus = etapR + etapL
        vk_plus = gampR + gampL
        ck_minus = etamR + etamL
        vk_minus = gammR + gammL

        return ck_plus, vk_plus, ck_minus, vk_minus


_HAMILTONIAN_EVO_KINDS = {
    "qobj": lambda H: H,
    "qobjevo": lambda H: QobjEvo([H]),
    "listevo": lambda H: [H],
}


def hamiltonian_to_sys(H, evo, liouvillianize):
    if liouvillianize:
        H = liouvillian(H)
    H = _HAMILTONIAN_EVO_KINDS[evo](H)
    return H


class TestHEOMSolver:
    def test_create_bosonic(self):
        Q = sigmaz()
        H = sigmax()
        exponents = [
            BathExponent("R", None, Q=Q, ck=1.1, vk=2.1),
            BathExponent("I", None, Q=Q, ck=1.2, vk=2.2),
            BathExponent("RI", None, Q=Q, ck=1.3, vk=2.3, ck2=3.3),
        ]
        bath = Bath(exponents)

        hsolver = HEOMSolver(H, bath, 2)
        assert hsolver.ados.exponents == exponents
        assert hsolver.ados.max_depth == 2

        hsolver = HEOMSolver(H, [bath] * 3, 2)
        assert hsolver.ados.exponents == exponents * 3
        assert hsolver.ados.max_depth == 2

    def test_create_fermionic(self):
        Q = sigmaz()
        H = sigmax()
        exponents = [
            BathExponent("+", 2, Q=Q, ck=1.1, vk=2.1, sigma_bar_k_offset=1),
            BathExponent("-", 2, Q=Q, ck=1.2, vk=2.2, sigma_bar_k_offset=-1),
        ]
        bath = Bath(exponents)

        hsolver = HEOMSolver(H, bath, 2)
        assert hsolver.ados.exponents == exponents
        assert hsolver.ados.max_depth == 2

        hsolver = HEOMSolver(H, [bath] * 3, 2)
        assert hsolver.ados.exponents == exponents * 3
        assert hsolver.ados.max_depth == 2

    def test_create_progress_bar(self):
        Q = sigmaz()
        H = sigmax()
        bath = Bath([
            BathExponent("R", None, Q=Q, ck=1.1, vk=2.1),
        ])

        hsolver = HEOMSolver(H, bath, 2)
        assert isinstance(hsolver.progress_bar, BaseProgressBar)

        hsolver = HEOMSolver(H, bath, 2, progress_bar=True)
        assert isinstance(hsolver.progress_bar, TextProgressBar)

        hsolver = HEOMSolver(H, bath, 2, progress_bar=TextProgressBar())
        assert isinstance(hsolver.progress_bar, BaseProgressBar)

        with pytest.raises(TypeError):
            HEOMSolver(H, bath, 2, progress_bar=False)

    def test_create_bath_errors(self):
        Q = sigmaz()
        H = sigmax()
        mixed_types = [
            BathExponent("+", 2, Q=Q, ck=1.1, vk=2.1, sigma_bar_k_offset=1),
            BathExponent("-", 2, Q=Q, ck=1.2, vk=2.2, sigma_bar_k_offset=-1),
            BathExponent("R", 2, Q=Q, ck=1.2, vk=2.2),
        ]
        mixed_q_dims = [
            BathExponent("I", 2, Q=tensor(Q, Q), ck=1.2, vk=2.2),
            BathExponent("R", 2, Q=Q, ck=1.2, vk=2.2),
        ]

        with pytest.raises(ValueError) as err:
            HEOMSolver(H, Bath(mixed_types), 2)
        assert str(err.value) == (
            "Bath exponents are currently restricted to being either all"
            " bosonic or all fermionic, but a mixture of bath exponents was"
            " given."
        )

        with pytest.raises(ValueError) as err:
            HEOMSolver(H, Bath(mixed_q_dims), 2)
        assert str(err.value) == (
            "All bath exponents must have system coupling operators with the"
            " same dimensions but a mixture of dimensions was given."
        )

    def test_create_h_sys_errors(self):
        H = object()

        with pytest.raises(TypeError) as err:
            HEOMSolver(H, Bath([]), 2)
        assert str(err.value) == (
            "Hamiltonian (H_sys) has unsupported type: <class 'object'>"
        )

        with pytest.raises(ValueError) as err:
            HEOMSolver([H], Bath([]), 2)
        assert str(err.value) == (
            "Hamiltonian (H_sys) of type list cannot be converted to QObjEvo"
        )

    @pytest.mark.parametrize(['evo', 'combine'], [
        pytest.param("qobj", True, id="qobj"),
        pytest.param("qobjevo", True, id="qobjevo"),
        pytest.param("listevo", True, id="listevo"),
    ])
    @pytest.mark.parametrize(['liouvillianize'], [
        pytest.param(False, id="hamiltonian"),
        pytest.param(True, id="liouvillian"),
    ])
    def test_pure_dephasing_model_bosonic_bath(
        self, evo, combine, liouvillianize, atol=1e-3
    ):
        dlm = DrudeLorentzPureDephasingModel(
            lam=0.025, gamma=0.05, T=1/0.95, Nk=2,
        )
        ck_real, vk_real, ck_imag, vk_imag = dlm.bath_coefficients()
        H_sys = hamiltonian_to_sys(dlm.H, evo, liouvillianize)

        bath = BosonicBath(dlm.Q, ck_real, vk_real, ck_imag, vk_imag)
        options = Options(nsteps=15000, store_states=True)
        hsolver = HEOMSolver(H_sys, bath, 14, options=options)

        tlist = np.linspace(0, 10, 21)
        result = hsolver.run(dlm.rho(), tlist)

        test = dlm.state_results(result.states)
        expected = dlm.analytic_results(tlist)
        np.testing.assert_allclose(test, expected, atol=atol)

        rho_final, ado_state = hsolver.steady_state()
        test = dlm.state_results([rho_final])
        expected = dlm.analytic_results([100])
        np.testing.assert_allclose(test, expected, atol=atol)
        assert rho_final == ado_state.extract(0)

    @pytest.mark.parametrize(['terminator'], [
        pytest.param(True, id="terminator"),
        pytest.param(False, id="noterminator"),
    ])
    @pytest.mark.parametrize(['bath_cls'], [
        pytest.param(DrudeLorentzBath, id="matsubara"),
        pytest.param(DrudeLorentzPadeBath, id="pade"),
    ])
    def test_pure_dephasing_model_drude_lorentz_baths(
        self, terminator, bath_cls, atol=1e-3
    ):
        dlm = DrudeLorentzPureDephasingModel(
            lam=0.025, gamma=0.05, T=1/0.95, Nk=2,
        )
        bath = bath_cls(
            Q=dlm.Q, lam=dlm.lam, gamma=dlm.gamma, T=dlm.T, Nk=dlm.Nk,
        )
        if terminator:
            _, terminator_op = bath.terminator()
            H_sys = liouvillian(dlm.H) + terminator_op
        else:
            H_sys = dlm.H

        options = Options(nsteps=15000, store_states=True)
        hsolver = HEOMSolver(H_sys, bath, 14, options=options)

        tlist = np.linspace(0, 10, 21)
        result = hsolver.run(dlm.rho(), tlist)

        test = dlm.state_results(result.states)
        expected = dlm.analytic_results(tlist)
        np.testing.assert_allclose(test, expected, atol=atol)

        rho_final, ado_state = hsolver.steady_state()
        test = dlm.state_results([rho_final])
        expected = dlm.analytic_results([100])
        np.testing.assert_allclose(test, expected, atol=atol)
        assert rho_final == ado_state.extract(0)

    def test_underdamped_pure_dephasing_model_underdamped_bath(
        self, atol=1e-3
    ):
        udm = UnderdampedPureDephasingModel(
            lam=0.1, gamma=0.05, w0=1, T=1/0.95, Nk=2,
        )
        bath = UnderDampedBath(
            Q=udm.Q, lam=udm.lam, T=udm.T, Nk=udm.Nk, gamma=udm.gamma,
            w0=udm.w0,
        )

        options = Options(nsteps=15000, store_states=True)
        hsolver = HEOMSolver(udm.H, bath, 14, options=options)

        tlist = np.linspace(0, 10, 21)
        result = hsolver.run(udm.rho(), tlist)

        test = udm.state_results(result.states)
        expected = udm.analytic_results(tlist)
        np.testing.assert_allclose(test, expected, atol=atol)

        rho_final, ado_state = hsolver.steady_state()
        test = udm.state_results([rho_final])
        expected = udm.analytic_results([5000])
        np.testing.assert_allclose(test, expected, atol=atol)
        assert rho_final == ado_state.extract(0)

    @pytest.mark.parametrize(['evo'], [
        pytest.param("qobj"),
        pytest.param("qobjevo"),
        pytest.param("listevo"),
    ])
    @pytest.mark.parametrize(['liouvillianize'], [
        pytest.param(False, id="hamiltonian"),
        pytest.param(True, id="liouvillian"),
    ])
    def test_discrete_level_model_fermionic_bath(self, evo, liouvillianize):
        dlm = DiscreteLevelCurrentModel(
            gamma=0.01, W=1, T=0.025851991, lmax=10,
        )
        H_sys = hamiltonian_to_sys(dlm.H, evo, liouvillianize)
        ck_plus, vk_plus, ck_minus, vk_minus = dlm.bath_coefficients()

        options = Options(
            nsteps=15_000, store_states=True, rtol=1e-7, atol=1e-7,
        )
        bath = FermionicBath(dlm.Q, ck_plus, vk_plus, ck_minus, vk_minus)
        # for a single impurity we converge with max_depth = 2
        hsolver = HEOMSolver(H_sys, bath, 2, options=options)

        tlist = [0, 600]
        result = hsolver.run(dlm.rho(), tlist, ado_return=True)
        current = dlm.state_current(result.ado_states[-1])
        analytic_current = dlm.analytic_current()
        np.testing.assert_allclose(analytic_current, current, rtol=1e-3)

        rho_final, ado_state = hsolver.steady_state()
        current = dlm.state_current(ado_state)
        analytic_current = dlm.analytic_current()
        np.testing.assert_allclose(analytic_current, current, rtol=1e-3)

    @pytest.mark.parametrize(['bath_cls', 'analytic_current'], [
        pytest.param(LorentzianBath, 0.001101, id="matsubara"),
        pytest.param(LorentzianPadeBath, 0.000813, id="pade"),
    ])
    def test_discrete_level_model_lorentzian_baths(
        self, bath_cls, analytic_current, atol=1e-3
    ):
        dlm = DiscreteLevelCurrentModel(
            gamma=0.01, W=1, T=0.025851991, lmax=10,
        )

        options = Options(
            nsteps=15_000, store_states=True, rtol=1e-7, atol=1e-7,
        )
        bath_l = bath_cls(
            dlm.Q, gamma=dlm.gamma, w=dlm.W, T=dlm.T, mu=dlm.theta / 2,
            Nk=dlm.lmax,
        )
        bath_r = bath_cls(
            dlm.Q, gamma=dlm.gamma, w=dlm.W, T=dlm.T, mu=- dlm.theta / 2,
            Nk=dlm.lmax,
        )
        # for a single impurity we converge with max_depth = 2
        hsolver = HEOMSolver(dlm.H, [bath_r, bath_l], 2, options=options)

        tlist = [0, 600]
        result = hsolver.run(dlm.rho(), tlist, ado_return=True)
        current = dlm.state_current(result.ado_states[-1])
        # analytic_current = dlm.analytic_current()
        np.testing.assert_allclose(analytic_current, current, rtol=1e-3)

        rho_final, ado_state = hsolver.steady_state()
        current = dlm.state_current(ado_state)
        # analytic_current = dlm.analytic_current()
        np.testing.assert_allclose(analytic_current, current, rtol=1e-3)


    @pytest.mark.parametrize(['ado_format'], [
        pytest.param("hierarchy-ados-state", id="hierarchy-ados-state"),
        pytest.param("numpy", id="numpy"),
    ])
    def test_ado_return_and_ado_init(self, ado_format):
        dlm = DrudeLorentzPureDephasingModel(
            lam=0.025, gamma=0.05, T=1/0.95, Nk=2,
        )
        ck_real, vk_real, ck_imag, vk_imag = dlm.bath_coefficients()

        bath = BosonicBath(dlm.Q, ck_real, vk_real, ck_imag, vk_imag)
        options = Options(nsteps=15_000, store_states=True)
        hsolver = HEOMSolver(dlm.H, bath, 6, options=options)

        tlist_1 = [0, 1, 2]
        result_1 = hsolver.run(dlm.rho(), tlist_1, ado_return=True)

        tlist_2 = [2, 3, 4]
        rho0 = result_1.ado_states[-1]
        if ado_format == "numpy":
            rho0 = rho0._ado_state  # extract the raw numpy array
        result_2 = hsolver.run(
            rho0, tlist_2, ado_return=True, ado_init=True,
        )

        tlist_full = tlist_1 + tlist_2[1:]
        result_full = hsolver.run(dlm.rho(), tlist_full, ado_return=True)

        times_12 = result_1.times + result_2.times[1:]
        times_full = result_full.times
        assert times_12 == tlist_full
        assert times_full == tlist_full

        ado_states_12 = result_1.ado_states + result_2.ado_states[1:]
        ado_states_full = result_full.ado_states
        assert len(ado_states_12) == len(tlist_full)
        assert len(ado_states_full) == len(tlist_full)
        for ado_12, ado_full in zip(ado_states_12, ado_states_full):
            for label in hsolver.ados.labels:
                np.testing.assert_allclose(
                    ado_12.extract(label).full(),
                    ado_full.extract(label).full(),
                    atol=1e-6,
                )

        states_12 = result_1.states + result_2.states[1:]
        states_full = result_full.states
        assert len(states_12) == len(tlist_full)
        assert len(states_full) == len(tlist_full)
        for ado_12, state_12 in zip(ado_states_12, states_12):
            assert ado_12.rho == state_12
        for ado_full, state_full in zip(ado_states_full, states_full):
            assert ado_full.rho == state_full

        expected = dlm.analytic_results(tlist_full)
        test_12 = dlm.state_results(states_12)
        np.testing.assert_allclose(test_12, expected, atol=1e-3)
        test_full = dlm.state_results(states_full)
        np.testing.assert_allclose(test_full, expected, atol=1e-3)


class TestHSolverDL:
    @pytest.mark.parametrize(['bnd_cut_approx', 'atol'], [
        pytest.param(True, 1e-4, id="bnd_cut_approx"),
        pytest.param(False,  1e-3, id="no_bnd_cut_approx"),
    ])
    @pytest.mark.parametrize(['evo', 'combine'], [
        pytest.param("qobj", True, id="qobj-combined"),
        pytest.param("qobjevo", True, id="qobjevo-combined"),
        pytest.param("listevo", True, id="listevo-combined"),
        pytest.param("qobj", False, id="qobj-uncombined"),
    ])
    @pytest.mark.parametrize(['liouvillianize'], [
        pytest.param(False, id="hamiltonian"),
        pytest.param(True, id="liouvillian"),
    ])
    def test_pure_dephasing_model(
        self, bnd_cut_approx, atol, evo, combine, liouvillianize,
    ):
        dlm = DrudeLorentzPureDephasingModel(
            lam=0.025, gamma=0.05, T=1/0.95, Nk=2,
        )
        ck_real, vk_real, ck_imag, vk_imag = dlm.bath_coefficients()
        H_sys = hamiltonian_to_sys(dlm.H, evo, liouvillianize)

        options = Options(nsteps=15_000, store_states=True)
        hsolver = HSolverDL(H_sys, dlm.Q, dlm.lam, dlm.T,
                            14, 2, dlm.gamma,
                            bnd_cut_approx=bnd_cut_approx,
                            options=options, combine=combine)

        tlist = np.linspace(0, 10, 21)
        result = hsolver.run(dlm.rho(), tlist)

        test = dlm.state_results(result.states)
        expected = dlm.analytic_results(tlist)
        np.testing.assert_allclose(test, expected, atol=atol)

        rho_final, ado_state = hsolver.steady_state()
        test = dlm.state_results([rho_final])
        expected = dlm.analytic_results([100])
        np.testing.assert_allclose(test, expected, atol=atol)
        assert rho_final == ado_state.extract(0)

    @pytest.mark.parametrize(['bnd_cut_approx', 'tol'], [
        pytest.param(True, 1e-4, id="bnd_cut_approx"),
        pytest.param(False, 1e-3, id="renorm"),
    ])
    def test_hsolverdl_backwards_compatibility(self, bnd_cut_approx, tol):
        # This is an exact copy of the pre-4.7 QuTiP HSolverDL test and
        # is repeated here to ensure the new HSolverDL remains compatibile
        # with the old one until it is removed.
        cut_frequency = 0.05
        coupling_strength = 0.025
        lam_c = coupling_strength / np.pi
        temperature = 1 / 0.95
        times = np.linspace(0, 10, 21)

        def _integrand(omega, t):
            J = 2*lam_c * omega * cut_frequency / (omega**2 + cut_frequency**2)
            return (-4 * J * (1 - np.cos(omega*t))
                    / (np.tanh(0.5*omega / temperature) * omega**2))

        # Calculate the analytical results by numerical integration
        expected = [
            0.5*np.exp(quad(_integrand, 0, np.inf, args=(t,), limit=5000)[0])
            for t in times
        ]

        H_sys = Qobj(np.zeros((2, 2)))
        Q = sigmaz()
        initial_state = 0.5*Qobj(np.ones((2, 2)))
        projector = basis(2, 0) * basis(2, 1).dag()
        options = Options(nsteps=15_000, store_states=True)
        hsolver = HSolverDL(H_sys, Q, coupling_strength, temperature,
                            20, 2, cut_frequency,
                            bnd_cut_approx=bnd_cut_approx,
                            options=options)
        test = expect(hsolver.run(initial_state, times).states, projector)
        np.testing.assert_allclose(test, expected, atol=tol)

    @pytest.mark.filterwarnings("ignore:zvode.*Excess work done:UserWarning")
    def test_integration_error(self):
        dlm = DrudeLorentzPureDephasingModel(
            lam=0.025, gamma=0.05, T=1/0.95, Nk=2,
        )
        ck_real, vk_real, ck_imag, vk_imag = dlm.bath_coefficients()

        bath = BosonicBath(dlm.Q, ck_real, vk_real, ck_imag, vk_imag)
        options = Options(nsteps=10)
        hsolver = HEOMSolver(dlm.H, bath, 14, options=options)

        with pytest.raises(RuntimeError) as err:
            hsolver.run(dlm.rho(), tlist=[0, 10])

        assert str(err.value) == (
            "HEOMSolver ODE integration error. Try increasing the nsteps given"
            " in the HEOMSolver options (which increases the allowed substeps"
            " in each step between times given in tlist)."
        )


class Test_GatherHEOMRHS:
    def test_simple_gather(self):
        def f(label):
            return int(label.lstrip("o"))

        gather_heoms = _GatherHEOMRHS(f, block=2, nhe=3)

        for i in range(3):
            for j in range(3):
                base = 10 * (j * 2) + (i * 2)
                block_op = csr_matrix(np.array([
                    [base, base + 10],
                    [base + 1, base + 11],
                ], dtype=np.complex128))
                gather_heoms.add_op(f"o{i}", f"o{j}", block_op)

        op = gather_heoms.gather()

        expected_op = np.array([
            [10 * i + j for i in range(2 * 3)]
            for j in range(2 * 3)
        ], dtype=np.complex128)

        np.testing.assert_array_equal(op.toarray(), expected_op)
        assert isinstance(op, fast_csr_matrix)
