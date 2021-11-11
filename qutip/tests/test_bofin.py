"""
Tests for the Bosonic HEOM solvers.
"""

import numpy as np
import pytest
from numpy.linalg import eigvalsh
from scipy.integrate import quad

from qutip import (
    basis, destroy, expect, isequal, liouvillian, spre, spost, sigmax, sigmaz,
    Qobj, QobjEvo, Options,
)
from qutip.nonmarkov.bofin import (
    BathExponent,
    Bath,
    BosonicBath,
    DrudeLorentzBath,
    DrudeLorentzPadeBath,
    FermionicBath,
    HierarchyADOs,
    HierarchyADOsState,
    BosonicHEOMSolver,
    FermionicHEOMSolver,
    HSolverDL,
)


def check_exponent(
    exp, type, dim, Q, ck, vk, ck2=None, sigma_bar_k_offset=None, tag=None,
):
    """ Check the attributes of a BathExponent. """
    assert exp.type is BathExponent.types[type]
    assert exp.dim == dim
    assert exp.Q == Q
    assert exp.ck == ck
    assert exp.vk == vk
    assert exp.ck2 == ck2
    assert exp.sigma_bar_k_offset == sigma_bar_k_offset
    assert exp.tag == tag


class TestBathExponent:
    def test_create(self):
        exp_r = BathExponent("R", None, Q=None, ck=1.0, vk=2.0)
        check_exponent(exp_r, "R", None, Q=None, ck=1.0, vk=2.0)

        exp_i = BathExponent("I", None, Q=None, ck=1.0, vk=2.0)
        check_exponent(exp_i, "I", None, Q=None, ck=1.0, vk=2.0)

        exp_ri = BathExponent("RI", None, Q=None, ck=1.0, vk=2.0, ck2=3.0)
        check_exponent(exp_ri, "RI", None, Q=None, ck=1.0, vk=2.0, ck2=3.0)

        exp_p = BathExponent(
            "+", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=-1,
        )
        check_exponent(
            exp_p, "+", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=-1,
        )

        exp_m = BathExponent(
            "-", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=+1,
        )
        check_exponent(
            exp_m, "-", 2, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=+1,
        )

        exp_tag = BathExponent("R", None, Q=None, ck=1.0, vk=2.0, tag="tag1")
        check_exponent(exp_tag, "R", None, Q=None, ck=1.0, vk=2.0, tag="tag1")

        for exp_type, kw in [
            ("R", {}),
            ("I", {}),
            ("+", {"sigma_bar_k_offset": 1}),
            ("-", {"sigma_bar_k_offset": -1}),
        ]:
            with pytest.raises(ValueError) as err:
                BathExponent(
                    exp_type, None, Q=None, ck=1.0, vk=2.0, ck2=3.0, **kw,
                )
            assert str(err.value) == (
                "Second co-efficient (ck2) should only be specified for RI"
                " bath exponents"
            )

        for exp_type, kw in [("R", {}), ("I", {}), ("RI", {"ck2": 3.0})]:
            with pytest.raises(ValueError) as err:
                BathExponent(
                    exp_type, None, Q=None, ck=1.0, vk=2.0,
                    sigma_bar_k_offset=1, **kw,
                )
            assert str(err.value) == (
                "Offset of sigma bar (sigma_bar_k_offset) should only be"
                " specified for + and - bath exponents"
            )

    def test_repr(self):
        exp1 = BathExponent("R", None, Q=sigmax(), ck=1.0, vk=2.0)
        assert repr(exp1) == (
            "<BathExponent type=R dim=None Q.dims=[[2], [2]]"
            " ck=1.0 vk=2.0 ck2=None"
            " sigma_bar_k_offset=None tag=None>"
        )
        exp2 = BathExponent(
            "+", None, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=-1,
            tag="bath1",
        )
        assert repr(exp2) == (
            "<BathExponent type=+ dim=None Q.dims=None"
            " ck=1.0 vk=2.0 ck2=None"
            " sigma_bar_k_offset=-1 tag='bath1'>"
        )


class TestBath:
    def test_create(self):
        exp_r = BathExponent("R", None, Q=None, ck=1.0, vk=2.0)
        exp_i = BathExponent("I", None, Q=None, ck=1.0, vk=2.0)
        bath = Bath([exp_r, exp_i])
        assert bath.exponents == [exp_r, exp_i]


class TestBosonicBath:
    def test_create(self):
        Q = sigmaz()
        bath = BosonicBath(Q, [1.], [0.5], [2.], [0.6])
        exp_r, exp_i = bath.exponents
        check_exponent(exp_r, "R", dim=None, Q=Q, ck=1.0, vk=0.5)
        check_exponent(exp_i, "I", dim=None, Q=Q, ck=2.0, vk=0.6)

        bath = BosonicBath(Q, [1.], [0.5], [2.], [0.5])
        [exp_ri] = bath.exponents
        check_exponent(exp_ri, "RI", dim=None, Q=Q, ck=1.0, vk=0.5, ck2=2.0)

        bath = BosonicBath(Q, [1.], [0.5], [2.], [0.6], tag="bath1")
        exp_r, exp_i = bath.exponents
        check_exponent(exp_r, "R", dim=None, Q=Q, ck=1.0, vk=0.5, tag="bath1")
        check_exponent(exp_i, "I", dim=None, Q=Q, ck=2.0, vk=0.6, tag="bath1")

        with pytest.raises(ValueError) as err:
            BosonicBath("not-a-qobj", [1.], [0.5], [2.], [0.6])
        assert str(err.value) == "The coupling operator Q must be a Qobj."

        with pytest.raises(ValueError) as err:
            BosonicBath(Q, [1.], [], [2.], [0.6])
        assert str(err.value) == (
            "The bath exponent lists ck_real and vk_real, and ck_imag and"
            " vk_imag must be the same length."
        )

        with pytest.raises(ValueError) as err:
            BosonicBath(Q, [1.], [0.5], [2.], [])
        assert str(err.value) == (
            "The bath exponent lists ck_real and vk_real, and ck_imag and"
            " vk_imag must be the same length."
        )

    def test_combine(self):
        exp_ix = BathExponent("I", None, Q=sigmax(), ck=2.0, vk=0.5)
        exp_rx = BathExponent("R", None, Q=sigmax(), ck=1.0, vk=0.5)
        exp_rix = BathExponent("RI", None, Q=sigmax(), ck=0.1, vk=0.5, ck2=0.2)
        exp_rz = BathExponent("R", None, Q=sigmaz(), ck=1.0, vk=0.5)

        [exp] = BosonicBath.combine([exp_rx, exp_rx])
        check_exponent(exp, "R", dim=None, Q=sigmax(), ck=2.0, vk=0.5)

        [exp] = BosonicBath.combine([exp_ix, exp_ix])
        check_exponent(exp, "I", dim=None, Q=sigmax(), ck=4.0, vk=0.5)

        [exp] = BosonicBath.combine([exp_rix, exp_rix])
        check_exponent(
            exp, "RI", dim=None, Q=sigmax(), ck=0.2, vk=0.5, ck2=0.4,
        )

        [exp1, exp2] = BosonicBath.combine([exp_rx, exp_rz])
        check_exponent(exp1, "R", dim=None, Q=sigmax(), ck=1.0, vk=0.5)
        check_exponent(exp2, "R", dim=None, Q=sigmaz(), ck=1.0, vk=0.5)

        [exp] = BosonicBath.combine([exp_rx, exp_ix])
        check_exponent(
            exp, "RI", dim=None, Q=sigmax(), ck=1.0, vk=0.5, ck2=2.0,
        )

        [exp] = BosonicBath.combine([exp_rx, exp_ix, exp_rix])
        check_exponent(
            exp, "RI", dim=None, Q=sigmax(), ck=1.1, vk=0.5, ck2=2.2,
        )


class TestDrudeLorentzBath:
    def test_create(self):
        Q = sigmaz()
        ck_real = [0.05262168274189053, 0.0007958201977617107]
        vk_real = [0.05, 6.613879270715354]
        ck_imag = [-0.0012500000000000002]
        vk_imag = [0.05]

        bath = DrudeLorentzBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, tag="bath1",
        )
        [exp1, exp2] = bath.exponents
        check_exponent(
            exp1, "RI", dim=None, Q=Q,
            ck=ck_real[0], vk=vk_real[0], ck2=ck_imag[0],
            tag="bath1",
        )
        check_exponent(
            exp2, "R", dim=None, Q=Q,
            ck=ck_real[1], vk=vk_real[1],
            tag="bath1",
        )
        assert bath.delta is None
        assert bath.terminator is None

        bath = DrudeLorentzBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, combine=False,
        )
        [exp1, exp2, exp3] = bath.exponents
        check_exponent(exp1, "R", dim=None, Q=Q, ck=ck_real[0], vk=vk_real[0])
        check_exponent(exp2, "R", dim=None, Q=Q, ck=ck_real[1], vk=vk_real[1])
        check_exponent(exp3, "I", dim=None, Q=Q, ck=ck_imag[0], vk=vk_imag[0])

        bath = DrudeLorentzBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, terminator=True,
        )
        bath2 = DrudeLorentzBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, terminator=True, combine=False,
        )
        op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)

        assert np.abs(bath.delta - (0.00031039 / 4.0)) < 1e-8
        assert np.abs(bath2.delta - (0.00031039 / 4.0)) < 1e-8

        assert isequal(bath.terminator, - (0.00031039 / 4.0) * op, tol=1e-8)
        assert isequal(bath2.terminator, - (0.00031039 / 4.0) * op, tol=1e-8)


class TestDrudeLorentzPadeBath:
    def test_create(self):
        Q = sigmaz()
        ck_real = [0.05262168274189053, 0.0016138037466648136]
        vk_real = [0.05, 8.153649149910352]
        ck_imag = [-0.0012500000000000002]
        vk_imag = [0.05]

        bath = DrudeLorentzPadeBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, tag="bath1",
        )
        [exp1, exp2] = bath.exponents
        check_exponent(
            exp1, "RI", dim=None, Q=Q,
            ck=ck_real[0], vk=vk_real[0], ck2=ck_imag[0],
            tag="bath1",
        )
        check_exponent(
            exp2, "R", dim=None, Q=Q,
            ck=ck_real[1], vk=vk_real[1],
            tag="bath1",
        )

        bath = DrudeLorentzPadeBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, combine=False,
        )
        [exp1, exp2, exp3] = bath.exponents
        check_exponent(exp1, "R", dim=None, Q=Q, ck=ck_real[0], vk=vk_real[0])
        check_exponent(exp2, "R", dim=None, Q=Q, ck=ck_real[1], vk=vk_real[1])
        check_exponent(exp3, "I", dim=None, Q=Q, ck=ck_imag[0], vk=vk_imag[0])


class TestFermionicBath:
    def test_create(self):
        Q = sigmaz()
        bath = FermionicBath(Q, [1.], [0.5], [2.], [0.6])
        exp_p, exp_m = bath.exponents
        check_exponent(
            exp_p, "+", dim=2, Q=Q, ck=1.0, vk=0.5, sigma_bar_k_offset=1,
        )
        check_exponent(
            exp_m, "-", dim=2, Q=Q, ck=2.0, vk=0.6, sigma_bar_k_offset=-1,
        )

        bath = FermionicBath(Q, [1.], [0.5], [2.], [0.6], tag="bath1")
        exp_p, exp_m = bath.exponents
        check_exponent(
            exp_p, "+", dim=2, Q=Q, ck=1.0, vk=0.5, sigma_bar_k_offset=1,
            tag="bath1",
        )
        check_exponent(
            exp_m, "-", dim=2, Q=Q, ck=2.0, vk=0.6, sigma_bar_k_offset=-1,
            tag="bath1",
        )

        with pytest.raises(ValueError) as err:
            FermionicBath("not-a-qobj", [1.], [0.5], [2.], [0.6])
        assert str(err.value) == "The coupling operator Q must be a Qobj."

        with pytest.raises(ValueError) as err:
            FermionicBath(Q, [1.], [], [2.], [0.6])
        assert str(err.value) == (
            "The bath exponent lists ck_plus and vk_plus, and ck_minus and"
            " vk_minus must be the same length."
        )

        with pytest.raises(ValueError) as err:
            FermionicBath(Q, [1.], [0.5], [2.], [])
        assert str(err.value) == (
            "The bath exponent lists ck_plus and vk_plus, and ck_minus and"
            " vk_minus must be the same length."
        )


class TestHierarchyADOs:
    def mk_exponents(self, dims):
        return [
            BathExponent("I", dim, Q=None, ck=1.0, vk=2.0) for dim in dims
        ]

    def test_create(self):
        exponents = self.mk_exponents([2, 3])
        ados = HierarchyADOs(exponents, cutoff=2)

        assert ados.exponents == exponents
        assert ados.cutoff == 2

        assert ados.dims == [2, 3]
        assert ados.vk == [2.0, 2.0]
        assert ados.ck == [1.0, 1.0]
        assert ados.ck2 == [None, None]
        assert ados.sigma_bar_k_offset == [None, None]

        assert ados.labels == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
        ]

    def test_state_idx(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
        assert ados.idx((0, 0)) == 0
        assert ados.idx((0, 1)) == 1
        assert ados.idx((0, 2)) == 2
        assert ados.idx((1, 0)) == 3
        assert ados.idx((1, 1)) == 4

    def test_next(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
        assert ados.next((0, 0), 0) == (1, 0)
        assert ados.next((0, 0), 1) == (0, 1)
        assert ados.next((1, 0), 0) is None
        assert ados.next((1, 0), 1) == (1, 1)
        assert ados.next((1, 1), 1) is None

    def test_prev(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
        assert ados.prev((0, 0), 0) is None
        assert ados.prev((0, 0), 1) is None
        assert ados.prev((1, 0), 0) == (0, 0)
        assert ados.prev((0, 1), 1) == (0, 0)
        assert ados.prev((1, 1), 1) == (1, 0)
        assert ados.prev((0, 2), 1) == (0, 1)

    def test_exps(self):
        ados = HierarchyADOs(self.mk_exponents([3, 3, 2]), cutoff=4)
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
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
        assert ados.filter() == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
        ]

    def test_filter_by_level(self):
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
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
        ados = HierarchyADOs(self.mk_exponents([2, 3]), cutoff=2)
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
            "The cutoff for the hierarchy is 2 but 3 levels of excitation"
            " filters were given."
        )

        with pytest.raises(ValueError) as err:
            ados.filter(level=0, dims=[2])
        assert str(err.value) == (
            "The level parameter is 0 but 1 levels of excitation filters"
            " were given."
        )


class TestHierarchyADOsState:
    def mk_ados(self, bath_dims, cutoff):
        exponents = [
            BathExponent("I", dim, Q=None, ck=1.0, vk=2.0) for dim in bath_dims
        ]
        ados = HierarchyADOs(exponents, cutoff=cutoff)
        return ados

    def mk_rho_and_soln(self, ados, rho_dims):
        n_ados = len(ados.labels)
        ado_soln = np.random.rand(n_ados, *[2**d for d in rho_dims])
        rho = Qobj(ado_soln[0, :], dims=[2, 2])
        return rho, ado_soln

    def test_create(self):
        ados = self.mk_ados([2, 3], cutoff=2)
        rho, ado_soln = self.mk_rho_and_soln(ados, [2, 2])
        ado_state = HierarchyADOsState(rho, ados, ado_soln)
        assert ado_state.rho == rho
        assert ado_state.labels == ados.labels
        assert ado_state.exponents == ados.exponents
        assert ado_state.idx((0, 0)) == ados.idx((0, 0))
        assert ado_state.idx((0, 1)) == ados.idx((0, 1))

    def test_extract(self):
        ados = self.mk_ados([2, 3], cutoff=2)
        rho, ado_soln = self.mk_rho_and_soln(ados, [2, 2])
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
            0.5 * np.exp(quad(_integrand, 0, np.inf, args=(t,))[0])
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


class TestBosonicHEOMSolver:
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
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
        self, evo, combine, liouvillianize, atol=1e-3
    ):
        dlm = DrudeLorentzPureDephasingModel(
            lam=0.025, gamma=0.05, T=1/0.95, Nk=2,
        )
        ck_real, vk_real, ck_imag, vk_imag = dlm.bath_coefficients()
        H_sys = hamiltonian_to_sys(dlm.H, evo, liouvillianize)

        options = Options(nsteps=15000, store_states=True)
        hsolver = BosonicHEOMSolver(
            H_sys, dlm.Q, ck_real, vk_real, ck_imag, vk_imag,
            14, combine=combine, options=options,
        )

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


class TestHSolverDL:
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
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


class DiscreteLevelCurrentModel:
    """ Analytic discrete level current model for testing the HEOM solver.
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

        # right hand modes are the first k modes in ck/vk_plus and ck/vk_minus
        # and thus the first 2 * k exponents
        k = self.lmax + 1
        return 1.0j * sum(
            exp_sign(exp) * (exp.Q * aux).tr()
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
            if j == k:
                return 1.
            else:
                return 0.

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


class TestFermionicHEOMSolver:
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
    @pytest.mark.parametrize(['evo'], [
        pytest.param("qobj"),
        pytest.param("qobjevo"),
        pytest.param("listevo"),
    ])
    @pytest.mark.parametrize(['liouvillianize'], [
        pytest.param(False, id="hamiltonian"),
        pytest.param(True, id="liouvillian"),
    ])
    def test_discrete_level_model(self, evo, liouvillianize, atol=1e-3):
        """ Compare to discrete-level current analytics. """
        dlm = DiscreteLevelCurrentModel(
            gamma=0.01, W=1, T=0.025851991, lmax=10,
        )
        H_sys = hamiltonian_to_sys(dlm.H, evo, liouvillianize)
        ck_plus, vk_plus, ck_minus, vk_minus = dlm.bath_coefficients()

        options = Options(
            nsteps=15_000, store_states=True, rtol=1e-14, atol=1e-14,
        )
        Ncc = 2  # For a single impurity we converge with Ncc = 2
        hsolver = FermionicHEOMSolver(
            H_sys, dlm.Q, ck_plus, vk_plus, ck_minus, vk_minus, Ncc,
            options=options,
        )

        tlist = [0, 10]
        result = hsolver.run(dlm.rho(), tlist, ado_return=True)
        current = dlm.state_current(result.ado_states[-1])
        analytic_current = dlm.analytic_current()
        np.testing.assert_allclose(analytic_current, current, atol=atol)

        rho_final, ado_state = hsolver.steady_state()
        current = dlm.state_current(ado_state)
        analytic_current = dlm.analytic_current()
        np.testing.assert_allclose(analytic_current, current, atol=atol)
