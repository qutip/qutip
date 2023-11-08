"""
Tests for qutip.nonmarkov.bofin_baths.
"""

import numpy as np
import pytest

from qutip import spre, spost, sigmax, sigmaz
from qutip.core import data as _data
from qutip.solver.heom.bofin_baths import (
    BathExponent,
    Bath,
    BosonicBath,
    DrudeLorentzBath,
    DrudeLorentzPadeBath,
    UnderDampedBath,
    FermionicBath,
    LorentzianBath,
    LorentzianPadeBath,
    FitCorr,
    FitSpectral,
    OhmicBath
)


def isequal(Q1, Q2, tol):
    """ Return true if Q1 and Q2 are equal to within the given tolerance. """
    return _data.iszero(_data.sub(Q1.data, Q2.data), tol=tol)


def check_exponent(
    exp, type, dim, Q, ck, vk, ck2=None, sigma_bar_k_offset=None, tag=None,
):
    """ Check the attributes of a BathExponent. """
    assert exp.type is BathExponent.types[type]
    assert exp.fermionic == (type in ["+", "-"])
    assert exp.dim == dim
    assert exp.Q == Q
    assert exp.ck == pytest.approx(ck)
    assert exp.vk == pytest.approx(vk)
    assert exp.ck2 == pytest.approx(ck2)
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
            " sigma_bar_k_offset=None fermionic=False tag=None>"
        )
        exp2 = BathExponent(
            "+", None, Q=None, ck=1.0, vk=2.0, sigma_bar_k_offset=-1,
            tag="bath1",
        )
        assert repr(exp2) == (
            "<BathExponent type=+ dim=None Q.dims=None"
            " ck=1.0 vk=2.0 ck2=None"
            " sigma_bar_k_offset=-1 fermionic=True tag='bath1'>"
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

        bath = DrudeLorentzBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, combine=False,
        )
        [exp1, exp2, exp3] = bath.exponents
        check_exponent(exp1, "R", dim=None, Q=Q, ck=ck_real[0], vk=vk_real[0])
        check_exponent(exp2, "R", dim=None, Q=Q, ck=ck_real[1], vk=vk_real[1])
        check_exponent(exp3, "I", dim=None, Q=Q, ck=ck_imag[0], vk=vk_imag[0])

    @pytest.mark.parametrize(['combine'], [
        pytest.param(True, id="combine"),
        pytest.param(False, id="no-combine"),
    ])
    def test_terminator(self, combine):
        Q = sigmaz()
        op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)

        bath = DrudeLorentzBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, combine=combine,
        )
        delta, terminator = bath.terminator()

        assert np.abs(delta - (0.00031039 / 4.0)) < 1e-8
        assert isequal(terminator, - (0.00031039 / 4.0) * op, tol=1e-8)


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

    @pytest.mark.parametrize(['combine'], [
        pytest.param(True, id="combine"),
        pytest.param(False, id="no-combine"),
    ])
    def test_terminator(self, combine):
        Q = sigmaz()
        op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)

        bath = DrudeLorentzPadeBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, combine=combine,
        )
        delta, terminator = bath.terminator()

        assert np.abs(delta - (0.0 / 4.0)) < 1e-8
        assert isequal(terminator, - (0.0 / 4.0) * op, tol=1e-8)


class TestUnderDampedBath:
    def test_create(self):
        Q = sigmaz()
        ck_real = [
            0.0003533235200322013-7.634500355085952e-06j,
            0.0003533235200322013+7.634500355085952e-06j,
            -2.173594134367819e-07+0j,
        ]
        vk_real = [
            0.025-0.9996874511566103j,
            0.025+0.9996874511566103j,
            6.613879270715354,
        ]
        ck_imag = [0.00015629885102511106j, -0.00015629885102511106j]
        vk_imag = [
            0.025-0.9996874511566103j,
            0.025+0.9996874511566103j,
        ]

        bath = UnderDampedBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, w0=1, tag="bath1",
        )
        [exp1, exp2, exp3] = bath.exponents
        check_exponent(
            exp1, "RI", dim=None, Q=Q,
            ck=ck_real[0], vk=vk_real[0], ck2=ck_imag[0],
            tag="bath1",
        )
        check_exponent(
            exp2, "RI", dim=None, Q=Q,
            ck=ck_real[1], vk=vk_real[1], ck2=ck_imag[1],
            tag="bath1",
        )
        check_exponent(
            exp3, "R", dim=None, Q=Q,
            ck=ck_real[2], vk=vk_real[2],
            tag="bath1",
        )

        bath = UnderDampedBath(
            Q=Q, lam=0.025, T=1 / 0.95, Nk=1, gamma=0.05, w0=1, combine=False,
        )
        [exp1, exp2, exp3, exp4, exp5] = bath.exponents
        check_exponent(exp1, "R", dim=None, Q=Q, ck=ck_real[0], vk=vk_real[0])
        check_exponent(exp2, "R", dim=None, Q=Q, ck=ck_real[1], vk=vk_real[1])
        check_exponent(exp3, "R", dim=None, Q=Q, ck=ck_real[2], vk=vk_real[2])
        check_exponent(exp4, "I", dim=None, Q=Q, ck=ck_imag[0], vk=vk_imag[0])
        check_exponent(exp5, "I", dim=None, Q=Q, ck=ck_imag[1], vk=vk_imag[1])


class TestFermionicBath:
    def test_create(self):
        Q = 1j * sigmaz()
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


class TestLorentzianBath:
    def test_create(self):
        Q = sigmaz()
        ck_plus = [
            0.0025-0.0013376935246402686j,
            -0.00026023645770006167j,
            -0.0002748355116913478j,
        ]
        vk_plus = [
            1+1j,
            0.08121642500626945+1j,
            0.24364927501880834+1j,
        ]
        ck_minus = [
            0.0025-0.0013376935246402686j,
            -0.00026023645770006167j,
            -0.0002748355116913478j,
        ]
        vk_minus = [
            1-1j,
            0.08121642500626945-1j,
            0.24364927501880834-1j,
        ]

        bath = LorentzianBath(
            Q=Q, gamma=0.01, w=1, mu=-1, T=0.025851991, Nk=2, tag="bath1",
        )
        assert len(bath.exponents) == 6
        for i in range(len(ck_plus)):
            exp_p = bath.exponents[2 * i]
            exp_m = bath.exponents[2 * i + 1]
            check_exponent(
                exp_p, "+", dim=2, Q=Q,
                ck=ck_plus[i], vk=vk_plus[i],
                sigma_bar_k_offset=1,
                tag="bath1",
            )
            check_exponent(
                exp_m, "-", dim=2, Q=Q,
                ck=ck_minus[i], vk=vk_minus[i],
                sigma_bar_k_offset=-1,
                tag="bath1",
            )


class TestLorentzianPadeBath:
    def test_create(self):
        Q = sigmaz()
        ck_plus = [
            0.0025+0.0014269000277373958j,
            -0.0002608459250597002j,
            -0.0011660541026776957j,
        ]
        vk_plus = [
            1+1j,
            0.08123902308117874+1j,
            0.3371925267385835+1j,
        ]
        ck_minus = [
            0.0025+0.0014269000277373958j,
            -0.0002608459250597002j,
            -0.0011660541026776957j,
        ]
        vk_minus = [
            1-1j,
            0.08123902308117874-1j,
            0.3371925267385835-1j,
        ]

        bath = LorentzianPadeBath(
            Q=Q, gamma=0.01, w=1, mu=-1, T=0.025851991, Nk=2, tag="bath1",
        )
        assert len(bath.exponents) == 6
        for i in range(len(ck_plus)):
            exp_p = bath.exponents[2 * i]
            exp_m = bath.exponents[2 * i + 1]
            check_exponent(
                exp_p, "+", dim=2, Q=Q,
                ck=ck_plus[i], vk=vk_plus[i],
                sigma_bar_k_offset=1,
                tag="bath1",
            )
            check_exponent(
                exp_m, "-", dim=2, Q=Q,
                ck=ck_minus[i], vk=vk_minus[i],
                sigma_bar_k_offset=-1,
                tag="bath1",
            )


class TestFitUtils:
    cks = [1] * 5
    bath = BosonicBath(sigmax(), cks, cks, cks, cks)

    def spectral_density(self, w, a, b, c):
        tot = 0
        for i in range(len(a)):
            tot += (
                2
                * a[i]
                * b[i]
                * w
                / (((w + c[i]) ** 2 + b[i] ** 2)
                   * ((w - c[i]) ** 2 + b[i] ** 2))
            )
        return tot

    def test_pack(self):
        n = np.random.randint(100)
        before = np.random.rand(3, n)
        a, b, c = before
        assert len(self.bath.pack(a, b, c)) == n * 3
        assert (self.bath.pack(a, b, c) == before.flatten()).all()

    def test_unpack(self):
        n = np.random.randint(100)
        before = np.random.rand(3, n)
        a, b, c = before
        assert (self.bath.unpack(self.bath.pack(a, b, c)) == before).all()

    def test_rmse(self):
        lam = 0.05
        gamma = 4
        w0 = 2
        def func(x, lam, gamma, w0): return np.exp(-lam * x) + gamma / w0
        x = np.linspace(1, 100, 10)
        y = func(x, lam, gamma * 4, w0 * 4)
        assert np.isclose(self.bath._rmse(func, x, y, lam, gamma, w0), 0)

    def test_leastsq(self):
        t = np.linspace(0.1, 10 * 5, 1000)
        a, b, c = [list(range(2))] * 3
        N = 2
        C = self.spectral_density(t, a, b, c)
        sigma = 1e-4
        C_max = abs(max(C, key=abs))
        wc = t[np.argmax(C)]
        guesses = self.bath.pack([C_max] * N, [wc] * N, [wc] * N)
        lower = self.bath.pack(
            [-100 * C_max] * N, [0.1 * wc] * N, [0.1 * wc] * N)
        upper = self.bath.pack(
            [100 * C_max] * N, [100 * wc] * N, [100 * wc] * N)
        a2, b2, c2 = self.bath._leastsq(
            self.spectral_density,
            C,
            t,
            guesses=guesses,
            lower=lower,
            upper=upper,
            sigma=sigma,
        )
        C2 = self.spectral_density(t, a2, b2, c2)
        assert np.isclose(C, C2).all()

    def test_fit(self):
        a, b, c = [list(range(2))] * 3
        w = np.linspace(0.1, 10 * 5, 1000)
        J = self.spectral_density(w, a, b, c)
        rmse, [a2, b2, c2] = self.bath._fit(self.spectral_density, J, w, N=2)
        J2 = self.spectral_density(w, a2, b2, c2)
        assert np.isclose(J, J2).all()
        assert rmse < 1e-15


class TestFitSpectral:
    alpha = 0.005
    wc = 1
    T = 1
    bath = FitSpectral(T, sigmax(), Nk=2)

    def test_spectral_density_approx(self):
        J = 0.4
        a, b, c = [list(range(2))] * 3
        w = 1
        assert self.bath.spectral_density_approx(w, a, b, c) == J
        a, b, c = [list(range(3))] * 3
        w = 2
        assert self.bath.spectral_density_approx(w, a, b, c) == J

    def test_spec_spectrum_approx(self):
        a, b, c = [list(range(2))] * 3
        w = np.linspace(0.1, 10 * self.wc, 1000)
        J = self.bath.spectral_density_approx(w, a, b, c)
        pow = J * ((1 / (np.e ** (w / self.T) - 1)) + 1) * 2
        rmse, self.bath.params_spec = self.bath._fit(
            self.bath.spectral_density_approx, J, w, N=2, label="spectral"
        )
        pow2 = self.bath.spec_spectrum_approx(w)
        assert np.isclose(pow, pow2).all()

    def test_get_fit(self):
        a, b, c = [list(range(5))] * 3
        w = np.linspace(0.1, 10 * self.wc, 1000)
        J = self.bath.spectral_density_approx(w, a, b, c)
        self.bath.get_fit(J, w)
        a2, b2, c2 = self.bath.params_spec
        J2 = self.bath.spectral_density_approx(w, a2, b2, c2)
        assert np.sum(J - J2) < (1e-12)
        assert self.bath.spec_n <= 5

    def test_matsubara_coefficients_from_spectral_fit(self):
        ckAR, vkAr, ckAI, vkAI = (
            [
                (0.29298628613108785 - 0.2097848947562078j),
                (0.29298628613108785 + 0.2097848947562078j),
                (-0.01608448645347298 + 0j),
                (-0.002015397620259254 + 0j),
            ],
            [(1 - 1j), (1 + 1j), (6.283185307179586 + 0j),
             (12.566370614359172 + 0j)],
            [(-0 + 0.25j), -0.25j],
            [(1 - 1j), (1 + 1j)],
        )
        self.bath.params_spec = np.array([1]), np.array([1]), np.array([1])
        self.bath._matsubara_spectral_fit()
        ckAR2, vkAr2, ckAI2, vkAI2 = self.bath.cvars
        assert np.isclose(ckAR, ckAR2).all()
        assert np.isclose(vkAr, vkAr2).all()
        assert np.isclose(ckAI, ckAI2).all()
        assert np.isclose(vkAI, vkAI2).all()


class TestFitCorr:
    alpha = 0.005
    wc = 1
    T = 1
    bath = FitCorr(sigmax())

    def test_corr_approx(self):
        t = np.array([1 / 10, 1 / 5, 1])
        C = np.array(
            [
                -0.900317 + 0.09033301j,
                -0.80241065 + 0.16265669j,
                -0.19876611 + 0.30955988j,
            ]
        )
        a, b, c = -np.array([list(range(2))] * 3)
        C2 = self.bath.corr_approx(t, a, b, c)
        assert np.isclose(C, C2).all()

    def test_fit_correlation(self):
        a, b, c = -np.array([list(range(2))] * 3)
        t = np.linspace(0.1, 10 / self.wc, 1000)
        C = self.bath.corr_approx(t, a, b, c)
        self.bath.fit_correlation(t, C, Nr=3, Ni=3)
        a2, b2, c2 = self.bath.params_real
        ai2, bi2, ci2 = self.bath.params_imag
        C2re = self.bath.corr_approx(t, a2, b2, c2)
        C2imag = self.bath.corr_approx(t, ai2, bi2, ci2)
        assert np.isclose(np.real(C), np.real(C2re)).all()
        assert np.isclose(np.imag(C), np.imag(C2imag)).all()

    def test_matsubara_coefficients(self):
        ans = np.array(
            [
                [(0.5 + 0j), (0.5 + 0j), (0.5 - 0j), (0.5 - 0j)],
                [(-1 - 1j), (-1 - 1j), (-1 + 1j), (-1 + 1j)],
                [-0.5j, -0.5j, 0.5j, 0.5j],
                [(-1 - 1j), (-1 - 1j), (-1 + 1j), (-1 + 1j)],
            ]
        )
        self.bath.params_real = [[1] * 2] * 3
        self.bath.params_imag = [[1] * 2] * 3
        self.bath._matsubara_coefficients()
        assert np.isclose(np.array(self.bath.matsubara_coeff), ans).all()

    def test_corr_spectrum_approx(self):
        self.bath.params_real = [[1] * 2] * 3
        self.bath.params_imag = [[1] * 2] * 3
        self.bath._matsubara_coefficients()
        S = self.bath.corr_spectrum_approx(np.array([1]))
        assert np.isclose(S, np.array([-0.8 + 0j]))


class TestOhmicBath:
    alpha = 0.5
    s = 1
    wc = 1
    T = 1
    Nk = 4
    bath = OhmicBath(T, sigmax(), alpha, wc, s, Nk,
                     method="spectral", rmse=1e-4)

    def test_ohmic_spectral_density(self):
        w = np.linspace(0, 50 * self.wc, 10000)
        J = self.bath.ohmic_spectral_density(w)
        J2 = self.alpha * w * np.exp(-abs(w) / self.wc)
        assert np.isclose(J, J2).all()

    def test_ohmic_correlation(self):
        t = np.linspace(0, 10, 10)
        C = self.bath.ohmic_correlation(t, s=3)
        Ctest = np.array(
            [
                1.11215545e00 + 0.00000000e00j,
                -2.07325102e-01 + 3.99285208e-02j,
                -3.56854914e-02 + 2.68834062e-02j,
                -1.02997412e-02 + 5.98374459e-03j,
                -3.85107084e-03 + 1.71629063e-03j,
                -1.71424113e-03 + 6.14748921e-04j,
                -8.66216773e-04 + 2.59388769e-04j,
                -4.81154330e-04 + 1.23604055e-04j,
                -2.87395509e-04 + 6.46270269e-05j,
                -1.81762994e-04 + 3.63396778e-05j,
            ]
        )
        assert np.isclose(C, Ctest).all()

    def test_ohmic_power_spectrum(self):
        w = np.linspace(0.1, 50 * self.wc, 10)
        self.bath.s = 3
        pow = self.bath.ohmic_power_spectrum(w)
        testpow = np.array(
            [
                9.50833194e-03,
                6.38339038e-01,
                1.93684196e-02,
                2.53252115e-04,
                2.33614419e-06,
                1.77884399e-08,
                1.19944201e-10,
                7.43601157e-13,
                4.33486581e-15,
                2.41093731e-17,
            ]
        )
        assert np.isclose(pow, testpow).all()
