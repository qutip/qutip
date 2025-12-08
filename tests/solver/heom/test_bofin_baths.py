"""
Tests for qutip.solver.heom.bofin_baths.
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
                " exponents"
            )

        for exp_type, kw in [("R", {}), ("I", {}), ("RI", {"ck2": 3.0})]:
            with pytest.raises(ValueError) as err:
                BathExponent(
                    exp_type, None, Q=None, ck=1.0, vk=2.0,
                    sigma_bar_k_offset=1, **kw,
                )
            assert str(err.value) == (
                "Offset of sigma bar (sigma_bar_k_offset) should only be"
                " specified for + and - type exponents"
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
            "The exponent lists ck_real and vk_real, and ck_imag and"
            " vk_imag must be the same length."
        )

        with pytest.raises(ValueError) as err:
            BosonicBath(Q, [1.], [0.5], [2.], [])
        assert str(err.value) == (
            "The exponent lists ck_real and vk_real, and ck_imag and"
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
        print(delta)
        print(terminator)
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
            "The exponent lists ck_plus and vk_plus, and ck_minus and"
            " vk_minus must be the same length."
        )

        with pytest.raises(ValueError) as err:
            FermionicBath(Q, [1.], [0.5], [2.], [])
        assert str(err.value) == (
            "The exponent lists ck_plus and vk_plus, and ck_minus and"
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
