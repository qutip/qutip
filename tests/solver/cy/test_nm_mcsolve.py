import numbers
import pickle

import numpy as np
import pytest

from qutip import coefficient
from qutip.core.coefficient import ConstantCoefficient
from qutip.solver.cy.nm_mcsolve import (
    RateShiftCoefficient, SqrtRealCoefficient,
)


def assert_functions_equal(f, g, tlist, rtol=1e-12, atol=1e-12):
    """ Assert that to functions of t are equal at a list of specified times.
    """
    assert len(tlist) > 0
    np.testing.assert_allclose(
        [f(t) for t in tlist],
        [g(t) for t in tlist],
        rtol=rtol, atol=atol,
    )


class RateSet:
    """ A list of coefficients and a tlist of times to test at. """
    def __init__(self, coeffs, tlist):
        self.coeffs = coeffs
        self.tlist = tlist


def rate_set(coeffs, *, tlist=np.linspace(0, 1, 20), args=None, **kw):
    id = kw.pop("id")
    args = args or {}
    coeffs = [
        ConstantCoefficient(c) if isinstance(c, numbers.Number)
        else coefficient(c, args=args)
        for c in coeffs
    ]
    return pytest.param(RateSet(coeffs, tlist), id=id)


@pytest.fixture(params=[
    rate_set([], id="no_rates"),
    rate_set([0], id="single_zero_rate"),
    rate_set([1], id="single_positive_rate"),
    rate_set([-1], id="single_negative_rate"),
    rate_set([0, 0, 0], id="multiple_zero_rates"),
    rate_set([0, 1], id="zero_and_positive_rate"),
    rate_set([0, -1], id="zero_and_negative_rate"),
    rate_set([1, -1], id="positive_and_negative_rate"),
    rate_set(
        [lambda t: np.sin(t)],
        tlist=np.linspace(0, 2 * np.pi, 20),
        id="sin_rate",
    ),
    rate_set(
        [lambda t: np.sin(t), -0.5],
        tlist=np.linspace(0, 2 * np.pi, 20),
        id="sin_and_negative_rate",
    )
])
def rates(request):
    return request.param


def sin_t(t):
    """ Pickle-able and coefficient-able sin(t). """
    return np.sin(t)


class TestRateShiftCoefficient:

    @staticmethod
    def assert_f_equals_rate_shift(f, coeffs, tlist, **kw):
        def g(t):
            return 2 * np.abs(min(
                [0] + [np.real(c(t)) for c in coeffs]
            ))
        assert_functions_equal(f, g, tlist, **kw)

    def test_call(self, rates):
        rs = RateShiftCoefficient(rates.coeffs)
        self.assert_f_equals_rate_shift(rs, rates.coeffs, rates.tlist)

    def test_as_double(self, rates):
        rs = RateShiftCoefficient(rates.coeffs)
        self.assert_f_equals_rate_shift(
            rs.as_double, rates.coeffs, rates.tlist,
        )
        assert all(isinstance(rs.as_double(t), float) for t in rates.tlist)

    def test_copy(self, rates):
        rs = RateShiftCoefficient(rates.coeffs)
        rs = rs.copy()
        self.assert_f_equals_rate_shift(rs, rates.coeffs, rates.tlist)

    def test_replace_arguments(self):
        coeff = coefficient(lambda t, w: np.sin(w * t), args={"w": 1.0})
        tlist = np.linspace(0, 2 * np.pi, 100)
        rs = RateShiftCoefficient([coeff])

        for w in [0, 1, 2, 3]:
            rs2 = rs.replace_arguments(w=w)
            self.assert_f_equals_rate_shift(
                rs2, [coeff.replace_arguments(w=w)], tlist,
            )

    def test_reduce(self):
        coeff = coefficient(sin_t)
        tlist = np.linspace(0, 2 * np.pi, 20)
        rs = RateShiftCoefficient([coeff])

        data = pickle.dumps(rs, protocol=-1)
        rs = pickle.loads(data)
        self.assert_f_equals_rate_shift(rs, [coeff], tlist)


class TestSqrtRealCoefficient:

    @staticmethod
    def assert_f_equals_sqrt_real(f, coeff, tlist, **kw):
        def g(t):
            return np.sqrt(np.real(coeff(t)))
        assert_functions_equal(f, g, tlist, **kw)

    def test_call(self):
        coeff = coefficient(lambda t: np.abs(np.sin(t)))
        tlist = np.linspace(0, 2 * np.pi, 20)
        sr = SqrtRealCoefficient(coeff)
        self.assert_f_equals_sqrt_real(sr, coeff, tlist)

    def test_copy(self):
        coeff = coefficient(lambda t: np.abs(np.sin(t)))
        tlist = np.linspace(0, 2 * np.pi, 20)
        sr = SqrtRealCoefficient(coeff)
        sr = sr.copy()
        self.assert_f_equals_sqrt_real(sr, coeff, tlist)

    def test_replace_arguments(self):
        coeff = coefficient(
            lambda t, w: np.abs(np.sin(w * t)),
            args={"w": 1.0},
        )
        tlist = np.linspace(0, 2 * np.pi, 100)
        sr = SqrtRealCoefficient(coeff)

        for w in [0, 1, 2, 3]:
            sr2 = sr.replace_arguments(w=w)
            self.assert_f_equals_sqrt_real(
                sr2, coeff.replace_arguments(w=w), tlist,
            )

    def test_reduce(self):
        coeff = coefficient(sin_t)
        tlist = np.linspace(0, np.pi, 10)
        sr = SqrtRealCoefficient(coeff)

        data = pickle.dumps(sr, protocol=-1)
        sr = pickle.loads(data)
        self.assert_f_equals_sqrt_real(sr, coeff, tlist)
