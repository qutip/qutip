"""
Tests for qutip.solver.heom.bofin_fit
"""
import numpy as np
import sys
import pytest
from qutip.solver.heom.bofin_fit import (
    pack, unpack, _rmse, _fit, _leastsq, SpectralFitter,
    CorrelationFitter, OhmicBath

)
from qutip import sigmax


def spectral(w, a, b, c):
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


def test_pack():
    n = np.random.randint(100)
    before = np.random.rand(3, n)
    a, b, c = before
    assert len(pack(a, b, c)) == n * 3
    assert (pack(a, b, c) == before.flatten()).all()


def test_unpack():
    n = np.random.randint(100)
    before = np.random.rand(3, n)
    a, b, c = before
    assert (unpack(pack(a, b, c)) == before).all()


def test_rmse():
    lam = 0.05
    gamma = 4
    w0 = 2

    def func(x, lam, gamma, w0):
        return np.exp(-lam * x) + gamma / w0
    x = np.linspace(1, 100, 10)
    y = func(x, lam, gamma + 1e-8, w0)
    assert np.isclose(_rmse(func, x, y, lam, gamma, w0), 0)


def test_leastsq():
    t = np.linspace(0.1, 10 * 5, 1000)
    N = 2
    a, b, c = [list(range(2))]*3
    C = spectral(t, a, b, c)
    sigma = 1e-4
    C_max = abs(max(C, key=abs))
    wc = t[np.argmax(C)]
    guesses = pack([C_max] * N, [wc] * N, [wc] * N)
    lower = pack(
        [-100 * C_max] * N, [0.1 * wc] * N, [0.1 * wc] * N)
    upper = pack(
        [100 * C_max] * N, [100 * wc] * N, [100 * wc] * N)
    a2, b2, c2 = _leastsq(
        spectral,
        C,
        t,
        guesses=guesses,
        lower=lower,
        upper=upper,
        sigma=sigma,
    )
    C2 = spectral(t, a2, b2, c2)
    assert np.isclose(C, C2).all()


def test_fit():
    a, b, c = [list(range(2))] * 3
    w = np.linspace(0.1, 10 * 5, 1000)
    J = spectral(w, a, b, c)
    rmse, [a2, b2, c2] = _fit(spectral, J, w, N=2)
    J2 = spectral(w, a2, b2, c2)
    assert np.isclose(J, J2).all()
    assert rmse < 1e-15


@pytest.fixture
def bath_fixture():
    wc = 1
    T = 1
    bath = SpectralFitter(T, sigmax(), Nk=2)
    return bath


class TestSpectralFitter:
    def test_spectral_density_approx(bath_fixture):
        J = 0.4
        a, b, c = [list(range(2))] * 3
        w = 1
        bath = bath_fixture
        assert bath._spectral_density_approx(w, a, b, c) == J
        a, b, c = [list(range(3))] * 3
        w = 2
        assert bath._spectral_density_approx(w, a, b, c) == J

    def test_spec_spectrum_approx(bath_fixture):
        bath = bath_fixture
        wc = 1
        a, b, c = [list(range(2))] * 3
        w = np.linspace(0.1, 10 * wc, 1000)
        J = bath._spectral_density_approx(w, a, b, c)
        pow = J * ((1 / (np.e ** (w / bath.T) - 1)) + 1) * 2
        rmse, params = _fit(
            bath._spectral_density_approx, J, w, N=2, label="spectral"
        )
        a2, b2, c2 = params
        pow2 = bath._spectral_density_approx(
            w, a2, b2, c2) * ((1 / (np.e ** (w / bath.T) - 1)) + 1) * 2
        assert np.isclose(pow, pow2).all()

    def test_get_fit(bath_fixture):
        bath, wc = bath_fixture
        a, b, c = [list(range(5))] * 3
        w = np.linspace(0.1, 10 * wc, 1000)
        J = bath._spectral_density_approx(w, a, b, c)
        bath.get_fit(J, w)
        a2, b2, c2 = bath.fitInfo['params']
        J2 = bath._spectral_density_approx(w, a2, b2, c2)
        assert np.sum(J - J2) < (1e-12)
        assert bath.fitInfo['N'] <= 5

    # def test_matsubara_coefficients_from_spectral_fit(self):
    #     ckAR, vkAr, ckAI, vkAI = (
    #         [
    #             (0.29298628613108785 - 0.2097848947562078j),
    #             (0.29298628613108785 + 0.2097848947562078j),
    #             (-0.01608448645347298 + 0j),
    #             (-0.002015397620259254 + 0j),
    #         ],
    #         [(1 - 1j), (1 + 1j), (6.283185307179586 + 0j),
    #          (12.566370614359172 + 0j)],
    #         [(-0 + 0.25j), -0.25j],
    #         [(1 - 1j), (1 + 1j)],
    #     )
    #     self.bath.params_spec = np.array([1]), np.array([1]), np.array([1])
    #     self.bath._matsubara_spectral_fit()
    #     ckAR2, vkAr2, ckAI2, vkAI2 = self.bath.cvars
    #     assert np.isclose(ckAR, ckAR2).all()
    #     assert np.isclose(vkAr, vkAr2).all()
    #     assert np.isclose(ckAI, ckAI2).all()
    #     assert np.isclose(vkAI, vkAI2).all()


class CorrelationFitter:
    alpha = 0.005
    wc = 1
    T = 1
    bath = CorrelationFitter(sigmax())

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
        C2 = self.bath._corr_approx(t, a, b, c)
        assert np.isclose(C, C2).all()

    def test_fit_correlation(self):
        a, b, c = -np.array([list(range(2))] * 3)
        t = np.linspace(0.1, 10 / self.wc, 1000)
        C = self.bath._corr_approx(t, a, b, c)
        self.bath.fit_correlation(t, C, Nr=3, Ni=3)
        a2, b2, c2 = self.bath.params_real
        ai2, bi2, ci2 = self.bath.params_imag
        C2re = self.bath._corr_approx(t, a2, b2, c2)
        C2imag = self.bath._corr_approx(t, ai2, bi2, ci2)
        assert np.isclose(np.real(C), np.real(C2re)).all()
        assert np.isclose(np.imag(C), np.imag(C2imag)).all()

    # def test_matsubara_coefficients(self):
    #     ans = np.array(
    #         [
    #             [(0.5 + 0j), (0.5 + 0j), (0.5 - 0j), (0.5 - 0j)],
    #             [(-1 - 1j), (-1 - 1j), (-1 + 1j), (-1 + 1j)],
    #             [-0.5j, -0.5j, 0.5j, 0.5j],
    #             [(-1 - 1j), (-1 - 1j), (-1 + 1j), (-1 + 1j)],
    #         ]
    #     )
    #     self.bath.params_real = [[1] * 2] * 3
    #     self.bath.params_imag = [[1] * 2] * 3
    #     self.bath._matsubara_coefficients()
    #     assert np.isclose(np.array(self.bath.cvars), ans).all()

    def test_power_spectrum(self):
        self.bath.params_real = [[1] * 2] * 3
        self.bath.params_imag = [[1] * 2] * 3
        self.bath._matsubara_coefficients()
        S = self.bath.power_spectrum(np.array([1]))
        assert np.isclose(S, np.array([-2.4 + 0j]))


@pytest.mark.skipif('mpmath' not in sys.modules,
                    reason="requires the mpmath library")
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
        J = self.bath.spectral_density(w)
        J2 = self.alpha * w * np.exp(-abs(w) / self.wc)
        assert np.isclose(J, J2).all()

    def test_ohmic_correlation(self):
        t = np.linspace(0, 10, 10)
        try:
            C = self.bath.correlation(t, s=3)
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
        except NameError:
            pass

    def test_ohmic_power_spectrum(self):
        w = np.linspace(0.1, 50 * self.wc, 10)
        self.bath.s = 3
        pow = self.bath.power_spectrum(w)
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
