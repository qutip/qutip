"""
Tests for qutip.solver.heom.bofin_fit
"""
import numpy as np
import pytest
from qutip.solver.heom.bofin_fit import (
    pack, unpack, _rmse, _fit, _leastsq, _run_fit, SpectralFitter,
    CorrelationFitter, OhmicBath

)
from qutip.solver.heom.bofin_baths import (
    UnderDampedBath
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


def test_run_fit():
    a, b, c = [list(range(5))] * 3
    w = np.linspace(0.1, 10 * 5, 1000)
    J = spectral(w, a, b, c)
    rmse, [a2, b2, c2] = _run_fit(spectral, J, w, final_rmse=1e-4)
    J2 = spectral(w, a2, b2, c2)
    # assert np.isclose(J, J2).all()
    assert rmse < 1e-4


class TestSpectralFitter:

    def test_spectral_density_approx(self):
        J = 0.4
        a, b, c = [list(range(2))] * 3
        w = 1
        T = 1
        bath = SpectralFitter(T, sigmax(), Nk=2)
        assert bath._spectral_density_approx(w, a, b, c) == J
        a, b, c = [list(range(3))] * 3
        w = 2
        assert bath._spectral_density_approx(w, a, b, c) == J

    def test_get_fit(self):
        T = 1
        wc = 1
        bath = SpectralFitter(T, sigmax(), Nk=2)
        a, b, c = [list(range(5))] * 3
        w = np.linspace(0.1, 10 * wc, 1000)
        J = bath._spectral_density_approx(w, a, b, c)
        bath.get_fit(w, J, N=5)
        a2, b2, c2 = bath.fitInfo['params']
        J2 = bath._spectral_density_approx(w, a2, b2, c2)
        assert np.isclose(J, J2).all()
        assert bath.fitInfo['N'] <= 5

    def test_matsubara_coefficients(self):
        Q = sigmax()
        T = 1
        w = np.linspace(0, 15, 20000)
        ud = UnderDampedBath(Q, lam=0.05, w0=1, gamma=1, T=T, Nk=1)
        fs = SpectralFitter(T, Q, Nk=1)
        _, fitinfo = fs.get_fit(w, ud.spectral_density, N=1)
        fbath = fs._matsubara_coefficients(fitinfo['params'])
        for i in range(len(fbath.exponents)):
            assert np.isclose(fbath.exponents[i].ck, ud.exponents[i].ck)
            if (fbath.exponents[i].ck2 != ud.exponents[i].ck2):
                assert np.isclose(fbath.exponents[i].ck2, ud.exponents[i].ck2)
            assert np.isclose(fbath.exponents[i].vk, ud.exponents[i].vk)


class TestCorrelationFitter:

    def test_corr_approx(self):
        t = np.array([1 / 10, 1 / 5, 1])
        C = np.array(
            [
                -0.900317 + 0.09033301j,
                -0.80241065 + 0.16265669j,
                -0.19876611 + 0.30955988j,
            ]
        )
        bath = CorrelationFitter(sigmax(), 1)
        a, b, c = -np.array([list(range(2))] * 3)
        C2 = bath._corr_approx(t, a, b, c)
        assert np.isclose(C, C2).all()

    def test_get_fit(self):
        bath = CorrelationFitter(sigmax(), 1)
        a, b, c = [[1, 1, 1], [-1, -1, -1], [1, 1, 1]]
        t = np.linspace(0, 10, 1000)
        C = bath._corr_approx(t, a, b, c)
        bbath, fitInfo = bath.get_fit(t, C, Nr=3, Ni=3)
        a2, b2, c2 = fitInfo['params_real']
        a3, b3, c3 = fitInfo['params_imag']
        C2 = np.real(bath._corr_approx(t, a2, b2, c2))
        C3 = np.imag(bath._corr_approx(t, a3, b3, c3))
        assert np.isclose(np.real(C), C2).all()
        assert np.isclose(np.imag(C), C3).all()

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_matsubara_coefficients(self):
        Q = sigmax()
        T = 1
        t = np.linspace(0, 30, 200)
        ud = UnderDampedBath(Q, lam=0.05, w0=1, gamma=1, T=T, Nk=1)
        fc = CorrelationFitter(Q, T)
        _, fitInfo = fc.get_fit(t, ud.correlation_function, final_rmse=1e-5)
        fbath = fc._matsubara_coefficients(
            fitInfo['params_real'],
            fitInfo['params_imag'])
        fittedbath = fbath.correlation_function_approx(t)
        udc = ud.correlation_function(t)
        assert np.isclose(
            np.real(udc),
            np.real(fittedbath),
            atol=1e-4).all()
        assert np.isclose(
            np.imag(udc),
            np.imag(fittedbath),
            atol=1e-4).all()  # one order below final_rmse


class TestOhmicBath:
    def test_ohmic_spectral_density(self):
        mp = pytest.importorskip("mpmath")
        alpha = 0.5
        wc = 1
        T = 1
        Q = sigmax()
        w = np.linspace(0, 50 * wc, 10000)
        bath = OhmicBath(s=1, alpha=alpha, Q=Q, T=T, wc=wc)
        J = bath.spectral_density(w)
        J2 = bath.alpha * w * np.exp(-abs(w) / wc)
        assert np.isclose(J, J2).all()

    def test_ohmic_correlation(self):
        mp = pytest.importorskip("mpmath")
        alpha = 0.5
        wc = 1
        T = 1
        Q = sigmax()
        t = np.linspace(0, 10, 10)
        bath = OhmicBath(s=3, alpha=alpha, Q=Q, T=T, wc=wc)
        C = bath.correlation(t)
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
        mp = pytest.importorskip("mpmath")
        alpha = 0.5
        wc = 1
        T = 1
        Q = sigmax()
        w = np.linspace(0.1, 50 * wc, 10)
        bath = OhmicBath(s=3, alpha=alpha, wc=wc, Q=Q, T=T)
        batho, fitinfo = bath.get_fit(w, N=(1,2), method='spectral')
        pow = batho.power_spectrum(w)
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
