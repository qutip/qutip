

import numpy as np
import pytest
from qutip.utilities import (n_thermal, iterated_fit)
from qutip.core.environment import (
    BosonicEnvironment
)
from scipy.integrate import quad_vec

# Tests are slow if precise, so tolerance is slow for speed
# tolerance should be one order of magnitude below the epsabs,epsrel in
# correlation
inttol = 1e-3
comtol = 1e-2


@pytest.fixture()
def params():
    N = 3
    np.random.seed(42)
    lam = np.random.uniform(low=0.05, high=0.5, size=(N,))
    wc = np.random.uniform(low=2, high=5, size=(N,))
    T = np.random.uniform(low=1, high=5, size=(N,))
    t = np.linspace(0, 25, 100)
    w = np.linspace(0, 15, 100)
    w2 = np.linspace(-20, 20, 100)
    corr = [correlation(t, wc[k], lam[k], T[k])
            for k in range(len(lam))]
    return lam, wc, T, t, w, w2, corr


def spectral_density(w, gamma, w0, lam):
    return lam**2 * gamma * w / ((w**2 - w0**2)**2
                                 + (gamma*w)**2)


def power(w, gamma, w0, lam, T):
    zero_mask = (w == 0)
    nonzero_mask = np.invert(zero_mask)

    S = np.zeros_like(w)
    S[zero_mask] = 2 * T * spectral_density(1e-16, gamma, w0, lam) / 1e-16
    S[nonzero_mask] = 2*np.sign(w[nonzero_mask])*spectral_density(
        np.abs(w[nonzero_mask]), gamma, w0, lam)*(
        n_thermal(w[nonzero_mask], T)+1)
    return S


def correlation(t, gamma, w0, lam, T):
    def integrand(w, t):
        return spectral_density(w, gamma, w0, lam) / np.pi * (
            (2 * n_thermal(w, T) + 1) * np.cos(w * t)
            - 1j * np.sin(w * t)
        )

    result = quad_vec(lambda w: integrand(w, t), 0,
                      np.inf, epsabs=1e-3, epsrel=1e-3)
    return result[0]


@pytest.fixture()
def params():
    N = 3
    np.random.seed(42)
    lam = np.random.uniform(low=0.1, high=0.9, size=(N,))
    gamma = np.random.uniform(low=1, high=2, size=(N,))
    w0 = np.random.uniform(low=1.1, high=5, size=(N,))
    T = np.random.uniform(low=1.5, high=5, size=(N,))
    t = np.linspace(0, 10, 500)
    w = np.linspace(0, 25, 500)
    w2 = np.linspace(-50, 50, 500)
    corr = [correlation(t, gamma[k], w0[k], lam[k], T[k])
            for k in range(len(lam))]
    return lam, gamma, w0, T, t, w, w2, corr


class TestBosonicEnvironment:
    # Tests are slow if precise, so tolerance is slow for speed
    # tolerance should be one order of magnitude below the epsabs,epsrel in
    # correlation
    def test_from_correlation(self, params):
        lam, gamma, w0, T, t, w, w2, corr = params
        for k in range(len(lam)):
            bb5 = BosonicEnvironment.from_correlation_function(
                corr[k], t, T=T[k])
            corr_approx = bb5.correlation_function(t)
            assert np.isclose(corr_approx, corr[k]).all()
            assert np.isclose(bb5.power_spectrum(w2), power(
                w2, gamma[k], w0[k], lam[k], T[k]), atol=1e-2).all()
            assert np.isclose(bb5.spectral_density(w), spectral_density(
                w, gamma[k], w0[k], lam[k]), atol=1e-2).all()

    def test_from_spectral_density(self, params):
        lam, gamma, w0, T, t, w, w2, _ = params
        for k in range(len(lam)):
            sd = spectral_density(w, gamma[k], w0[k], lam[k])
            bb5 = BosonicEnvironment.from_spectral_density(
                sd, w, wMax=10*gamma[k], T=T[k])
            assert np.isclose(bb5.correlation_function(t), correlation(
                t, gamma[k], w0[k], lam[k], T[k]), atol=1e-2).all()
            assert np.isclose(bb5.power_spectrum(w2), power(
                w2, gamma[k], w0[k], lam[k], T[k]), atol=1e-2).all()
            assert np.isclose(bb5.spectral_density(
                w), spectral_density(w, gamma[k], w0[k], lam[k])).all()

    def test_from_power_spectrum(self, params):
        lam, gamma, w0, T, t, w, w2, corr = params
        for k in range(len(lam)):
            pow = power(w2, gamma[k], w0[k], lam[k], T[k])
            bb5 = BosonicEnvironment.from_power_spectrum(
                pow, w2, wMax=gamma[k], T=T[k])
            assert np.isclose(bb5.correlation_function(t),
                              corr[k], atol=comtol).all()
            assert np.isclose(bb5.power_spectrum(w2), power(
                w2, gamma[k], w0[k], lam[k], T[k])).all()
            assert np.isclose(bb5.spectral_density(w), spectral_density(
                w, gamma[k], w0[k], lam[k]), atol=comtol).all()

    def test_approx_by_cf_fit(self, params):
        lam, gamma, w0, T, t, w, w2, corr = params
        for k in range(len(lam)):
            bb5 = BosonicEnvironment.from_correlation_function(
                corr[k], t, T=T[k])
            bb6, finfo = bb5.approx_by_cf_fit(
                t, target_rsme=None, Nr_max=2, Ni_max=1,sigma=1e-2)
            assert np.isclose(bb6.correlation_function(t),
                              corr[k], atol=5*comtol).all()
            assert np.isclose(bb6.power_spectrum(w2), power(
                w2, gamma[k], w0[k], lam[k], T[k]), atol=5*comtol).all()
            assert np.isclose(bb6.spectral_density(w), spectral_density(
                w, gamma[k], w0[k], lam[k]), atol=5*comtol).all()

    def test_approx_by_sd_fit(self, params):
        lam, gamma, w0, T, t, w, w2, corr = params
        for k in range(len(lam)):
            sd = spectral_density(t, gamma[k], w0[k], lam[k])
            bb5 = BosonicEnvironment.from_spectral_density(sd, t, T=T[k])
            bb6, finfo = bb5.approx_by_sd_fit(w, Nmax=1, Nk=10,sigma=1)
            # asking for more precision that the Bosonic enviroment has may
            # violate this (due to the interpolation and it's easily fixed
            # using a denser range). I could have set N=1 but thought this was
            # a sensible test for N
            assert finfo["N"] == 1
            assert np.isclose(bb6.correlation_function(t),
                              corr[k],
                              atol=comtol).all()
            assert np.isclose(bb6.power_spectrum(w2), power(
                w2, gamma[k], w0[k], lam[k], T[k]), atol=comtol).all()
            assert np.isclose(bb6.spectral_density(w), spectral_density(
                w, gamma[k], w0[k], lam[k]), atol=comtol).all()


class TestFits:

    def model1(self, t, a, b, c):
        return np.real(a * np.exp(-(b + 1j * c) * t))

    def model2(self, t, a, b, c):
        return a*(t-c)+b

    def test_same_model(self):
        t = np.linspace(0, 10, 100)
        fparams = np.random.uniform(low=0.05, high=5, size=(3,))
        _, params = iterated_fit(self.model1, 3, t, self.model1(t, *fparams))
        real_fit_result = 0
        for x in params:
            real_fit_result += self.model1(t, *x)
        assert np.isclose(real_fit_result, self.model1(t, *fparams)).all()
