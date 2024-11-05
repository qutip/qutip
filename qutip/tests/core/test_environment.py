import pytest

from numbers import Number

import numpy as np
from qutip.utilities import n_thermal
from scipy.integrate import quad_vec

from qutip.core.environment import (
    BosonicEnvironment
)


def assert_environment_guarantees(
        env, skip_sd=False, skip_cf=False, skip_ps=False):
    """
    Checks the argument types accepted by the SD, CF and PS of the provided
    environment, and that these functions satisfy certain symmetries
    """
    functions = []
    if not skip_sd:
        functions.append(env.spectral_density)
    if not skip_cf:
        functions.append(env.correlation_function)
    if not skip_ps:
        functions.append(env.power_spectrum)

    # SD, CF and PS can be called with number and array
    # Result is required to have same type, ndim and len as argument
    for fun in functions:
        assert isinstance(fun(1), Number)
        assert isinstance(fun(0.), Number)
        assert isinstance(fun(-1.), Number)

        res = fun(np.array([]))
        assert isinstance(res, np.ndarray)
        assert res.ndim == 1
        assert len(res) == 0

        res = fun(np.array([-1, 0, 1]))
        assert isinstance(res, np.ndarray)
        assert res.ndim == 1
        assert len(res) == 3

    if not skip_sd:
        # For negative frequencies, SD must return zero
        res = env.spectral_density(np.linspace(-10, 10, 20))
        np.testing.assert_allclose(res[:10], np.zeros_like(res[:10]))
        # SD must be real
        np.testing.assert_allclose(np.imag(res), np.zeros_like(res))

    if not skip_cf:
        # CF symmetry
        # Intentionally not testing at t=0 because of D-L spectral density
        res = env.correlation_function(np.linspace(-10, 10, 20))
        res_reversed = res[::-1]
        np.testing.assert_allclose(res, np.conjugate(res_reversed))

    if not skip_ps:
        # PS must be real
        res = env.power_spectrum(np.linspace(-10, 10, 20))
        np.testing.assert_allclose(np.imag(res), np.zeros_like(res))

def assert_equivalent(env1, env2, *, tol,
                      skip_sd=False, skip_cf=False, skip_ps=False,
                      tMax=25, wMax=10):
    """
    Checks that two environments have the same SD, CF and PS
    (up to given tolerance)
    """
    tlist = np.linspace(0, tMax, 100)
    wlist = np.linspace(0, wMax, 100)
    wlist2 = np.linspace(-wMax, wMax, 100)

    if not skip_sd:
        assert_allclose(env1.spectral_density(wlist),
                        env2.spectral_density(wlist), tol)
    if not skip_cf:
        assert_allclose(env1.correlation_function(tlist),
                        env2.correlation_function(tlist), tol)
    if not skip_ps:
        assert_allclose(env1.power_spectrum(wlist2),
                        env2.power_spectrum(wlist2), tol)

def assert_allclose(actual, desired, tol):
    # We want to compare to arrays and provide both an abs and a rel tolerance
    # We use the same parameter for both for simplicity
    # However we reduce the abs tolerance if the numerical values are all small
    atol = tol * min(np.max(np.abs(desired)), 1)
    # If desired diverges somewhere, we allow actual to be anything
    desired_finite = np.isfinite(desired)
    np.testing.assert_allclose(actual[desired_finite], desired[desired_finite],
                               rtol=tol, atol=atol)


class UDReference:
    def __init__(self, T, lam, gamma, w0):
        self.T = T
        self.lam = lam
        self.gamma = gamma
        self.w0 = w0

    def spectral_density(self, w):
        # only valid for w >= 0
        return self.lam**2 * self.gamma * w / (
            (w**2 - self.w0**2)**2 + (self.gamma * w)**2
        )

    def power_spectrum(self, w, eps=1e-16):
        # at zero frequency, take limit w->0
        w = np.array(w)
        w[w == 0] = eps
        return 2 * np.sign(w) * self.spectral_density(np.abs(w)) * (
            n_thermal(w, self.T) + 1
        )

    def correlation_function(self, t, Nk=1000):
        # only valid for t >= 0
        if self.T == 0:
            return self._cf_zeroT(t)

        def coth(x):
            return 1 / np.tanh(x)

        Om = np.sqrt(self.w0**2 - (self.gamma / 2)**2)
        Gamma = self.gamma / 2
        beta = 1 / self.T

        ckAR = [
            (self.lam**2 / (4*Om)) * coth(beta * (Om + 1.0j * Gamma) / 2),
            (self.lam**2 / (4*Om)) * coth(beta * (Om - 1.0j * Gamma) / 2),
        ]
        ckAR.extend(
            (-2 * self.lam**2 * self.gamma / beta) * (2 * np.pi * k / beta) /
            (((Om + 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2) *
            ((Om - 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2)) + 0.j
            for k in range(1, Nk + 1)
        )
        vkAR = [
            -1.0j * Om + Gamma,
            1.0j * Om + Gamma,
        ]
        vkAR.extend(
            2 * np.pi * k * self.T + 0.j
            for k in range(1, Nk + 1)
        )

        factor = 1j / 4
        ckAI = [-factor * self.lam**2 / Om, factor * self.lam**2 / Om]
        vkAI = [-(-1j * Om - Gamma), -(1j * Om - Gamma)]

        result = np.zeros_like(t, dtype=complex)
        for ck, vk in zip(ckAR, vkAR):
            result += ck * np.exp(-vk * t)
        for ck, vk in zip(ckAI, vkAI):
            result += 1j * ck * np.exp(-vk * t)
        return result

    def _cf_zeroT(self, t, eps=1e-3):
        Om = np.sqrt(self.w0**2 - (self.gamma / 2)**2)
        Gamma = self.gamma / 2

        term1 = self.lam**2 / (2 * Om) * np.exp(-(1j * Om + self.gamma / 2) * t)

        def integrand(x, t):
            return x * np.exp(-x * t) / (
                ((Om + 1j * Gamma)**2 + x**2) * ((Om - 1j * Gamma)**2 + x**2)
            )
        integral = quad_vec(
            lambda w: integrand(w, t), 0, np.inf, epsabs=eps, epsrel=eps
        )
        result = term1 - self.gamma * self.lam**2 / np.pi * integral[0]
        return result

    def _cf_by_integral(self, t, eps=1e-3):
        # alternative implementation of correlation function
        # currently unused because it makes tests slow
        # but kept here in case we might need it in the future
        def integrand(w, t):
            return self.spectral_density(w) / np.pi * (
                (2 * n_thermal(w, self.T) + 1) * np.cos(w * t)
                - 1j * np.sin(w * t)
            )

        result = quad_vec(lambda w: integrand(w, t), 0, np.inf,
                          epsabs=eps, epsrel=eps)
        return result[0]


class DLReference:
    def __init__(self, T, lam, gamma):
        self.T = T
        self.lam = lam
        self.gamma = gamma

    def spectral_density(self, w):
        # only valid for w >= 0
        return 2 * self.lam * self.gamma * w / (w**2 + self.gamma**2)

    def power_spectrum(self, w, eps=1e-16):
        # at zero frequency, take limit w->0
        w = np.array(w)
        w[w == 0] = eps
        return 2 * np.sign(w) * self.spectral_density(np.abs(w)) * (
            n_thermal(w, self.T) + 1
        )

    def correlation_function(self, t, Nk=1000):
        def cot(x):
            return 1 / np.tan(x)

        # C_real expansion terms:
        ck_real = [self.lam * self.gamma / np.tan(self.gamma / (2 * self.T))]
        ck_real.extend([
            (8 * self.lam * self.gamma * self.T * np.pi * k * self.T /
                ((2 * np.pi * k * self.T)**2 - self.gamma**2))
            for k in range(1, Nk + 1)
        ])
        vk_real = [self.gamma]
        vk_real.extend([2 * np.pi * k * self.T for k in range(1, Nk + 1)])

        # C_imag expansion terms (this is the full expansion):
        ck_imag = [self.lam * self.gamma * (-1.0)]
        vk_imag = [self.gamma]

        result = np.zeros_like(t, dtype=complex)
        for ck, vk in zip(ck_real, vk_real):
            result += ck * np.exp(-vk * np.abs(t))
        for ck, vk in zip(ck_imag, vk_imag):
            result += self._sign(t) * 1j * ck * np.exp(-vk * np.abs(t))
        result[t == 0] += np.inf  # real part of CF diverges at t=0
        return result

    def _sign(self, x):
        return np.sign(x) + (x == 0)


class TestBosonicEnvironment:
    @pytest.mark.parametrize(["ref", "info"], [
        pytest.param(UDReference(gamma=.1, lam=.5, w0=1, T=0),
                     {'tMax': 125, 'npoints': 300, 'tol': 5e-3},
                     id="UD zero T"),
        pytest.param(DLReference(gamma=.5, lam=.1, T=.5),
                     {'tMax': 25, 'npoints': 500, 'tol': 1e-2},
                     id="DL finite T"),
    ])
    @pytest.mark.parametrize(["interpolate", "provide_tmax"], [
        [True, False], [False, False], [False, True]
    ])
    @pytest.mark.parametrize("provide_temp", [True, False])
    def test_from_cf(
        self, ref, info, interpolate, provide_tmax, provide_temp
    ):
        tMax = info['tMax']
        npoints = info['npoints']
        tol = info['tol']

        # Collect arguments
        if interpolate:
            tlist = np.linspace(0, tMax, npoints)
            clist = ref.correlation_function(tlist)
            if np.isfinite(clist[0]):
                args1 = {'tlist': tlist, 'C': clist}
            else:
                # CF may diverge at t=0, e.g. Drude-Lorentz
                args1 = {'tlist': tlist[1:], 'C': clist[1:]}
        else:
            args1 = {'C': ref.correlation_function}
        args2 = {'tMax': tMax} if provide_tmax else {}
        args3 = {'T': ref.T} if provide_temp else {}

        env = BosonicEnvironment.from_correlation_function(
            **args1, **args2, **args3,
        )

        # Determine which characteristic functions should be accessible
        skip_ps = False
        skip_sd = False
        if not interpolate and not provide_tmax:
            skip_ps = True
            skip_sd = True
            with np.testing.assert_raises(ValueError):
                env.power_spectrum(0)
            with np.testing.assert_raises(ValueError):
                env.spectral_density(0)
        elif not provide_temp:
            skip_sd = True
            with np.testing.assert_raises(ValueError):
                env.spectral_density(0)

        assert_environment_guarantees(env, skip_sd=skip_sd, skip_ps=skip_ps)
        assert_equivalent(env, ref, skip_sd=skip_sd, skip_ps=skip_ps, tol=tol)

    @pytest.mark.parametrize(["ref", "info"], [
        pytest.param(UDReference(gamma=.1, lam=.5, w0=1, T=0),
                     {'wMax': 5, 'npoints': 200, 'tol': 5e-3},
                     id="UD zero T"),
        pytest.param(DLReference(gamma=.5, lam=.1, T=.5),
                     {'wMax': 200, 'npoints': 1500, 'tol': 5e-3},
                     id="DL finite T"),
    ])
    @pytest.mark.parametrize(["interpolate", "provide_wmax"], [
        [True, False], [False, False], [False, True]
    ])
    @pytest.mark.parametrize("provide_temp", [True, False])
    def test_from_sd(
        self, ref, info, interpolate, provide_wmax, provide_temp
    ):
        wMax = info['wMax']
        npoints = info['npoints']
        tol = info['tol']

        # Collect arguments
        if interpolate:
            wlist = np.linspace(0, wMax, npoints)
            jlist = ref.spectral_density(wlist)
            args1 = {'wlist': wlist, 'J': jlist}
        else:
            args1 = {'J': ref.spectral_density}
        args2 = {'wMax': wMax} if provide_wmax else {}
        args3 = {'T': ref.T} if provide_temp else {}

        env = BosonicEnvironment.from_spectral_density(
            **args1, **args2, **args3,
        )

        # Determine which characteristic functions should be accessible
        skip_ps = False
        skip_cf = False
        if not provide_temp:
            skip_ps = True
            skip_cf = True
            with np.testing.assert_raises(ValueError):
                env.power_spectrum(0)
            with np.testing.assert_raises(ValueError):
                env.correlation_function(0)
        elif not interpolate and not provide_wmax:
            skip_cf = True
            with np.testing.assert_raises(ValueError):
                env.correlation_function(0)

        assert_environment_guarantees(env, skip_cf=skip_cf, skip_ps=skip_ps)
        assert_equivalent(env, ref, skip_cf=skip_cf, skip_ps=skip_ps, tol=tol)

    @pytest.mark.parametrize(["ref", "info"], [
        pytest.param(UDReference(gamma=.1, lam=.5, w0=1, T=0),
                     {'wMax': 5, 'npoints': 200, 'tol': 5e-3},
                     id="UD zero T"),
        pytest.param(DLReference(gamma=.5, lam=.1, T=.5),
                     {'wMax': 200, 'npoints': 1500, 'tol': 5e-3},
                     id="DL finite T"),
    ])
    @pytest.mark.parametrize(["interpolate", "provide_wmax"], [
        [True, False], [False, False], [False, True]
    ])
    @pytest.mark.parametrize("provide_temp", [True, False])
    def test_from_ps(
        self, ref, info, interpolate, provide_wmax, provide_temp
    ):
        wMax = info['wMax']
        npoints = info['npoints']
        tol = info['tol']

        # Collect arguments
        if interpolate:
            wlist = np.linspace(-wMax, wMax, 2 * npoints)
            slist = ref.power_spectrum(wlist)
            args1 = {'wlist': wlist, 'S': slist}
        else:
            args1 = {'S': ref.power_spectrum}
        args2 = {'wMax': wMax} if provide_wmax else {}
        args3 = {'T': ref.T} if provide_temp else {}

        env = BosonicEnvironment.from_power_spectrum(
            **args1, **args2, **args3,
        )

        # Determine which characteristic functions should be accessible
        skip_sd = False
        skip_cf = False
        if not provide_temp:
            skip_sd = True
            with np.testing.assert_raises(ValueError):
                env.spectral_density(0)
        if not interpolate and not provide_wmax:
            skip_cf = True
            with np.testing.assert_raises(ValueError):
                env.correlation_function(0)

        assert_environment_guarantees(env, skip_cf=skip_cf, skip_sd=skip_sd)
        assert_equivalent(env, ref, skip_cf=skip_cf, skip_sd=skip_sd, tol=tol)




todo="""
    def test_approx_by_cf_fit(self, params):
        lam, gamma, w0, T, t, w, w2, corr = params
        for k in range(len(lam)):
            bb5 = BosonicEnvironment.from_correlation_function(
                corr[k], t, T=T[k])
            bb6, finfo = bb5.approx_by_cf_fit(
                t, target_rsme=None, Nr_max=2, Ni_max=1)
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
            bb6, finfo = bb5.approx_by_sd_fit(w, Nmax=1, Nk=10)
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
"""
