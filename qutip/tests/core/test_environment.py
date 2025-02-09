import pytest
from importlib.util import find_spec

from numbers import Number

import numpy as np
from scipy.integrate import quad_vec
from qutip.utilities import n_thermal

from qutip.core.environment import (
    BosonicEnvironment,
    DrudeLorentzEnvironment,
    UnderDampedEnvironment,
    OhmicEnvironment,
    CFExponent,
    ExponentialBosonicEnvironment,
    LorentzianEnvironment,
    ExponentialFermionicEnvironment
)


def assert_guarantees(env, skip_sd=False, skip_cf=False, skip_ps=False):
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


class OhmicReference:
    def __init__(self, T, alpha, wc, s):
        self.T = T
        self.alpha = alpha
        self.wc = wc
        self.s = s

    def spectral_density(self, w):
        # only valid for w >= 0
        return (self.alpha * self.wc**(1 - self.s) *
                w**self.s * np.exp(-w / self.wc))

    def power_spectrum(self, w, eps=1e-16):
        # at zero frequency, take limit w->0
        w = np.array(w)
        w[w == 0] = eps
        return 2 * np.sign(w) * self.spectral_density(np.abs(w)) * (
            n_thermal(w, self.T) + 1
        )

    def correlation_function(self, t):
        mp = pytest.importorskip("mpmath")

        # only valid for t >= 0
        if self.T == 0:
            return self._cf_zeroT(t)

        beta = 1 / self.T
        corr = (
            (self.alpha / np.pi) * self.wc**(1 - self.s) *
            beta**(-(self.s + 1)) * mp.gamma(self.s + 1)
        )
        z1_u = (1 + beta * self.wc - 1.0j * self.wc * t) / (beta * self.wc)
        z2_u = (1 + 1.0j * self.wc * t) / (beta * self.wc)

        return np.array([
            complex(corr * (mp.zeta(self.s + 1, u1) + mp.zeta(self.s + 1, u2)))
            for u1, u2 in zip(z1_u, z2_u)
        ])

    def _cf_zeroT(self, t):
        mp = pytest.importorskip("mpmath")

        return (
            self.alpha / np.pi * self.wc**(self.s + 1) *
            complex(mp.gamma(self.s + 1)) *
            (1 + 1j * self.wc * t)**(-(self.s + 1))
        )


class SingleExponentReference:
    def __init__(self, coefficient, exponent, T):
        self.coefficient = coefficient
        self.exponent = exponent
        self.T = T

    def spectral_density(self, w):
        # only valid for w >= 0
        w = np.asarray(w)
        result = self.power_spectrum(w) / 2 / (n_thermal(w, self.T) + 1)
        result[w == 0] = 0
        return result

    def power_spectrum(self, w):
        return 2 * np.real(
            self.coefficient / (self.exponent - 1j * w)
        )

    def correlation_function(self, t):
        # only valid for t >= 0
        return self.coefficient * np.exp(-self.exponent * t)


class TestBosonicEnvironment:
    @pytest.mark.parametrize(["ref", "info"], [
        pytest.param(UDReference(gamma=.1, lam=.5, w0=1, T=.5),
                     {'tMax': 125, 'npoints': 300, 'tol': 5e-3},
                     id="UD finite T"),
        pytest.param(DLReference(gamma=.5, lam=.1, T=.5),
                     {'tMax': 15, 'npoints': 300, 'tol': 1e-2},
                     id="DL finite T"),
        pytest.param(OhmicReference(T=0, alpha=1, wc=10, s=1),
                     {'tMax': 10, 'npoints': 500, 'tol': 1e-2},
                     id="Ohmic zero T"),
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
            with pytest.raises(ValueError) as err:
                env.power_spectrum(0)
            assert str(err.value) == (
                'The support of the correlation function (tMax) must be '
                'specified for this operation.')
            with pytest.raises(ValueError) as err:
                env.spectral_density(0)
            assert str(err.value) == (
                'The support of the correlation function (tMax) must be '
                'specified for this operation.')
        elif not provide_temp:
            skip_sd = True
            with pytest.raises(ValueError) as err:
                env.spectral_density(0)
            assert str(err.value) == (
                'The temperature must be specified for this operation.')

        assert_guarantees(env, skip_sd=skip_sd, skip_ps=skip_ps)
        assert_equivalent(env, ref, skip_sd=skip_sd, skip_ps=skip_ps,
                          tol=tol, tMax=tMax)

    @pytest.mark.parametrize(["ref", "info"], [
        pytest.param(UDReference(gamma=.1, lam=.5, w0=1, T=.5),
                     {'wMax': 5, 'npoints': 200, 'tol': 5e-3},
                     id="UD finite T"),
        pytest.param(DLReference(gamma=.5, lam=.1, T=.5),
                     {'wMax': 200, 'npoints': 1500, 'tol': 5e-3},
                     id="DL finite T"),
        pytest.param(OhmicReference(T=0, alpha=1, wc=10, s=1),
                     {'wMax': 150, 'npoints': 150, 'tol': 5e-3},
                     id="Ohmic zero T"),
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
            with pytest.raises(ValueError) as err:
                env.power_spectrum(0)
            assert str(err.value) == (
                'The temperature must be specified for this operation.')
            with pytest.raises(ValueError) as err:
                env.correlation_function(0)
            assert str(err.value) == (
                'The temperature must be specified for this operation.')
        elif not interpolate and not provide_wmax:
            skip_cf = True
            with pytest.raises(ValueError) as err:
                env.correlation_function(0)
            assert str(err.value) == (
                'The support of the spectral density (wMax) must be '
                'specified for this operation.')

        assert_guarantees(env, skip_cf=skip_cf, skip_ps=skip_ps)
        assert_equivalent(env, ref, skip_cf=skip_cf, skip_ps=skip_ps,
                          tol=tol, wMax=wMax)

    @pytest.mark.parametrize(["ref", "info"], [
        pytest.param(UDReference(gamma=.1, lam=.5, w0=1, T=.5),
                     {'wMax': 5, 'npoints': 200, 'tol': 5e-3},
                     id="UD finite T"),
        pytest.param(DLReference(gamma=.5, lam=.1, T=.5),
                     {'wMax': 200, 'npoints': 1500, 'tol': 5e-3},
                     id="DL finite T"),
        pytest.param(OhmicReference(T=0, alpha=1, wc=10, s=1),
                     {'wMax': 150, 'npoints': 2000, 'tol': 5e-3},
                     id="Ohmic zero T"),
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
            wlist = np.linspace(-wMax, wMax, 2 * npoints + 1)
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
            with pytest.raises(ValueError) as err:
                env.spectral_density(0)
            assert str(err.value) == (
                'The temperature must be specified for this operation.')
        if not interpolate and not provide_wmax:
            skip_cf = True
            with pytest.raises(ValueError) as err:
                env.correlation_function(0)
            assert str(err.value) == (
                'The support of the power spectrum (wMax) must be '
                'specified for this operation.')

        assert_guarantees(env, skip_cf=skip_cf, skip_sd=skip_sd)
        assert_equivalent(env, ref, skip_cf=skip_cf, skip_sd=skip_sd,
                          tol=tol, wMax=wMax)

    @pytest.mark.parametrize(["reference", "tMax"], [
        pytest.param(DLReference(.5, .1, .5), 15, id="DL Example"),
    ])
    @pytest.mark.parametrize(["full_ansatz", "tol"], [
        pytest.param(True, 1e-3, id="Full ansatz"),
        pytest.param(False, 5e-3, id="Not-full ansatz"),
    ])
    def test_fixed_cf_fit(self, reference, tMax, full_ansatz, tol):
        # fixed number of exponents
        env = BosonicEnvironment.from_correlation_function(
            reference.correlation_function, T=reference.T
        )
        tlist = np.linspace(0, tMax, 100)[1:] # exclude t=0
        fit, info = env.approx_by_cf_fit(
            tlist, target_rsme=None, Nr_max=2, Ni_max=2,
            full_ansatz=full_ansatz, combine=False
        )

        assert isinstance(fit, ExponentialBosonicEnvironment)
        assert len(fit.exponents) == 8
        assert fit.T == env.T
        assert fit.tag is None
        assert_equivalent(
            fit, env, tol=tol, skip_sd=True, skip_ps=True, tMax=tMax
        )

        assert info["Nr"] == 2
        assert info["Ni"] == 2
        assert info["rmse_real"] < 5e-3
        assert info["rmse_imag"] < 5e-3
        for key in ["fit_time_real", "fit_time_imag",
                    "params_real", "params_imag", "summary"]:
            assert key in info

    @pytest.mark.parametrize(["reference", "tMax", "tol"], [
        pytest.param(DLReference(.5, .1, .5), 15, 5e-2, id="DL Example"),
        pytest.param(SingleExponentReference(1, 2 + .5j, None), 10, 1e-5,
                     id="Exponential function")
    ])
    @pytest.mark.parametrize("full_ansatz", [True, False])
    def test_dynamic_cf_fit(self, reference, tMax, tol, full_ansatz):
        # dynamic number of exponents
        env = BosonicEnvironment.from_correlation_function(
            reference.correlation_function, tag="test"
        )
        tlist = np.linspace(0, tMax, 100)[1:] # exclude t=0
        fit, info = env.approx_by_cf_fit(
            tlist, target_rsme=0.01, Nr_max=3, Ni_max=3,
            full_ansatz=full_ansatz
        )

        assert isinstance(fit, ExponentialBosonicEnvironment)
        assert fit.T == env.T
        assert fit.tag == ("test", "CF Fit")
        assert_equivalent(
            fit, env, tol=tol, skip_sd=True, skip_ps=True, tMax=tMax
        )

        assert info["Nr"] == 1
        assert info["Ni"] == 1
        assert info["rmse_real"] <= 0.01
        assert info["rmse_imag"] <= 0.01
        for key in ["fit_time_real", "fit_time_imag",
                    "params_real", "params_imag", "summary"]:
            assert key in info

    @pytest.mark.parametrize(["reference", "wMax", "tol"], [
        pytest.param(OhmicReference(3, .75, 10, 1), 15, 5e-2, id="DL Example"),
    ])
    def test_fixed_sd_fit(self, reference, wMax, tol):
        # fixed number of lorentzians
        env = BosonicEnvironment.from_spectral_density(
            reference.spectral_density, T=reference.T
        )
        wlist = np.linspace(0, wMax, 100)
        fit, info = env.approx_by_sd_fit(
            wlist, Nk=1, target_rmse=None, Nmax=4, combine=False
        )

        assert isinstance(fit, ExponentialBosonicEnvironment)
        assert len(fit.exponents) == 4 * (1 + 4)
        assert fit.T == env.T
        assert fit.tag is None
        assert_equivalent(
            fit, env, tol=tol, skip_cf=True, skip_ps=True, wMax=wMax
        )

        assert info["N"] == 4
        assert info["Nk"] == 1
        assert info["rmse"] < 5e-3
        for key in ["fit_time", "params", "summary"]:
            assert key in info

    @pytest.mark.parametrize(["reference", "wMax", "tol", "params"], [
        pytest.param(OhmicReference(3, .75, 10, 1), 15, .2, {},
                     id="DL Example"),
        pytest.param(UDReference(1, .5, .1, 1), 2, 1e-4,
                     {'guess': [.2, .5, 1]}, id='UD Example'),
    ])
    def test_dynamic_sd_fit(self, reference, wMax, tol, params):
        # dynamic number of exponents
        env = BosonicEnvironment.from_spectral_density(
            reference.spectral_density, T=reference.T, tag="test"
        )
        wlist = np.linspace(0, wMax, 100)
        fit, info = env.approx_by_sd_fit(
            wlist, Nk=1, target_rmse=0.01, Nmax=5, **params
        )

        assert isinstance(fit, ExponentialBosonicEnvironment)
        assert fit.T == env.T
        assert fit.tag == ("test", "SD Fit")
        assert_equivalent(
            fit, env, tol=tol, skip_cf=True, skip_ps=True, wMax=wMax
        )

        assert info["N"] < 5
        assert info["Nk"] == 1
        assert info["rmse"] < 0.01
        for key in ["fit_time", "params", "summary"]:
            assert key in info


@pytest.mark.parametrize("params", [
    {'gamma': 2.5, 'lam': .75, 'T': 1.5}
])
class TestDLEnvironment:
    def test_matches_reference(self, params):
        ref = DLReference(**params)
        env = DrudeLorentzEnvironment(**params)

        assert_guarantees(env)
        assert_equivalent(env, ref, tol=1e-8)

    def test_zero_temperature(self, params):
        params_T0 = {**params, 'T': 0}
        ref = DLReference(**params_T0)
        env = DrudeLorentzEnvironment(**params_T0)

        with pytest.raises(ValueError) as err:
            env.correlation_function(0)
        assert str(err.value) == (
            'The Drude-Lorentz correlation function '
            'diverges at zero temperature.')
        with pytest.raises(ValueError) as err:
            env.approx_by_matsubara(10)
        assert str(err.value) == (
            'The Drude-Lorentz correlation function '
            'diverges at zero temperature.')
        with pytest.raises(ValueError) as err:
            env.approx_by_pade(10)
        assert str(err.value) == (
            'The Drude-Lorentz correlation function '
            'diverges at zero temperature.')

        assert_guarantees(env, skip_cf=True)
        assert_equivalent(env, ref, tol=1e-6, skip_cf=True)

    @pytest.mark.parametrize("tag", [None, "test"])
    def test_matsubara_approx(self, params, tag):
        Nk = 25
        original_tag = object()
        env = DrudeLorentzEnvironment(**params, tag=original_tag)

        approx = env.approx_by_matsubara(Nk, combine=False, tag=tag)
        assert isinstance(approx, ExponentialBosonicEnvironment)
        assert len(approx.exponents) == Nk + 2  # (Nk+1) real + 1 imag
        if tag is None:
            assert approx.tag == (original_tag, "Matsubara Truncation")
        else:
            assert approx.tag == tag
        assert approx.T == env.T

        # CF at t=0 might not match, which is okay
        assert_equivalent(approx, env, tol=1e-2, skip_cf=True)

        approx_combine, delta = env.approx_by_matsubara(
            Nk, compute_delta=True, combine=True
        )
        assert isinstance(approx_combine, ExponentialBosonicEnvironment)
        assert len(approx_combine.exponents) < Nk + 2
        assert approx_combine.T == env.T

        assert_equivalent(approx_combine, approx, tol=1e-8)

        # Check terminator amplitude
        delta_ref = 2 * env.lam * env.T / env.gamma - 1j * env.lam
        for exp in approx_combine.exponents:
            delta_ref -= exp.coefficient / exp.exponent
        assert_allclose(delta, delta_ref, tol=1e-8)

    @pytest.mark.parametrize("tag", [None, "test"])
    def test_pade_approx(self, params, tag):
        Nk = 4
        original_tag = object()
        env = DrudeLorentzEnvironment(**params, tag=original_tag)

        approx = env.approx_by_pade(Nk, combine=False, tag=tag)
        assert isinstance(approx, ExponentialBosonicEnvironment)
        assert len(approx.exponents) == Nk + 2  # (Nk+1) real + 1 imag
        if tag is None:
            assert approx.tag == (original_tag, "Pade Truncation")
        else:
            assert approx.tag == tag
        assert approx.T == env.T

        # Wow, Pade is so much better
        assert_equivalent(approx, env, tol=1e-8, skip_cf=True)

        approx_combine, delta = env.approx_by_pade(
            Nk, combine=True, compute_delta=True
        )
        assert isinstance(approx_combine, ExponentialBosonicEnvironment)
        assert len(approx_combine.exponents) < Nk + 2
        assert approx_combine.T == env.T

        assert_equivalent(approx_combine, approx, tol=1e-8)

        # Check terminator amplitude
        delta_ref = 2 * env.lam * env.T / env.gamma - 1j * env.lam
        for exp in approx_combine.exponents:
            delta_ref -= exp.coefficient / exp.exponent
        assert_allclose(delta, delta_ref, tol=1e-8)


class TestUDEnvironment:
    @pytest.mark.parametrize("params", [
        pytest.param({'gamma': 2.5, 'lam': .75, 'w0': 5, 'T': 1.5},
                     id='finite T'),
        pytest.param({'gamma': 2.5, 'lam': .75, 'w0': 5, 'T': 0},
                     id='zero T'),
    ])
    def test_matches_reference(self, params):
        ref = UDReference(**params)
        env = UnderDampedEnvironment(**params)

        assert_guarantees(env)
        # a bit higher tolerance since we currently compute CF via FFT
        assert_equivalent(env, ref, tol=1e-3)

    @pytest.mark.parametrize("params", [
        {'gamma': 2.5, 'lam': .75, 'w0': 5, 'T': 0},
    ])
    def test_zero_temperature(self, params):
        # Attempting a Matsubara expansion at T=0 should raise a warning
        # and return only the resonant (non-Matsubara) part of the expansion
        env = UnderDampedEnvironment(**params)

        tlist = np.linspace(0, 5, 100)
        resonant_approx = env.approx_by_matsubara(Nk=0)

        with pytest.warns(UserWarning) as record:
            test_approx = env.approx_by_matsubara(Nk=3)
        assert str(record[0].message) == (
            'The Matsubara expansion cannot be performed at zero temperature. '
            'Use other approaches such as fitting the correlation function.')

        assert_allclose(resonant_approx.correlation_function(tlist),
                        test_approx.correlation_function(tlist),
                        tol=1e-8)

    @pytest.mark.parametrize("params", [
        {'gamma': 2.5, 'lam': .75, 'w0': 5, 'T': 1.5},
    ])
    @pytest.mark.parametrize("tag", [None, "test"])
    def test_matsubara_approx(self, params, tag):
        Nk = 25
        original_tag = object()
        env = UnderDampedEnvironment(**params, tag=original_tag)

        approx = env.approx_by_matsubara(Nk, combine=False, tag=tag)
        assert isinstance(approx, ExponentialBosonicEnvironment)
        assert len(approx.exponents) == Nk + 4  # (Nk+2) real + 2 imag
        if tag is None:
            assert approx.tag == (original_tag, "Matsubara Truncation")
        else:
            assert approx.tag == tag
        assert approx.T == env.T

        assert_equivalent(approx, env, tol=1e-3)

        approx_combine = env.approx_by_matsubara(Nk, combine=True)
        assert isinstance(approx_combine, ExponentialBosonicEnvironment)
        assert len(approx_combine.exponents) < Nk + 4
        assert approx_combine.T == env.T

        assert_equivalent(approx_combine, approx, tol=1e-8)


@pytest.mark.parametrize("params", [
    pytest.param({'alpha': .75, 'wc': 10, 's': 1, 'T': 3},
                id='finite T ohmic'),
    pytest.param({'alpha': .75, 'wc': .5, 's': 1, 'T': 0},
                id='zero T ohmic'),
    pytest.param({'alpha': .75, 'wc': 10, 's': .5, 'T': 3},
                id='finite T subohmic'),
    pytest.param({'alpha': .75, 'wc': .5, 's': .5, 'T': 0},
                id='zero T subohmic'),
    pytest.param({'alpha': .75, 'wc': 10, 's': 5, 'T': 3},
                id='finite T superohmic'),
    pytest.param({'alpha': .75, 'wc': .5, 's': 5, 'T': 0},
                id='zero T superohmic'),
])
class TestOhmicEnvironment:
    def test_matches_reference(self, params):
        mpmath_missing = (find_spec('mpmath') is None)

        ref = OhmicReference(**params)
        if mpmath_missing:
            with pytest.warns(UserWarning):
                env = OhmicEnvironment(**params)
        else:
            env = OhmicEnvironment(**params)

        assert_guarantees(env, skip_cf=mpmath_missing)
        assert_equivalent(env, ref, tol=1e-8, skip_cf=mpmath_missing)


class TestCFExponent:
    def test_create(self):
        exp_r = CFExponent("R", ck=1.0, vk=2.0)
        check_exponent(exp_r, "R", 1.0, 2.0)

        exp_i = CFExponent("I", ck=1.0, vk=2.0)
        check_exponent(exp_i, "I", 1.0j, 2.0)

        exp_ri = CFExponent("RI", ck=1.0, vk=2.0, ck2=3.0)
        check_exponent(exp_ri, "RI", 1.0 + 3.0j, 2.0)

        exp_p = CFExponent("+", ck=1.0, vk=2.0)
        check_exponent(exp_p, "+", 1.0, 2.0)

        exp_m = CFExponent("-", ck=1.0, vk=2.0)
        check_exponent(exp_m, "-", 1.0, 2.0)

        exp_tag = CFExponent("R", ck=1.0, vk=2.0, tag="tag1")
        check_exponent(exp_tag, "R", 1.0, 2.0, tag="tag1")

        for exp_type in ["R", "I", "+", "-"]:
            with pytest.raises(ValueError) as err:
                CFExponent(exp_type, ck=1.0, vk=2.0, ck2=3.0)
            assert str(err.value) == (
                "Second co-efficient (ck2) should only be specified for RI"
                " exponents"
            )

        with pytest.raises(ValueError) as err:
            CFExponent('RI', ck=1.0, vk=2.0)
        assert str(err.value) == "RI exponents require ck2"

    def test_repr(self):
        exp1 = CFExponent("R", ck=1.0, vk=2.0)
        assert repr(exp1) == (
            "<CFExponent type=R ck=1.0 vk=2.0 ck2=None"
            " fermionic=False tag=None>"
        )
        exp2 = CFExponent("+", ck=1.0, vk=2.0, tag="bath1")
        assert repr(exp2) == (
            "<CFExponent type=+ ck=1.0 vk=2.0 ck2=None"
            " fermionic=True tag='bath1'>"
        )


def check_exponent(exp, type, coefficient, exponent, tag=None):
    assert exp.type is CFExponent.types[type]
    assert exp.fermionic == (type in ["+", "-"])
    assert exp.coefficient == pytest.approx(coefficient)
    assert exp.exponent == pytest.approx(exponent)
    assert exp.tag == tag


class TestExpBosonicEnv:
    def test_create(self):
        env = ExponentialBosonicEnvironment([1.], [0.5], [2.], [0.6])
        exp_r, exp_i = env.exponents
        check_exponent(exp_r, "R", 1.0, 0.5)
        check_exponent(exp_i, "I", 2.0j, 0.6)

        env = ExponentialBosonicEnvironment([1.], [0.5], [2.], [0.5])
        [exp_ri] = env.exponents
        check_exponent(exp_ri, "RI", 1.0 + 2.0j, 0.5)

        env = ExponentialBosonicEnvironment(
            [1.], [0.5], [2.], [0.6], tag="bath1"
        )
        exp_r, exp_i = env.exponents
        check_exponent(exp_r, "R", 1.0, 0.5, tag="bath1")
        check_exponent(exp_i, "I", 2.0j, 0.6, tag="bath1")

        exp1 = CFExponent("RI", ck=1.0, vk=2.0, ck2=3.0)
        exp2 = CFExponent("I", ck=1.0, vk=2.0)
        env = ExponentialBosonicEnvironment(
            exponents=[exp1, exp2], combine=False)
        assert env.exponents == [exp1, exp2]

        env = ExponentialBosonicEnvironment(
            [], [], [1.0], [2.0], exponents=[exp1], combine=True)
        [exp_ri] = env.exponents
        check_exponent(exp_ri, "RI", 1.0 + 4.0j, 2)

        with pytest.raises(ValueError) as err:
            ExponentialBosonicEnvironment([1.], [], [2.], [0.6])
        assert str(err.value) == (
            "The exponent lists ck_real and vk_real, and ck_imag and"
            " vk_imag must be the same length."
        )

        with pytest.raises(ValueError) as err:
            ExponentialBosonicEnvironment([1.], [0.5], [2.], [])
        assert str(err.value) == (
            "The exponent lists ck_real and vk_real, and ck_imag and"
            " vk_imag must be the same length."
        )

        with pytest.raises(ValueError) as err:
            ExponentialBosonicEnvironment()
        assert str(err.value) == (
            "Either the parameter `exponents` or the parameters "
            "`ck_real`, `vk_real`, `ck_imag`, `vk_imag` must be provided."
        )

        with pytest.raises(ValueError) as err:
            ExponentialBosonicEnvironment(
                exponents=[exp1, CFExponent('+', 1.0, 2.0)]
            )
        assert str(err.value) == (
            "Fermionic exponent passed to exponential bosonic environment."
        )

    @pytest.mark.parametrize("provide_temp", [True, False])
    def test_matches_reference(self, provide_temp):
        if provide_temp:
            args = {'T': .5}
            skip_sd = False
        else:
            args = {}
            skip_sd = True

        env = ExponentialBosonicEnvironment([1.], [0.5], [2.], [0.5], **args)
        assert_guarantees(env, skip_sd=skip_sd)
        ref = SingleExponentReference(coefficient=(1 + 2j), exponent=.5, T=.5)
        assert_equivalent(env, ref, tol=1e-8, skip_sd=skip_sd)

        if not provide_temp:
            with pytest.raises(ValueError) as err:
                env.spectral_density(0)
            assert str(err.value) == (
                'The temperature must be specified for this operation.'
            )


# ----- Fermionic Environments -----


def assert_guarantees_f(env, check_db=False, beta=None, mu=None):
    """
    Checks the argument types accepted by the SD, CFs and PSs of the provided
    fermionic environment, that the CFs satisfy their symmetries and that the
    PSs are real. If `check_db` is True, `beta` and `mu` must be provided and
    we also check the detailed balance condition S^- = e^{β(ω-μ)} S^+.
    """

    # SD, CF and PS can be called with number and array
    # Result is required to have same type, ndim and len as argument
    for fun in [env.spectral_density,
                env.correlation_function_plus,
                env.correlation_function_minus,
                env.power_spectrum_plus,
                env.power_spectrum_minus]:
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

    # SD must be real
    sd = env.spectral_density(np.linspace(-10, 10, 20))
    np.testing.assert_allclose(np.imag(sd), np.zeros_like(sd))

    # CF symmetry
    for fun in [env.correlation_function_plus, env.correlation_function_minus]:
        cf = fun(np.linspace(-10, 10, 20))
        cf_reversed = cf[::-1]
        np.testing.assert_allclose(cf, np.conjugate(cf_reversed))

    # PS must be real and their sum must be SD
    wlist = np.linspace(-10, 10, 20)
    psp = env.power_spectrum_plus(wlist)
    psm = env.power_spectrum_minus(wlist)
    np.testing.assert_allclose(np.imag(psp), np.zeros_like(psp))
    np.testing.assert_allclose(np.imag(psm), np.zeros_like(psm))
    np.testing.assert_allclose(psp + psm, sd)

    # Detailed balance
    if not check_db:
        return

    if beta == np.inf:
        foo = psp[wlist > mu]
        np.testing.assert_allclose(foo, np.zeros_like(foo))
        bar = psp[wlist < mu]
        np.testing.assert_allclose(bar, np.zeros_like(bar))
        return

    factor = np.exp(beta * (wlist - mu))
    np.testing.assert_allclose(psm, factor * psp)

def assert_equivalent_f(env1, env2, *, tol,
                      tMax=25, wMax=10):
    """
    Checks that two fermionic environments have the same SD, CFs and PSs
    (up to given tolerance)
    """
    tlist = np.linspace(0, tMax, 100)
    wlist = np.linspace(-wMax, wMax, 100)

    assert_allclose(env1.spectral_density(wlist),
                    env2.spectral_density(wlist), tol)
    assert_allclose(env1.correlation_function_plus(tlist),
                    env2.correlation_function_plus(tlist), tol)
    assert_allclose(env1.correlation_function_minus(tlist),
                    env2.correlation_function_minus(tlist), tol)
    assert_allclose(env1.power_spectrum_plus(wlist),
                    env2.power_spectrum_plus(wlist), tol)
    assert_allclose(env1.power_spectrum_minus(wlist),
                    env2.power_spectrum_minus(wlist), tol)


class LorentzianReference:
    def __init__(self, T, mu, gamma, W, omega0):
        self.T = T
        self.mu = mu
        self.gamma = gamma
        self.W = W
        self.omega0 = omega0

    def _f(self, x):
        if self.T == 0:
            return np.heaviside(-x, 0.5)
        else:
            return 1 / (np.exp(x / self.T) + 1)

    def spectral_density(self, w):
        return self.gamma * self.W**2 / (
            (w - self.omega0)**2 + self.W**2
        )

    def power_spectrum_plus(self, w):
        return self.spectral_density(w) * self._f(w - self.mu)

    def power_spectrum_minus(self, w):
        return self.spectral_density(w) * self._f(-w + self.mu)

    def correlation_function_plus(self, t, Nk=5000):
        # only valid for t >= 0
        T = self.T
        mu = self.mu
        gamma = self.gamma
        W = self.W
        omega0 = self.omega0

        # zero temperature CFs not implemented yet
        if T == 0:
            assert False

        ck = [(gamma * W / 2 * self._f(omega0 - mu + 1j * W))]
        ck.extend(
            1j * gamma * W**2 * T / (
                (1j * (omega0 - mu) + (2 * k - 1) * np.pi * T)**2 - W**2
            ) for k in range(1, Nk + 1)
        )
        vk = [W - 1j * omega0]
        vk.extend(
            (2 * k - 1) * np.pi * T - 1j * mu
            for k in range(1, Nk + 1)
        )

        result = np.zeros_like(t, dtype=complex)
        for c, v in zip(ck, vk):
            result += c * np.exp(-v * t)
        return result

    def correlation_function_minus(self, t, Nk=5000):
        # only valid for t >= 0
        T = self.T
        mu = self.mu
        gamma = self.gamma
        W = self.W
        omega0 = self.omega0

        # zero temperature CFs not implemented yet
        if T == 0:
            assert False

        ck = [(gamma * W / 2 * self._f(mu - omega0 + 1j * W))]
        ck.extend(
            1j * gamma * W**2 * T / (
                (1j * (mu - omega0) + (2 * k - 1) * np.pi * T)**2 - W**2
            ) for k in range(1, Nk + 1)
        )
        vk = [W + 1j * omega0]
        vk.extend(
            (2 * k - 1) * np.pi * T + 1j * mu
            for k in range(1, Nk + 1)
        )

        result = np.zeros_like(t, dtype=complex)
        for c, v in zip(ck, vk):
            result += c * np.exp(-v * t)
        return result


class SinglePlusExpReference:
    def __init__(self, coefficient, exponent):
        self.coefficient = coefficient
        self.exponent = exponent

    def spectral_density(self, w):
        return self.power_spectrum_plus(w)

    def power_spectrum_plus(self, w):
        return 2 * np.real(
            self.coefficient / (self.exponent + 1j * w)
        )

    def power_spectrum_minus(self, w):
        return np.zeros_like(w)

    def correlation_function_plus(self, t):
        # only valid for t >= 0
        return self.coefficient * np.exp(-self.exponent * t)

    def correlation_function_minus(self, t):
        return np.zeros_like(t)


class SingleMinusExpReference:
    def __init__(self, coefficient, exponent):
        self.coefficient = coefficient
        self.exponent = exponent

    def spectral_density(self, w):
        return self.power_spectrum_minus(w)

    def power_spectrum_plus(self, w):
        return np.zeros_like(w)

    def power_spectrum_minus(self, w):
        return 2 * np.real(
            self.coefficient / (self.exponent - 1j * w)
        )

    def correlation_function_plus(self, t):
        return np.zeros_like(t)

    def correlation_function_minus(self, t):
        # only valid for t >= 0
        return self.coefficient * np.exp(-self.exponent * t)


@pytest.mark.parametrize("params", [
    pytest.param({'T': 1.5, 'mu': 1, 'gamma': .75, 'W': .5, 'omega0': 1},
                 id='finite T - mu=omega0'),
    pytest.param({'T': 1.5, 'mu': 1, 'gamma': .75, 'W': .5, 'omega0': 3},
                 id='finite T - mu<omega0'),
    pytest.param({'T': 1.5, 'mu': 1, 'gamma': .75, 'W': 5, 'omega0': .5},
                 id='finite T - mu>omega0 - flat'),
    pytest.param({'T': 0, 'mu': 1, 'gamma': .75, 'W': .5, 'omega0': 1},
                 id='zero T',
                 marks=pytest.mark.xfail(reason='zero temp not implemented')),
])
class TestLorentzianEnvironment:
    def test_matches_reference(self, params):
        ref = LorentzianReference(**params)
        env = LorentzianEnvironment(**params)

        assert_guarantees_f(
            env, check_db=True, beta=(1 / params['T']), mu=params['mu'])
        assert_equivalent_f(env, ref, tol=5e-5)

    @pytest.mark.parametrize("tag", [None, "test"])
    def test_matsubara_approx(self, params, tag):
        Nk = 50
        original_tag = object()
        env = LorentzianEnvironment(**params, tag=original_tag)

        approx = env.approx_by_matsubara(Nk, tag=tag)
        assert isinstance(approx, ExponentialFermionicEnvironment)
        assert_guarantees_f(approx, check_db=False)
        assert len(approx.exponents) == 2 * (Nk + 1)  # (Nk+1) each + and -
        if tag is None:
            assert approx.tag == (original_tag, "Matsubara Truncation")
        else:
            assert approx.tag == tag
        assert approx.T == env.T
        assert approx.mu == env.mu

        assert_equivalent_f(approx, env, tol=5e-3)

    @pytest.mark.parametrize("tag", [None, "test"])
    def test_pade_approx(self, params, tag):
        Nk = 5
        original_tag = object()

        ref = LorentzianReference(**params)
        env = LorentzianEnvironment(**params, tag=original_tag)

        approx = env.approx_by_pade(Nk, tag=tag)
        assert isinstance(approx, ExponentialFermionicEnvironment)
        assert_guarantees_f(approx, check_db=False)
        assert len(approx.exponents) == 2 * (Nk + 1)  # (Nk+1) each + and -
        if tag is None:
            assert approx.tag == (original_tag, "Pade Truncation")
        else:
            assert approx.tag == tag
        assert approx.T == env.T
        assert approx.mu == env.mu

        assert_equivalent_f(approx, ref, tol=5e-5)


class TestExpFermionicEnv:
    def test_create(self):
        env = ExponentialFermionicEnvironment([1.], [0.5], [2.], [0.6])
        exp_p, exp_m = env.exponents
        check_exponent(exp_p, "+", 1.0, 0.5)
        check_exponent(exp_m, "-", 2.0, 0.6)

        env = ExponentialFermionicEnvironment(
            [1.], [0.5], [2.], [0.6], tag="bath1")
        exp_p, exp_m = env.exponents
        check_exponent(exp_p, "+", 1.0, 0.5, tag="bath1")
        check_exponent(exp_m, "-", 2.0, 0.6, tag="bath1")

        exp1 = CFExponent("+", ck=1.0j, vk=2.0)
        exp2 = CFExponent("-", ck=0.5, vk=1.0)
        env = ExponentialFermionicEnvironment(exponents=[exp1, exp2])
        assert env.exponents == [exp1, exp2]

        env = ExponentialFermionicEnvironment(
            [], [], [2.], [0.6], exponents=[exp1])
        exp_p, exp_m = env.exponents
        assert exp1 == exp_p
        check_exponent(exp_m, "-", 2.0, 0.6)

        with pytest.raises(ValueError) as err:
            ExponentialFermionicEnvironment([1.], [], [2.], [0.6])
        assert str(err.value) == (
            "The exponent lists ck_plus and vk_plus, and ck_minus and"
            " vk_minus must be the same length."
        )

        with pytest.raises(ValueError) as err:
            ExponentialFermionicEnvironment([1.], [0.5], [2.], [])
        assert str(err.value) == (
            "The exponent lists ck_plus and vk_plus, and ck_minus and"
            " vk_minus must be the same length."
        )

        with pytest.raises(ValueError) as err:
            ExponentialFermionicEnvironment()
        assert str(err.value) == (
            "Either the parameter `exponents` or the parameters "
            "`ck_plus`, `vk_plus`, `ck_minus`, `vk_minus` must be provided."
        )

        with pytest.raises(ValueError) as err:
            ExponentialFermionicEnvironment(
                exponents=[exp1, CFExponent('R', 1.0, 2.0)]
            )
        assert str(err.value) == (
            "Bosonic exponent passed to exponential fermionic environment."
        )

    def test_matches_reference(self):
        env = ExponentialFermionicEnvironment([1.], [0.5], [], [])
        assert_guarantees_f(env, check_db=False)
        ref = SinglePlusExpReference(coefficient=1, exponent=.5)
        assert_equivalent_f(env, ref, tol=1e-8)

        env = ExponentialFermionicEnvironment([], [], [2.], [0.6])
        assert_guarantees_f(env, check_db=False)
        ref = SingleMinusExpReference(coefficient=2, exponent=.6)
        assert_equivalent_f(env, ref, tol=1e-8)
