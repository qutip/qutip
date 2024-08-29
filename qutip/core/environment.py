"""
Classes that describe environments of open quantum systems
"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['BosonicEnvironment',
           'DrudeLorentzEnvironment',
           'UnderDampedEnvironment',
           'OhmicEnvironment',
           'CFExponent',
           'ExponentialBosonicEnvironment']

import abc
import enum
from typing import Any, Callable
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvalsh
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d

try:
    from mpmath import mp
    _mpmath_available = True
except ModuleNotFoundError:
    _mpmath_available = False

from ..utilities import (n_thermal, CorrelationFitter, SpectralFitter)


# TODO fermionic environment
# TODO environment, bath, or reservoir?
# TODO revise all documentation
# TODO tests for this module
# TODO add revised tests for HEOM (keeping the current tests as they are)
# TODO update HEOM example notebooks
# TODO clarify scope (single coupling operator, sometimes thermal)

class BosonicEnvironment(abc.ABC):
    """
    The bosonic environment of an open quantum system. It is characterized by
    its spectral density or, equivalently, its power spectrum or its two-time
    auto-correlation functions.

    Use one of the functions `BosonicEnvironment.from_spectral_density`,
    `BosonicEnvironment.from_power_spectrum` or
    `BosonicEnvironment.from_correlation_function` to contruct an environment
    manually from one of these characteristic functions, or use a predefined
    sub-class such as the `DrudeLorentzEnvironment` or the
    `UnderDampedEnvironment`.
    """
    # TODO: proper links in docstring
    # TODO: properly document all our definitions of these functions in the users guide
    # especially how we handle negative frequencies and all that stuff

    def __init__(self, T: float = None, *, tag: Any = None):
        self.T = T
        self.tag = tag

        # TODO unsure about names
        # TODO should these be available always?
        # TODO what if T is not set?
        # TODO add AAA fitting in later PR
        self._avail_approximators = {
            'correlation_fit': self._fit_correlation_function,
            'underdamped_fit': self._fit_spectral_density,
        }

    @abc.abstractmethod
    def spectral_density(self, w: float | ArrayLike):
        # TODO docstring
        ...

    @abc.abstractmethod
    def correlation_function(self, t: float | ArrayLike, **kwargs):
        # TODO docstring explaining kwargs
        # Default implementation: calculate CF from SD by numerical integration
        # TODO could this also be done via FFT?
        def integrand(w, t):
            return self.spectral_density(w) / np.pi * (
                (2 * n_thermal(w, self.T) + 1) * np.cos(w * t)
                - 1j * np.sin(w * t)
            )

        result = quad_vec(lambda w: integrand(w, t), 0, np.inf, **kwargs)
        return result[0]

    @abc.abstractmethod
    def power_spectrum(self, w: float | ArrayLike, dt: float = 1e-5):
        # TODO docstring explaining dt
        # Default implementation: calculate PS from SD directly
        w = np.array(w, dtype=float)

        # at omega=0, typically SD is zero and n_thermal is undefined
        # set omega to small positive value to capture limit as omega -> 0
        w[w == 0.0] += 1e-6
        if self.T != 0:
            S = (2 * np.sign(w) * self.spectral_density(np.abs(w)) *
                 (n_thermal(w, self.T) + 1))
        else:
            S = 2 * np.heaviside(w, 0) * self.spectral_density(w)
        return S

    def exponential_approximation(self, method, **kwargs):
        # TODO documentation
        # TODO is this the right way of doing it?
        approximator = self._avail_approximators.get(method, None)
        if approximator is None:
            raise ValueError(f"Unknown approximation method: {method}")
        return approximator(**kwargs)

    def _fit_correlation_function(self, tlist, Nr, Ni, full_ansatz=False):
        """
        Generates a reservoir from the correlation function

        Parameters
        ----------
        tlist: obj:`np.ndarray`
            The time range on which to perform the fit
        Nr: int
            The number of modes to use for the fit of the real part 
        Ni: int
            The number of modes to use for the fit of the real part
        full_ansatz: bool
            Whether to use a fit of the imaginary and real parts that is 
            complex

        Returns
        -------
        A bosonic reservoir
        """
        fitter = CorrelationFitter(self.T, tlist, self.correlation_function)
        return fitter.get_fit(Nr=Nr, Ni=Ni, full_ansatz=full_ansatz)

    def _fit_spectral_density(self, wlist, N, Nk):
        """
        Generates a reservoir from the spectral density

        Parameters
        ----------
        wlist: obj:`np.ndarray`
            The frequency range on which to perform the fit
        N: int
            The number of modes to use for the fit
        Nk: int
            The number of exponents to use in each mode

        Returns
        -------
        A bosonic reservoir
        """
        fitter = SpectralFitter(self.T, wlist, self.spectral_density)
        return fitter.get_fit(N=N, Nk=Nk)

    # TODO: I thought these temperatures can be `None`. In what cases is that ok?
    @classmethod
    def from_correlation_function(
        cls,
        T: float,
        C: Callable[[float], complex] | ArrayLike,
        tlist: ArrayLike = None,
        *,
        tag: Any = None
    ) -> BosonicEnvironment:
        """
        Constructs a bosonic environment with the provided correlation
        function and temperature

        Parameters
        ----------
        T : float
            Bath temperature.

        C: callable or :obj:`np.array`
            The correlation function

        tlist : :obj:`np.array` (optional)
            The times where the correlation function is sampled (if it is
            provided as an array)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment
        """
        return _BosonicEnvironment_fromCF(T, C, tlist, tag)

    @classmethod
    def from_power_spectrum(
        cls,
        T: float,
        S: Callable[[float], float] | ArrayLike,
        wlist: ArrayLike = None,
        *,
        tag: Any = None
    ) -> BosonicEnvironment:
        """
        Constructs a bosonic environment with the provided power spectrum
        and temperature

        Parameters
        ----------
        T : float
            Bath temperature.

        S: callable or :obj:`np.array.`
            The power spectrum

        wlist : :obj:`np.array` (optional)
            The frequencies where the power spectrum is sampled (if it is
            provided as an array)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment
        """
        return _BosonicEnvironment_fromPS(T, S, wlist, tag)

    @classmethod
    def from_spectral_density(
        cls,
        T: float,
        J: Callable[[float], float] | ArrayLike,
        wlist: ArrayLike = None,
        *,
        tag: Any = None
    ) -> BosonicEnvironment:
        """
        Constructs a bosonic environment with the provided spectral density
        and temperature

        Parameters
        ----------
        T : float
            Bath temperature.

        J : callable or :obj:`np.array`
            The spectral density

        wlist : :obj:`np.array` (optional)
            The frequencies where the spectral density is sampled (if it is
            provided as an array)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment
        """
        return _BosonicEnvironment_fromSD(T, J, wlist, tag)


class _BosonicEnvironment_fromCF(BosonicEnvironment):
    def __init__(self, T, C, tlist=None, tag=None):
        super().__init__(T, tag=tag)
        self._cf = _complex_interpolation(C, tlist, 'correlation function')

    def correlation_function(self, t, **kwargs):
        # TODO document that the provided CF is only used for t>0
        # (or change this?)
        result = np.zeros_like(t, dtype=complex)
        positive_mask = t > 0
        non_positive_mask = ~positive_mask

        result[positive_mask] = self._cf(t[positive_mask])
        result[non_positive_mask] = np.conj(
            self._cf(np.abs(t[non_positive_mask]))
        )
        return result

    def spectral_density(self, w):
        # TODO do we have to worry about w=0 or T=0 special cases?
        # TODO add tests including these cases
        return self.power_spectrum(w) / (n_thermal(w, self.T) + 1) / 2

    def power_spectrum(self, w, dt=1e-5):
        wMax = max(np.abs(w[0]), np.abs(w[-1]))
        negative = _fft(self.correlation_function, wMax, dt)
        return negative(-w)


class _BosonicEnvironment_fromPS(BosonicEnvironment):
    def __init__(self, T, S, wlist=None, tag=None):
        super().__init__(T, tag=tag)
        self._ps = _real_interpolation(S, wlist, 'power spectrum')

    def correlation_function(self, t, **kwargs):
        return super().correlation_function(t, **kwargs)

    def spectral_density(self, w):
        # TODO is this okay at w=0 or do we have to do something like in _fromSD?
        return self.power_spectrum(w) / (n_thermal(w, self.T) + 1) / 2

    def power_spectrum(self, w, dt=1e-5):
        return self._ps(w)


class _BosonicEnvironment_fromSD(BosonicEnvironment):
    def __init__(self, T, J, wlist=None, tag=None):
        super().__init__(T, tag=tag)
        self._sd = _real_interpolation(J, wlist, 'spectral density')

    def correlation_function(self, t, **kwargs):
        return super().correlation_function(t, **kwargs)

    def spectral_density(self, w):
        return self._sd(w)

    def power_spectrum(self, w, dt=1e-5):
        return super().power_spectrum(w, dt)


class DrudeLorentzEnvironment(BosonicEnvironment):
    """
    Describes a Drude-Lorentz bosonic environment with the following
    parameters:

    Parameters
    ----------
    T : float
        Bath temperature.

    lam : float
        Coupling strength.

    gamma : float
        Bath spectral density cutoff frequency.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment
    """

    def __init__(
        self, T: float, lam: float, gamma: float, *, tag: Any = None
    ):
        super().__init__(T, tag=tag)

        self.lam = lam
        self.gamma = gamma

        self._avail_approximators.update({
            'matsubara': self._matsubara_approx,
            'pade': self._pade_approx,
        })

    def spectral_density(self, w: float | ArrayLike):
        r"""
        Calculates the Drude-Lorentz spectral density,

        .. math::

            J(\omega) = \frac{2 \lambda \gamma \omega}{\gamma^{2}+\omega^{2}}

        (see Eq. 15 in DOI: 10.1103/PhysRevResearch.5.013181)

        Parameters
        ----------
        w: float or array
            Energy of the mode.

        Returns
        -------
        The spectral density of the mode with energy w.
        """

        return 2 * self.lam * self.gamma * w / (self.gamma**2 + w**2)

    def correlation_function(
        self, t: float | ArrayLike, Nk: int = 15000, **kwargs
    ):
        """
        Here we determine the correlation function by summing a large number
        of exponents, as the numerical integration is noisy for this spectral
        density.

        Parameters
        ----------
        t : np.array or float
            The time at which to evaluate the correlation function
        Nk : int, default 15000
            The number of exponents to use
        """

        ck_real, vk_real, ck_imag, vk_imag = self._matsubara_params(Nk)

        def C(c, v):
            return np.sum([ck * np.exp(-np.array(vk * t))
                           for ck, vk in zip(c, v)], axis=0)
        return C(ck_real, vk_real) + 1j * C(ck_imag, vk_imag)

    def power_spectrum(self, w: float | ArrayLike, dt: float = 1e-5):
        return super().power_spectrum(w, dt)

    def _matsubara_approx(self, Nk, combine=True):
        lists = self._matsubara_params(Nk)
        result = ExponentialBosonicEnvironment(
            *lists, T=self.T, combine=combine)
        # TODO what to do with the terminator?
        # TODO tag stuff
        return result

    def _matsubara_params(self, Nk):
        """ Calculate the Matsubara coefficients and frequencies. """
        ck_real = [self.lam * self.gamma / np.tan(self.gamma / (2 * self.T))]
        ck_real.extend([
            (8 * self.lam * self.gamma * self.T * np.pi * k * self.T /
                ((2 * np.pi * k * self.T)**2 - self.gamma**2))
            for k in range(1, Nk + 1)
        ])
        vk_real = [self.gamma]
        vk_real.extend([2 * np.pi * k * self.T for k in range(1, Nk + 1)])

        ck_imag = [-self.lam * self.gamma]
        vk_imag = [self.gamma]

        return ck_real, vk_real, ck_imag, vk_imag

    def _pade_approx(self, Nk, combine=True):
        eta_p, gamma_p = self._corr(Nk)

        ck_real = [np.real(eta) for eta in eta_p]
        vk_real = [gam for gam in gamma_p]
        # There is only one term in the expansion of the imaginary part of the
        # Drude-Lorentz correlation function.
        ck_imag = [np.imag(eta_p[0])]
        vk_imag = [gamma_p[0]]

        result = ExponentialBosonicEnvironment(
            ck_real, vk_real, ck_imag, vk_imag, T=self.T, combine=combine)
        # TODO what to do with the terminator?
        # TODO tag stuff
        return result

    # --- Pade approx calculation ---

    def _corr(self, Nk):
        beta = 1. / self.T
        kappa, epsilon = self._kappa_epsilon(Nk)

        eta_p = [self.lam * self.gamma * (self._cot(self.gamma * beta / 2.0) - 1.0j)]
        gamma_p = [self.gamma]

        for ll in range(1, Nk + 1):
            eta_p.append(
                (kappa[ll] / beta) * 4 * self.lam * self.gamma * (epsilon[ll] / beta)
                / ((epsilon[ll]**2 / beta**2) - self.gamma**2)
            )
            gamma_p.append(epsilon[ll] / beta)

        return eta_p, gamma_p

    def _cot(self, x):
        return 1. / np.tan(x)

    def _kappa_epsilon(self, Nk):
        eps = self._calc_eps(Nk)
        chi = self._calc_chi(Nk)

        kappa = [0]
        prefactor = 0.5 * Nk * (2 * (Nk + 1) + 1)
        for j in range(Nk):
            term = prefactor
            for k in range(Nk - 1):
                term *= (
                    (chi[k]**2 - eps[j]**2) /
                    (eps[k]**2 - eps[j]**2 + self._delta(j, k))
                )
            for k in [Nk - 1]:
                term /= (eps[k]**2 - eps[j]**2 + self._delta(j, k))
            kappa.append(term)

        epsilon = [0] + eps

        return kappa, epsilon

    def _delta(self, i, j):
        return 1.0 if i == j else 0.0

    def _calc_eps(self, Nk):
        alpha = np.diag([
            1. / np.sqrt((2 * k + 5) * (2 * k + 3))
            for k in range(2 * Nk - 1)
        ], k=1)
        alpha += alpha.transpose()
        evals = eigvalsh(alpha)
        eps = [-2. / val for val in evals[0: Nk]]
        return eps

    def _calc_chi(self, Nk):
        alpha_p = np.diag([
            1. / np.sqrt((2 * k + 7) * (2 * k + 5))
            for k in range(2 * Nk - 2)
        ], k=1)
        alpha_p += alpha_p.transpose()
        evals = eigvalsh(alpha_p)
        chi = [-2. / val for val in evals[0: Nk - 1]]
        return chi

class UnderDampedEnvironment(BosonicEnvironment):
    """
    Describes an underdamped environment with the following parameters:

    Parameters
    ----------
    T : float
        Bath temperature.

    lam : float
        Coupling strength.

    gamma : float
        Bath spectral density cutoff frequency.

    w0 : float
        Bath spectral density resonance frequency.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment
    """

    def __init__(
        self, T: float, lam: float, gamma: float, w0: float, *, tag=None
    ):
        super().__init__(T, tag=tag)

        self.lam = lam
        self.gamma = gamma
        self.w0 = w0

        self._avail_approximators.update({
            'matsubara': self._matsubara_approx,
        })

    def spectral_density(self, w: float | ArrayLike):
        r"""
        Calculates the underdamped spectral density,

        .. math::
            J(\omega) = \frac{\lambda^{2} \Gamma \omega}{(\omega_{c}^{2}-
            \omega^{2})^{2}+ \Gamma^{2} \omega^{2}}

        (see Eq. 16 in DOI: 10.1103/PhysRevResearch.5.013181)

        Parameters
        ----------
        w: float or array
            Energy of the mode.

        Returns
        -------
        The spectral density of the mode with energy w.
        """

        return self.lam**2 * self.gamma * w / ((w**2 - self.w0**2)**2
                                               + (self.gamma*w)**2)

    def correlation_function(self, t: float | ArrayLike, **kwargs):
        return super().correlation_function(t, **kwargs)

    def power_spectrum(self, w: float | ArrayLike, dt: float = 1e-5):
        return super().power_spectrum(w, dt)

    def _matsubara_approx(self, Nk, combine=True):
        lists = self._matsubara_params(Nk)
        result = ExponentialBosonicEnvironment(
            *lists, T=self.T, combine=combine)
        # TODO what to do with the terminator?
        # TODO tag stuff
        return result

    def _matsubara_params(self, Nk):
        """ Calculate the Matsubara coefficients and frequencies. """
        beta = 1 / self.T
        Om = np.sqrt(self.w0**2 - (self.gamma / 2)**2)
        Gamma = self.gamma / 2

        ck_real = ([
            (self.lam**2 / (4 * Om))
            * (1 / np.tanh(beta * (Om + 1j * Gamma) / 2)),
            (self.lam**2 / (4 * Om))
            * (1 / np.tanh(beta * (Om - 1j * Gamma) / 2)),
        ])

        ck_real.extend([
            (-2 * self.lam**2 * self.gamma / beta) * (2 * np.pi * k / beta)
            / (
                ((Om + 1j * Gamma)**2 + (2 * np.pi * k / beta)**2)
                * ((Om - 1j * Gamma)**2 + (2 * np.pi * k / beta)**2)
            )
            for k in range(1, Nk + 1)
        ])

        vk_real = [-1j * Om + Gamma, 1j * Om + Gamma]
        vk_real.extend([
            2 * np.pi * k * self.T
            for k in range(1, Nk + 1)
        ])

        ck_imag = [
            1j * self.lam**2 / (4 * Om),
            -1j * self.lam**2 / (4 * Om),
        ]

        vk_imag = [-1j * Om + Gamma, 1j * Om + Gamma]

        return ck_real, vk_real, ck_imag, vk_imag


class OhmicEnvironment(BosonicEnvironment):
    """
    Describes Ohmic environments as well as sub- or super-Ohmic environments
    (depending on the choice of the parameter `s`). This class requires the
    `mpmath` module to be installed.

    Parameters
    ----------
    T : float
        Temperature of the bath.

    alpha : float
        Coupling strength.

    wc : float
        Cutoff parameter.

    s : float
        Power of omega in the spectral density.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment
    """

    def __init__(
        self, T: float, alpha: float, wc: float, s: float, *, tag: Any = None
    ):
        super().__init__(T, tag=tag)
        self.alpha = alpha
        self.wc = wc
        self.s = s

        if _mpmath_available is False:
            warnings.warn(
                "The mpmath module is required for some operations on "
                "Ohmic environments, but it is not installed.")

    def spectral_density(self, w: float | ArrayLike):
        r"""
        Calculates the spectral density of an Ohmic Bath,

        .. math::
            J(w) = \alpha \frac{w^{s}}{w_{c}^{1-s}} e^{-\frac{|w|}{w_{c}}}

        Parameters
        ----------
        w : float or :obj:`np.array`
            Energy of the mode.

        Returns
        -------
        The spectral density of the mode with energy w.
        """

        return (
            self.alpha * w ** self.s
            / (self.wc ** (1 - self.s))
            * np.exp(-np.abs(w) / self.wc)
        )

    def correlation_function(self, t: float | ArrayLike, **kwargs):
        r"""
        Calculates the correlation function of an Ohmic bath,

        .. math::
            C(t)= \frac{1}{\pi} \alpha w_{c}^{1-s} \beta^{-(s+1)} \Gamma(s+1)
            \left[ \zeta\left(s+1,\frac{1+\beta w_{c} -i w_{c} t}{\beta w_{c}}
            \right) +\zeta\left(s+1,\frac{1+ i w_{c} t}{\beta w_{c}}\right)
            \right]

        where :math:`\Gamma` is the gamma function, and :math:`\zeta` the
        Riemann zeta function

        Parameters
        ----------
        t : float or :obj:`np.array`
            time.

        Returns
        -------
        The correlation function at time t.
        """

        if self.T != 0:
            corr = (self.alpha * self.wc ** (1 - self.s) / np.pi
                    * mp.gamma(self.s + 1) / self.T ** (-(self.s + 1)))
            z1_u = ((1 + self.wc / self.T - 1j * self.wc * t)
                    / (self.wc / self.T))
            z2_u = (1 + 1j * self.wc * t) / (self.wc / self.T)
            return np.array(
                [corr * (mp.zeta(self.s + 1, u1) + mp.zeta(self.s + 1, u2))
                 for u1, u2 in zip(z1_u, z2_u)],
                dtype=np.cdouble
            )
        else:
            corr = (self.alpha * self.wc ** (self.s+1) / np.pi
                    * mp.gamma(self.s + 1)
                    * (1 + 1j * self.wc * t) ** (-(self.s + 1)))
            return np.array(corr, dtype=np.cdouble)

    def power_spectrum(self, w: float | ArrayLike, dt: float = 1e-5):
        return super().power_spectrum(w, dt)


class CFExponent:
    """
    Represents a single exponent (naively, an excitation mode) within an
    exponential decomposition of the correlation function of a environment.

    Parameters
    ----------
    type : {"R", "I", "RI", "+", "-"} or one of `CFExponent.types`
        The type of exponent.

        "R" and "I" are bosonic exponents that appear in the real and
        imaginary parts of the correlation expansion, respectively.

        "RI" is a combined bosonic exponent that appears in both the real
        and imaginary parts of the correlation expansion. The combined exponent
        has a single ``vk``. The ``ck`` is the coefficient in the real
        expansion and ``ck2`` is the coefficient in the imaginary expansion.

        "+" and "-" are fermionic exponents. These fermionic exponents must
        specify ``sigma_bar_k_offset`` which specifies the amount to add to
        ``k`` (the exponent index within the environment of this exponent) to
        determine the ``k`` of the corresponding exponent with the opposite
        sign (i.e. "-" or "+").

    vk : complex
        The frequency of the exponent of the excitation term.

    ck : complex
        The coefficient of the excitation term.

    ck2 : optional, complex
        For exponents of type "RI" this is the coefficient of the term in the
        imaginary expansion (and ``ck`` is the coefficient in the real
        expansion).

    sigma_bar_k_offset : optional, int
        For exponents of type "+" this gives the offset (within the list of
        exponents within the environment) of the corresponding "-" type
        exponent. For exponents of type "-" it gives the offset of the
        corresponding "+" exponent.

    Attributes
    ----------
    fermionic : bool
        True if the type of the exponent is a Fermionic type (i.e. either
        "+" or "-") and False otherwise.

    All of the parameters are also available as attributes.
    """
    types = enum.Enum("ExponentType", ["R", "I", "RI", "+", "-"])

    def _check_ck2(self, type, ck2):
        if type == self.types["RI"]:
            if ck2 is None:
                raise ValueError("RI exponents require ck2")
        else:
            if ck2 is not None:
                raise ValueError(
                    "Second co-efficient (ck2) should only be specified for"
                    " RI exponents"
                )

    def _check_sigma_bar_k_offset(self, type, offset):
        if type in (self.types["+"], self.types["-"]):
            if offset is None:
                raise ValueError(
                    "+ and - type exponents require sigma_bar_k_offset"
                )
        else:
            if offset is not None:
                raise ValueError(
                    "Offset of sigma bar (sigma_bar_k_offset) should only be"
                    " specified for + and - type exponents"
                )

    def _type_is_fermionic(self, type):
        return type in (self.types["+"], self.types["-"])

    def __init__(
            self, type, ck, vk, ck2=None, sigma_bar_k_offset=None
    ):
        if not isinstance(type, self.types):
            type = self.types[type]
        self._check_ck2(type, ck2)
        self._check_sigma_bar_k_offset(type, sigma_bar_k_offset)

        self.type = type
        self.ck = ck
        self.vk = vk
        self.ck2 = ck2
        self.sigma_bar_k_offset = sigma_bar_k_offset

        self.fermionic = self._type_is_fermionic(type)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} type={self.type.name}"
            f" ck={self.ck!r} vk={self.vk!r} ck2={self.ck2!r}"
            f" sigma_bar_k_offset={self.sigma_bar_k_offset!r}"
            f" fermionic={self.fermionic!r}"
        )

    @property
    def coefficient(self):
        # TODO docstring, fermionic coefficients
        coeff = 0
        if (self.type == self.types['R'] or self.type == self.types['RI']):
            coeff += self.ck
        if self.type == self.types['I']:
            coeff += 1j * self.ck
        if self.type == self.types['RI']:
            coeff += 1j * self.ck2
        return coeff

    @property
    def exponent(self):
        # TODO docstring, fermionic coefficients
        return self.vk

    def _can_combine(self, other, rtol, atol):
        if type(self) is not type(other):
            return False
        if self.fermionic or other.fermionic:
            return False
        if not np.isclose(self.vk, other.vk, rtol=rtol, atol=atol):
            return False
        return True

    def _combine(self, other):
        # Assumes can combine was checked
        if self.type == self.types['RI'] or self.type != other.type:
            # Result will be RI
            real_part_coefficient = 0
            imag_part_coefficient = 0
            if self.type == self.types['RI'] or self.type == self.types['R']:
                real_part_coefficient += self.ck
            if other.type == self.types['RI'] or other.type == self.types['R']:
                real_part_coefficient += other.ck
            if self.type == self.types['I']:
                imag_part_coefficient += self.ck
            if other.type == self.types['I']:
                imag_part_coefficient += other.ck
            if self.type == self.types['RI']:
                imag_part_coefficient += self.ck2
            if other.type == self.types['RI']:
                imag_part_coefficient += other.ck2
            
            return CFExponent(self.types['RI'], real_part_coefficient,
                              self.vk, imag_part_coefficient)
        else:
            # Both R or both I
            return CFExponent(self.type, self.ck + other.ck, self.vk)


class ExponentialBosonicEnvironment(BosonicEnvironment):
    """
    Bosonic environment that is specified through an exponential decomposition
    of its correlation function. The list of coefficients and exponents in
    the decomposition may either be passed through the four lists `ck_real`,
    `vk_real`, `ck_imag`, `vk_imag`, or as a list of bosonic
    :class:`CFExponent` objects.

    Parameters
    ----------
    ck_real : list of complex
        The coefficients of the expansion terms for the real part of the
        correlation function. The corresponding frequencies are passed as
        vk_real.

    vk_real : list of complex
        The frequencies (exponents) of the expansion terms for the real part of
        the correlation function. The corresponding ceofficients are passed as
        ck_real.

    ck_imag : list of complex
        The coefficients of the expansion terms in the imaginary part of the
        correlation function. The corresponding frequencies are passed as
        vk_imag.

    vk_imag : list of complex
        The frequencies (exponents) of the expansion terms for the imaginary
        part of the correlation function. The corresponding ceofficients are
        passed as ck_imag.

    exponents : list of :class:`CFExponent`
        The expansion coefficients and exponents of both the real and the
        imaginary parts of the correlation function as :class:`CFExponent`
        objects.

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`combine` for details.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.

    T: optional, float
        The temperature of the bath.
    """

    def _check_cks_and_vks(self, ck_real, vk_real, ck_imag, vk_imag):
        lists = [ck_real, vk_real, ck_imag, vk_imag]
        if all(x is None for x in lists):
            return False
        if any(x is None for x in lists):
            raise ValueError(
                "If any of the exponent lists ck_real, vk_real, ck_imag, "
                "vk_imag is provided, all must be provided."
            )
        if len(ck_real) != len(vk_real) or len(ck_imag) != len(vk_imag):
            raise ValueError(
                "The exponent lists ck_real and vk_real, and ck_imag and "
                "vk_imag must be the same length."
            )
        return True

    def __init__(
        self, ck_real=None, vk_real=None, ck_imag=None, vk_imag=None, *,
        exponents=None, combine=True, T=None, tag=None
    ):
        super().__init__(T=T, tag=tag)

        lists_provided = self._check_cks_and_vks(
            ck_real, vk_real, ck_imag, vk_imag)
        if exponents is None and not lists_provided:
            raise ValueError(
                "Either the parameter `exponents` or the parameters "
                "`ck_real`, `vk_real`, `ck_imag`, `vk_imag` must be provided."
            )
        if exponents is not None and any(exp.fermionic for exp in exponents):
            raise ValueError(
                "Fermionic exponent passed to exponential bosonic environment."
            )

        exponents = exponents or []
        if lists_provided:
            exponents.extend(
                CFExponent("R", ck, vk) for ck, vk in zip(ck_real, vk_real)
            )
            exponents.extend(
                CFExponent("I", ck, vk) for ck, vk in zip(ck_imag, vk_imag)
            )

        if combine:
            exponents = self.combine(exponents)
        self.exponents = exponents

    @classmethod
    def combine(cls, exponents, rtol=1e-5, atol=1e-7):
        """
        Group bosonic exponents with the same frequency and return a
        single exponent for each frequency present.

        Parameters
        ----------
        exponents : list of CFExponent
            The list of exponents to combine.

        rtol : float, default 1e-5
            The relative tolerance to use to when comparing frequencies.

        atol : float, default 1e-7
            The absolute tolerance to use to when comparing frequencies.

        Returns
        -------
        list of CFExponent
            The new reduced list of exponents.
        """
        remaining = exponents[:]
        new_exponents = []

        while remaining:
            new_exponent = remaining.pop(0)
            for other_exp in remaining[:]:
                if new_exponent._can_combine(other_exp, rtol, atol):
                    new_exponent = new_exponent._combine(other_exp)
                    remaining.remove(other_exp)
            new_exponents.append(new_exponent)

        return new_exponents

    def correlation_function(self, t: float | ArrayLike, **kwargs):
        """
        Computes the correlation function represented by this exponential
        decomposition.

        Parameters
        ----------
        t: float or obj:`np.array`
            time to compute correlations.

        Returns
        -------
        The correlation function at time t.
        """

        corr = np.zeros_like(t, dtype=complex)
        for exp in self.exponents:
            corr += exp.coefficient * np.exp(-exp.exponent * t)
        return corr

    def power_spectrum(self, w: float | ArrayLike, **kwargs):
        """
        Calculates the power spectrum corresponding to the multi-exponential
        correlation function.

        Parameters
        ----------
        w: float or obj:`np.array`
            Energy of the mode.

        Returns
        -------
        The power spectrum of the mode with energy w.
        """

        S = np.zeros_like(w, dtype=float)
        for exp in self.exponents:
            coeff = exp.coefficient
            S += 2 * np.real(coeff / (exp.vk - 1j * w))
        return S

    def spectral_density(self, w: float | ArrayLike):
        """
        Calculates the spectral density corresponding to the multi-exponential
        correlation function.

        Parameters
        ----------
        w: float or obj:`np.array`
            Energy of the mode.

        Returns
        -------
        The spectral density of the mode with energy w.
        """
        if self.T is None:
            raise ValueError(
                "Bath temperature must be specified for this operation")

        return self.power_spectrum(w) / (n_thermal(w, self.T) + 1) / 2




# --- utility functions ---

def _real_interpolation(fun, xlist, name):
    if callable(fun):
        return fun
    else:
        if xlist is None or len(xlist) != len(fun):
            raise ValueError("A list of x-values with the same length must be "
                             f"provided for the discretized function ({name})")
        return interp1d(
            xlist, fun, kind='cubic', fill_value='extrapolate'
        )

def _complex_interpolation(fun, xlist, name):
    if callable(fun):
        return fun
    else:
        real_interp = _real_interpolation(np.real(fun), xlist, name)
        imag_interp = _real_interpolation(np.imag(fun), xlist, name)
        return lambda x: real_interp(x) + 1j * imag_interp(x)

def _fft(fun, t0=10, dt=1e-5):
    """
    Calculates the Fast Fourier transform of the given function. This
    is an alternative to numerical integration which is often noisy in the 
    settings we are interested on

    Parameters
    ----------
    t0: float or obj:`np.array.`
        Range to use for the fast fourier transform, the range is [-t0,t0].
    dt: float
        The timestep to be used.

    Returns
    -------
    The fourier transform of the correlation function
    """
    # Code adapted from https://stackoverflow.com/a/24077914

    t = np.arange(-t0, t0, dt)
    # Define function
    f = fun(t)

    # Compute Fourier transform by numpy's FFT function
    g = np.fft.fft(f)
    # frequency normalization factor is 2*np.pi/dt
    w = np.fft.fftfreq(f.size) * 2 * np.pi / dt
    # In order to get a discretisation of the continuous Fourier transform
    # we need to multiply g by a phase factor
    g *= dt * np.exp(-1j * w * t0)
    sorted_indices = np.argsort(w)
    zz = interp1d(w[sorted_indices], g[sorted_indices])
    return zz


class FermionicEnvironment(abc.ABC):
    def __init__():
        ...

    @abstractmethod
    def spectral_dnesity(self):
        ...

    @abstractmethod
    def correlation_function_plus(self):
        ...

    @abstractmethod
    def correlation_function_minus(self):
        ...

    @abstractmethod
    def power_spectrum_plus(self):
        ...

    @abstractmethod
    def power_spectrum_plus(self):
        ...

    def exponential_approximation(self):
        raise NotImplementedError
    
    @classmethod
    def from_spectral_density(cls, ...):
        raise NotImplementedError
    
    @classmethod
    def from_correlation_function(cls, ...):
        raise NotImplementedError
        
    @classmethod
    def from_power_spectrum(cls, ...):
        raise NotImplementedError


class LorentzianEnvironment(FermionicEnvironment):
    def exponential_approximation(self):
        ...


class ExponentialFermionicEnvironemnt(FermionicEnvironment):
    ...
