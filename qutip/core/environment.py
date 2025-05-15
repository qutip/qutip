"""
Classes that describe environments of open quantum systems
"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['BosonicEnvironment',
           'DrudeLorentzEnvironment',
           'UnderDampedEnvironment',
           'OhmicEnvironment',
           'ExponentialBosonicEnvironment',
           'FermionicEnvironment',
           'LorentzianEnvironment',
           'ExponentialFermionicEnvironment',
           'CFExponent',
           'system_terminator']

import abc
import enum
from time import time
from typing import Any, Callable, Literal, Sequence, overload, Union
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvalsh
from scipy.interpolate import CubicSpline

try:
    from mpmath import mp
    _mpmath_available = True
except ModuleNotFoundError:
    _mpmath_available = False

from ..utilities import (n_thermal, iterated_fit, aaa,
                         prony_methods, espira1, espira2)
from .superoperator import spre, spost
from .qobj import Qobj


class BosonicEnvironment(abc.ABC):
    """
    The bosonic environment of an open quantum system. It is characterized by
    its spectral density and temperature or, equivalently, its power spectrum
    or its two-time auto-correlation function.

    Use one of the classmethods :meth:`from_spectral_density`,
    :meth:`from_power_spectrum` or :meth:`from_correlation_function` to
    construct an environment manually from one of these characteristic
    functions, or use a predefined sub-class such as the
    :class:`DrudeLorentzEnvironment`, the :class:`UnderDampedEnvironment` or
    the :class:`OhmicEnvironment`.

    Bosonic environments offer various ways to approximate the environment with
    a multi-exponential correlation function, which can be used for example in
    the HEOM solver. The approximated environment is represented as a
    :class:`ExponentialBosonicEnvironment`.

    All bosonic environments can be approximated by various fitting methods,
    see the description of :meth:`approximate`.
    Subclasses may offer additional approximation methods such as the
    analytical Matsubara or Pade expansions.

    Parameters
    ----------
    T : optional, float
        The temperature of this environment.
    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(self, T: float = None, tag: Any = None):
        self.T = T
        self.tag = tag

    @abc.abstractmethod
    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        The spectral density of this environment. For negative frequencies,
        a value of zero will be returned. See the Users Guide on
        :ref:`bosonic environments <bosonic environments guide>` for specifics
        on the definitions used by QuTiP.

        If no analytical expression for the spectral density is known, it will
        be derived from the power spectrum. In this case, the temperature of
        this environment must be specified.

        If no analytical expression for the power spectrum is known either, it
        will be derived from the correlation function via a fast fourier
        transform.

        Parameters
        ----------
        w : array_like or float
            The frequencies at which to evaluate the spectral density.
        """

        ...

    @abc.abstractmethod
    def correlation_function(
        self, t: float | ArrayLike, *, eps: float = 1e-10
    ) -> (float | ArrayLike):
        """
        The two-time auto-correlation function of this environment. See the
        Users Guide on :ref:`bosonic environments <bosonic environments guide>`
        for specifics on the definitions used by QuTiP.

        If no analytical expression for the correlation function is known, it
        will be derived from the power spectrum via a fast fourier transform.

        If no analytical expression for the power spectrum is known either, it
        will be derived from the spectral density. In this case, the
        temperature of this environment must be specified.

        Parameters
        ----------
        t : array_like or float
            The times at which to evaluate the correlation function.

        eps : optional, float
            Used in case the power spectrum is derived from the spectral
            density; see the documentation of
            :meth:`BosonicEnvironment.power_spectrum`.
        """

        ...

    @abc.abstractmethod
    def power_spectrum(
        self, w: float | ArrayLike, *, eps: float = 1e-10
    ) -> (float | ArrayLike):
        """
        The power spectrum of this environment. See the Users Guide on
        :ref:`bosonic environments <bosonic environments guide>` for specifics
        on the definitions used by QuTiP.

        If no analytical expression for the power spectrum is known, it will
        be derived from the spectral density. In this case, the temperature of
        this environment must be specified.

        If no analytical expression for the spectral density is known either,
        the power spectrum will instead be derived from the correlation
        function via a fast fourier transform.

        Parameters
        ----------
        w : array_like or float
            The frequencies at which to evaluate the power spectrum.

        eps : optional, float
            To derive the zero-frequency power spectrum from the spectral
            density, the spectral density must be differentiated numerically.
            In that case, this parameter is used as the finite difference in
            the numerical differentiation.
        """

        ...

    # --- user-defined environment creation

    @classmethod
    def from_correlation_function(
        cls,
        C: Callable[[float], complex] | ArrayLike,
        tlist: ArrayLike = None,
        tMax: float = None,
        *,
        T: float = None,
        tag: Any = None,
        args: dict[str, Any] = None,
    ) -> BosonicEnvironment:
        r"""
        Constructs a bosonic environment from the provided correlation
        function. The provided function will only be used for times
        :math:`t \geq 0`. At times :math:`t < 0`, the symmetry relation
        :math:`C(-t) = C(t)^\ast` is enforced.

        Parameters
        ----------
        C : callable or array_like
            The correlation function. Can be provided as a Python function or
            as an array. When using a function, the signature should be

            ``C(t: array_like, **args) -> array_like``

            where ``t`` is time and ``args`` is a dict containing the
            other parameters of the function.

        tlist : optional, array_like
            The times where the correlation function is sampled (if it is
            provided as an array).

        tMax : optional, float
            Specifies that the correlation function is essentially zero outside
            the interval [-tMax, tMax]. Used for numerical integration
            purposes.

        T : optional, float
            Environment temperature. (The spectral density of this environment
            can only be calculated from the correlation function if a
            temperature is provided.)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment.

        args : optional, dict
            Extra arguments for the correlation function ``C``.
        """
        return _BosonicEnvironment_fromCF(C, tlist, tMax, T, tag, args)

    @classmethod
    def from_power_spectrum(
        cls,
        S: Callable[[float], float] | ArrayLike,
        wlist: ArrayLike = None,
        wMax: float = None,
        *,
        T: float = None,
        tag: Any = None,
        args: dict[str, Any] = None,
    ) -> BosonicEnvironment:
        r"""
        Constructs a bosonic environment with the provided power spectrum.

        Parameters
        ----------
        S : callable or array_like
            The power spectrum. Can be provided as a Python function or
            as an array. When using a function, the signature should be

            ``S(w: array_like, **args) -> array_like``

            where ``w`` is the frequency and ``args`` is a dict containing the
            other parameters of the function.

        wlist : optional, array_like
            The frequencies where the power spectrum is sampled (if it is
            provided as an array).

        wMax : optional, float
            Specifies that the power spectrum is essentially zero outside the
            interval [-wMax, wMax]. Used for numerical integration purposes.

        T : optional, float
            Environment temperature. (The spectral density of this environment
            can only be calculated from the powr spectrum if a temperature is
            provided.)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment.

        args : optional, dict
            Extra arguments for the power spectrum ``S``.
        """
        return _BosonicEnvironment_fromPS(S, wlist, wMax, T, tag, args)

    @classmethod
    def from_spectral_density(
        cls,
        J: Callable[[float], float] | ArrayLike,
        wlist: ArrayLike = None,
        wMax: float = None,
        *,
        T: float = None,
        tag: Any = None,
        args: dict[str, Any] = None,
    ) -> BosonicEnvironment:
        r"""
        Constructs a bosonic environment with the provided spectral density.
        The provided function will only be used for frequencies
        :math:`\omega > 0`. At frequencies :math:`\omega \leq 0`, the spectral
        density is zero according to the definition used by QuTiP. See the
        Users Guide on :ref:`bosonic environments <bosonic environments guide>`
        for a note on spectral densities with support at negative frequencies.

        Parameters
        ----------
        J : callable or array_like
            The spectral density. Can be provided as a Python function or
            as an array. When using a function, the signature should be

            ``J(w: array_like, **args) -> array_like``

            where ``w`` is the frequency and ``args`` is a tuple containing the
            other parameters of the function.

        wlist : optional, array_like
            The frequencies where the spectral density is sampled (if it is
            provided as an array).

        wMax : optional, float
            Specifies that the spectral density is essentially zero outside the
            interval [-wMax, wMax]. Used for numerical integration purposes.

        T : optional, float
            Environment temperature. (The correlation function and the power
            spectrum of this environment can only be calculated from the
            spectral density if a temperature is provided.)

        tag : optional, str, tuple or any other object
            An identifier (name) for this environment.

        args : optional, dict
            Extra arguments for the spectral density ``J``.
        """
        return _BosonicEnvironment_fromSD(J, wlist, wMax, T, tag, args)

    # --- spectral density, power spectrum, correlation function conversions

    def _ps_from_sd(self, w, eps, derivative=None):
        # derivative: value of J'(0)
        if self.T is None:
            raise ValueError(
                "The temperature must be specified for this operation.")

        w = np.asarray(w, dtype=float)
        if self.T == 0:
            return 2 * np.heaviside(w, 0) * self.spectral_density(w)

        # at zero frequency, we do numerical differentiation
        # S(0) = 2 J'(0) / beta
        zero_mask = (w == 0)
        nonzero_mask = np.invert(zero_mask)

        S = np.zeros_like(w)
        if derivative is None:
            S[zero_mask] = 2 * self.T * self.spectral_density(eps) / eps
        else:
            S[zero_mask] = 2 * self.T * derivative
        S[nonzero_mask] = (
            2 * np.sign(w[nonzero_mask])
            * self.spectral_density(np.abs(w[nonzero_mask]))
            * (n_thermal(w[nonzero_mask], self.T) + 1)
        )
        return S.item() if w.ndim == 0 else S

    def _sd_from_ps(self, w):
        w = np.asarray(w, dtype=float)
        J = np.zeros_like(w)
        positive_mask = (w > 0)
        power_spectrum = self.power_spectrum(w[positive_mask])

        if self.T is None:
            raise ValueError(
                "The temperature must be specified for this operation.")

        J[positive_mask] = (
            power_spectrum / 2 / (n_thermal(w[positive_mask], self.T) + 1)
        )
        return J.item() if w.ndim == 0 else J

    def _ps_from_cf(self, w, tMax):
        w = np.asarray(w, dtype=float)
        if w.ndim == 0:
            wMax = np.abs(w)
        elif len(w) == 0:
            return np.array([])
        else:
            wMax = max(np.abs(w[0]), np.abs(w[-1]))

        mirrored_result = _fft(self.correlation_function, wMax, tMax=tMax)
        result = np.real(mirrored_result(-w))
        return result.item() if w.ndim == 0 else result

    def _cf_from_ps(self, t, wMax, **ps_kwargs):
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            tMax = np.abs(t)
        elif len(t) == 0:
            return np.array([])
        else:
            tMax = max(np.abs(t[0]), np.abs(t[-1]))

        result_fct = _fft(lambda w: self.power_spectrum(w, **ps_kwargs),
                          tMax, tMax=wMax)
        result = result_fct(t) / (2 * np.pi)
        return result.item() if t.ndim == 0 else result

    # --- fitting

    @overload
    def approximate(
        self,
        method: Literal['cf'],
        tlist: ArrayLike,
        target_rmse: float = 2e-5,
        Nr_max: int = 10,
        Ni_max: int = 10,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        full_ansatz: bool = False,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['ps'],
        wlist: ArrayLike,
        target_rmse: float = 5e-6,
        Nmax: int = 5,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal["sd"],
        wlist: ArrayLike,
        Nk: int = 1,
        target_rmse: float = 5e-6,
        Nmax: int = 10,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['aaa'],
        wlist: ArrayLike,
        tol: float = 1e-13,
        Nmax: int = 10,
        combine: bool = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['prony', 'esprit', 'espira-I', 'espira-II'],
        tlist: ArrayLike,
        separate: bool = False,
        Nr: int = 3,
        Ni: int = 3,
        combine: bool = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @property
    def _approximation_methods(self):
        return {
            "cf": (self._approx_by_cf_fit, "Correlation Function NLSQ"),
            "ps": (self._approx_by_ps_fit, "Power Spectrum NLSQ"),
            "sd": (self._approx_by_sd_fit, "Spectral Density NLSQ"),
            "aaa": (self._approx_by_aaa, "Power spectrum AAA"),
            "prony": (self._approx_by_prony, "Correlation Function Prony"),
            "esprit": (self._approx_by_prony, "Correlation Function ESPRIT"),
            "espira-i": (self._approx_by_prony,
                         "Correlation Function ESPIRA-I"),
            "espira-ii": (self._approx_by_prony,
                          "Correlation Function ESPIRA-II"),
        }

    def approximate(self, method: str, *args, **kwargs):
        """
        Generates a multi-exponential approximation of this environment.
        The available methods are ``"cf"``, ``"ps"``, ``"sd"``, ``"aaa"``,
        ``"prony"``, ``"esprit"``, ``"espira-I"`` and
        ``"espira-II"``. The methods and the parameters required per method are
        documented in the :ref:`Users Guide<environment approximations api>`.
        """
        dispatch = self._approximation_methods

        if method.lower() not in dispatch:
            error_string = (f"Unsupported method: {method}."
                            " The available methods are:\n")
            for key in dispatch:
                error_string += f" - {dispatch[key][1]} ({key})\n"
            error_string += ("If unsure what fitting method to use, you should"
                             " probably use ESPIRA-I, or Power Spectrum NLSQ."
                             " For more information about the fitting methods,"
                             " see the Users Guide.")
            raise ValueError(error_string)

        func = dispatch[method.lower()][0]
        return func(method, *args, **kwargs)

    def _approx_by_cf_fit(
        self, method, tlist, target_rmse=2e-5, Nr_max=10, Ni_max=10,
        guess=None, lower=None, upper=None, sigma=None, maxfev=None,
        full_ansatz=False, combine=True, tag=None
    ):

        # Process arguments
        if tag is None and self.tag is not None:
            tag = (self.tag, f"{method.upper()} Fit")

        if full_ansatz:
            num_params = 4
        else:
            num_params = 3

        if target_rmse is None:
            target_rmse = 0
            Nr_min, Ni_min = Nr_max, Ni_max
        else:
            Nr_min, Ni_min = 1, 1

        clist = self.correlation_function(tlist)
        if guess is None and lower is None and upper is None:
            guess_re, lower_re, upper_re = _default_guess_cfreal(
                tlist, np.real(clist), full_ansatz)
            guess_im, lower_im, upper_im = _default_guess_cfimag(
                np.imag(clist), full_ansatz)
        else:
            guess_re, lower_re, upper_re = guess, lower, upper
            guess_im, lower_im, upper_im = guess, lower, upper

        # Fit real part
        start_real = time()
        rmse_real, params_real = iterated_fit(
            _cf_real_fit_model, num_params, tlist, np.real(clist), target_rmse,
            Nr_min, Nr_max, guess=guess_re, lower=lower_re, upper=upper_re,
            sigma=sigma, maxfev=maxfev
        )
        end_real = time()
        fit_time_real = end_real - start_real

        # Fit imaginary part
        start_imag = time()
        rmse_imag, params_imag = iterated_fit(
            _cf_imag_fit_model, num_params, tlist, np.imag(clist), target_rmse,
            Ni_min, Ni_max, guess=guess_im, lower=lower_im, upper=upper_im,
            sigma=sigma, maxfev=maxfev
        )
        end_imag = time()
        fit_time_imag = end_imag - start_imag

        # Generate summary
        Nr = len(params_real)
        Ni = len(params_imag)
        full_summary = _cf_fit_summary(
            params_real, params_imag, fit_time_real, fit_time_imag,
            Nr, Ni, rmse_real, rmse_imag, n=num_params
        )

        fit_info = {"Nr": Nr, "Ni": Ni, "fit_time_real": fit_time_real,
                    "fit_time_imag": fit_time_imag, "rmse_real": rmse_real,
                    "rmse_imag": rmse_imag, "params_real": params_real,
                    "params_imag": params_imag, "summary": full_summary}

        # Finally, generate environment and return
        ckAR = []
        vkAR = []
        for term in params_real:
            if full_ansatz:
                a, b, c, d = term
            else:
                a, b, c = term
                d = 0
            ckAR.extend([(a + 1j * d) / 2, (a - 1j * d) / 2])
            vkAR.extend([-b - 1j * c, -b + 1j * c])

        ckAI = []
        vkAI = []
        for term in params_imag:
            if full_ansatz:
                a, b, c, d = term
            else:
                a, b, c = term
                d = 0
            ckAI.extend([-1j * (a + 1j * d) / 2, 1j * (a - 1j * d) / 2])
            vkAI.extend([-b - 1j * c, -b + 1j * c])

        approx_env = ExponentialBosonicEnvironment(
            ckAR, vkAR, ckAI, vkAI, combine=combine, T=self.T, tag=tag)
        return approx_env, fit_info

    def _approx_by_sd_fit(
        self, method, wlist, Nk=1, target_rmse=5e-6, Nmax=10,
        guess=None, lower=None, upper=None, sigma=None, maxfev=None,
        combine=True, tag=None
    ):

        # Process arguments
        if tag is None and self.tag is not None:
            tag = (self.tag, f"{method.upper()} Fit")

        if target_rmse is None:
            target_rmse = 0
            Nmin = Nmax
        else:
            Nmin = 1

        jlist = self.spectral_density(wlist)
        if guess is None and lower is None and upper is None:
            guess, lower, upper = _default_guess_sd(wlist, jlist)

        # Fit
        start = time()
        rmse, params = iterated_fit(
            _sd_fit_model, 3, wlist, jlist, target_rmse, Nmin, Nmax,
            guess=guess, lower=lower, upper=upper, sigma=sigma, maxfev=maxfev
        )
        end = time()
        fit_time = end - start

        # Generate summary
        N = len(params)
        summary = _fit_summary(
            fit_time, rmse, N, "the spectral density", params
        )
        fit_info = {
            "N": N, "Nk": Nk, "fit_time": fit_time, "rmse": rmse,
            "params": params, "summary": summary}

        ckAR, vkAR, ckAI, vkAI = [], [], [], []
        # Finally, generate environment and return
        for a, b, c in params:
            lam = np.sqrt(a + 0j)
            gamma = 2 * b
            w0 = np.sqrt(c**2 + b**2)

            env = UnderDampedEnvironment(self.T, lam, gamma, w0)
            coeffs = env._matsubara_params(Nk)
            ckAR.extend(coeffs[0])
            vkAR.extend(coeffs[1])
            ckAI.extend(coeffs[2])
            vkAI.extend(coeffs[3])

        approx_env = ExponentialBosonicEnvironment(
            ckAR, vkAR, ckAI, vkAI, combine=combine, T=self.T, tag=tag)
        return approx_env, fit_info

    def _approx_by_ps_fit(
        self, method, wlist, target_rmse=5e-6, Nmax=5,
        guess=None, lower=None, upper=None, sigma=None, maxfev=None,
        combine=True, tag=None
    ):

        # Process arguments
        if tag is None and self.tag is not None:
            tag = (self.tag, f"{method.upper()} Fit")

        if target_rmse is None:
            target_rmse = 0
            Nmin = Nmax
        else:
            Nmin = 1

        jlist = self.power_spectrum(wlist)
        if guess is None and lower is None and upper is None:
            guess, lower, upper = _default_guess_ps(wlist, jlist)

        # Fit
        start = time()
        rmse, params = iterated_fit(
            _ps_fit_model, 4, wlist, jlist, target_rmse, Nmin, Nmax,
            guess=guess, lower=lower, upper=upper, sigma=sigma, maxfev=maxfev
        )
        end = time()
        fit_time = end - start

        # Generate summary
        N = len(params)
        summary = _fit_summary(
            fit_time, rmse, N, "the power spectrum", params,
            columns=['a', 'b', 'c', 'd']
        )
        fit_info = {
            "N": N, "fit_time": fit_time, "rmse": rmse,
            "params": params, "summary": summary}

        ck, vk = [], []
        # Finally, generate environment and return
        for a, b, c, d in params:
            ck.append(a+1j*b)
            vk.append(c+1j*d)
        ck = np.array(ck)
        vk = np.array(vk)
        ckAR = np.concatenate((ck/2, ck.conj()/2))
        ckAI = np.concatenate((-1j*ck/2, 1j*ck.conj()/2))
        vkAR = np.concatenate((vk, vk.conj()))
        approx_env = ExponentialBosonicEnvironment(
            ckAR, vkAR, ckAI, vkAR, T=self.T, combine=combine, tag=tag)
        return approx_env, fit_info

    def _approx_by_aaa(
        self, method, wlist, tol=1e-13, Nmax=10, combine=True, tag=None
    ):

        if tag is None and self.tag is not None:
            tag = (self.tag, f"{method.upper()} Fit")

        start = time()
        # The *2 is there because half the poles will be filtered out
        result = aaa(self.power_spectrum, wlist,
                     tol=tol,
                     max_iter=Nmax * 2)
        end = time()
        pol = result['poles']
        res = result['residues']
        mask = np.imag(pol) < 0

        new_pols, new_res = pol[mask], res[mask]

        vk = 1j * new_pols
        ck = -1j * new_res
        # Create complex conjugates for both vk and ck
        ckAR = np.concatenate((ck/2, ck.conj()/2))
        ckAI = np.concatenate((-1j*ck/2, 1j*ck.conj()/2))
        vkAR = np.concatenate((vk, vk.conj()))

        cls = ExponentialBosonicEnvironment(
            ck_real=ckAR, vk_real=vkAR, ck_imag=ckAI,
            vk_imag=vkAR, T=self.T, combine=combine, tag=tag)
        # Generate summary
        N = len(vk)
        fit_time = end - start
        params = [(ck.real[i], ck.imag[i], vk[i].real, vk[i].imag)
                  for i in range(len(ck))]
        summary = _fit_summary(
            fit_time, result['rmse'], N, "the power spectrum", params,
            columns=['ckr', 'cki', 'vkr', 'vki']
        )
        fitinfo = {
            "N": N, "fit_time": fit_time, "rmse": result['rmse'],
            "params": params, "summary": summary}
        return cls, fitinfo

    def _approx_by_prony(
        self, method, tlist, separate=False, Nr=3, Ni=None,
        combine=True, tag=None
    ):
        if not separate and Ni is not None:
            raise ValueError("The number of imaginary exponents (Ni) cannot be"
                             " specified if real and imaginary parts are fit"
                             " together (separate=False).")
        if Ni is None:
            Ni = 3  # default value
        if tag is None and self.tag is not None:
            tag = (self.tag, f"{method.upper()} Fit")

        def prony(x, n):
            return prony_methods(method, x, n)

        def phase_to_exponent(phases):
            return -((len(tlist) - 1) / tlist[-1]) * (
                np.log(np.abs(phases)) + 1j * np.angle(phases)
            )

        methods = {"prony": prony, "esprit": prony,
                   "espira-I": espira1, "espira-II": espira2}

        if separate:
            start_real = time()
            rmse_real, params_real = methods[method](
                self.correlation_function(tlist).real, Nr)
            end_real = time()

            start_imag = time()
            rmse_imag, params_imag = methods[method](
                self.correlation_function(tlist).imag, Ni)
            end_imag = time()

            ckAR, phases = params_real.T
            ckAI, phases2 = params_imag.T
            vkAR = phase_to_exponent(phases)
            vkAI = phase_to_exponent(phases2)
            cls = ExponentialBosonicEnvironment(
                ck_real=ckAR, vk_real=vkAR, ck_imag=ckAI,
                vk_imag=vkAI, T=self.T, combine=combine, tag=tag)

            params_real = [(ckAR[i].real, ckAR[i].imag, vkAR[i].real,
                            vkAR[i].imag) for i in range(len(ckAR))]
            params_imag = [(ckAI[i].real, ckAI[i].imag, vkAI[i].real,
                            vkAI[i].imag) for i in range(len(ckAI))]
            fit_time_real = end_real-start_real
            fit_time_imag = end_imag-start_imag
            full_summary = _cf_fit_summary(
                params_real, params_imag, fit_time_real, fit_time_imag,
                Nr, Ni, rmse_real, rmse_imag, n=4)
            fit_info = {
                "Nr": Nr, "Ni": Ni, "fit_time_real": fit_time_real,
                "fit_time_imag": fit_time_imag, "rmse_real": rmse_real,
                "rmse_imag": rmse_imag, "params_real": params_real,
                "params_imag": params_imag, "summary": full_summary,
            }

        else:
            start_real = time()
            rmse_real, params_real = methods[method](
                self.correlation_function(tlist), Nr)
            end_real = time()

            amp, phases = params_real.T
            ck = amp
            vk = phase_to_exponent(phases)
            # Create complex conjugates for both vk and ck
            ckAR = np.concatenate((ck/2, ck.conj()/2))
            ckAI = np.concatenate((-1j*ck/2, 1j*ck.conj()/2))
            vkAR = np.concatenate((vk, vk.conj()))
            cls = ExponentialBosonicEnvironment(
                ck_real=ckAR, vk_real=vkAR, ck_imag=ckAI,
                vk_imag=vkAR, T=self.T, combine=combine, tag=tag)

            params_real = [(ck[i].real, ck[i].imag, vk[i].real,
                            vk[i].imag) for i in range(len(amp))]
            fit_time_real = end_real-start_real
            full_summary = _fit_summary(fit_time_real, rmse_real, Nr,
                                        "Correlation Function", params_real,
                                        columns=['ckr', 'cki', 'vkr', 'vki'])
            fit_info = {
                "N": Nr, "fit_time": fit_time_real, "rmse": rmse_real,
                "params": params_real, "summary": full_summary,
            }

        return cls, fit_info

    def approx_by_cf_fit(self, *args, **kwargs):
        # TODO remove by 5.3
        warnings.warn('The API has changed. Please use approximate("cf", ...)'
                      ' instead of approx_by_cf_fit(...).', FutureWarning)
        return self.approximate("cf", *args, **kwargs)

    def approx_by_sd_fit(self, *args, **kwargs):
        # TODO remove by 5.3
        warnings.warn('The API has changed. Please use approximate("sd", ...)'
                      ' instead of approx_by_sd_fit(...).', FutureWarning)
        return self.approximate("sd", *args, **kwargs)


class _BosonicEnvironment_fromCF(BosonicEnvironment):
    def __init__(self, C, tlist, tMax, T, tag, args):
        super().__init__(T, tag)
        self._cf = _complex_interpolation(
            C, tlist, 'correlation function', args)
        if tlist is not None:
            self.tMax = max(np.abs(tlist[0]), np.abs(tlist[-1]))
        else:
            self.tMax = tMax

    def correlation_function(self, t, **kwargs):
        t = np.asarray(t, dtype=float)
        result = np.zeros_like(t, dtype=complex)
        positive_mask = (t >= 0)
        non_positive_mask = np.invert(positive_mask)

        result[positive_mask] = self._cf(t[positive_mask])
        result[non_positive_mask] = np.conj(
            self._cf(-t[non_positive_mask])
        )
        return result.item() if t.ndim == 0 else result

    def spectral_density(self, w):
        return self._sd_from_ps(w)

    def power_spectrum(self, w, **kwargs):
        if self.tMax is None:
            raise ValueError('The support of the correlation function (tMax) '
                             'must be specified for this operation.')
        return self._ps_from_cf(w, self.tMax)


class _BosonicEnvironment_fromPS(BosonicEnvironment):
    def __init__(self, S, wlist, wMax, T, tag, args):
        super().__init__(T, tag)
        self._ps = _real_interpolation(S, wlist, 'power spectrum', args)
        if wlist is not None:
            self.wMax = max(np.abs(wlist[0]), np.abs(wlist[-1]))
        else:
            self.wMax = wMax

    def correlation_function(self, t, **kwargs):
        if self.wMax is None:
            raise ValueError('The support of the power spectrum (wMax) '
                             'must be specified for this operation.')
        return self._cf_from_ps(t, self.wMax)

    def spectral_density(self, w):
        return self._sd_from_ps(w)

    def power_spectrum(self, w, **kwargs):
        w = np.asarray(w, dtype=float)
        ps = self._ps(w)
        return ps.item() if w.ndim == 0 else self._ps(w)


class _BosonicEnvironment_fromSD(BosonicEnvironment):
    def __init__(self, J, wlist, wMax, T, tag, args):
        super().__init__(T, tag)
        self._sd = _real_interpolation(J, wlist, 'spectral density', args)
        if wlist is not None:
            self.wMax = max(np.abs(wlist[0]), np.abs(wlist[-1]))
        else:
            self.wMax = wMax

    def correlation_function(self, t, *, eps=1e-10):
        if self.T is None:
            raise ValueError('The temperature must be specified for this '
                             'operation.')
        if self.wMax is None:
            raise ValueError('The support of the spectral density (wMax) '
                             'must be specified for this operation.')
        return self._cf_from_ps(t, self.wMax, eps=eps)

    def spectral_density(self, w):
        w = np.asarray(w, dtype=float)

        result = np.zeros_like(w)
        positive_mask = (w > 0)
        result[positive_mask] = self._sd(w[positive_mask])

        return result.item() if w.ndim == 0 else result

    def power_spectrum(self, w, *, eps=1e-10):
        return self._ps_from_sd(w, eps)


class DrudeLorentzEnvironment(BosonicEnvironment):
    r"""
    Describes a Drude-Lorentz bosonic environment with the spectral density

    .. math::

        J(\omega) = \frac{2 \lambda \gamma \omega}{\gamma^{2}+\omega^{2}}

    (see Eq. 15 in [BoFiN23]_).

    Parameters
    ----------
    T : float
        Environment temperature.

    lam : float
        Coupling strength.

    gamma : float
        Spectral density cutoff frequency.

    Nk : optional, int, defaults to 10
        The number of Pade exponents to be used for the calculation of the
        correlation function.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(
        self, T: float, lam: float, gamma: float, *,
        Nk: int = 10, tag: Any = None
    ):
        super().__init__(T, tag)

        self.lam = lam
        self.gamma = gamma
        self.Nk = Nk

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        Calculates the Drude-Lorentz spectral density.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        w = np.asarray(w, dtype=float)
        result = np.zeros_like(w)

        positive_mask = (w > 0)
        w_mask = w[positive_mask]
        result[positive_mask] = (
            2 * self.lam * self.gamma * w_mask / (self.gamma**2 + w_mask**2)
        )

        return result.item() if w.ndim == 0 else result

    def correlation_function(
        self, t: float | ArrayLike, Nk: int = None, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the two-time auto-correlation function of the Drude-Lorentz
        environment. The calculation is performed by summing a large number of
        exponents of the Pade expansion.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        Nk : int, optional
            The number of exponents to use. If not provided, then the value
            that was provided when the class was instantiated is used.
        """

        if self.T == 0:
            raise ValueError("The Drude-Lorentz correlation function diverges "
                             "at zero temperature.")

        t = np.asarray(t, dtype=float)
        abs_t = np.abs(t)
        Nk = Nk or self.Nk
        ck_real, vk_real, ck_imag, vk_imag = self._pade_params(Nk)

        def C(c, v):
            return np.sum([ck * np.exp(-np.asarray(vk * abs_t))
                           for ck, vk in zip(c, v)], axis=0)
        result = C(ck_real, vk_real) + 1j * C(ck_imag, vk_imag)

        result = np.asarray(result, dtype=complex)
        result[t < 0] = np.conj(result[t < 0])
        return result.item() if t.ndim == 0 else result

    def power_spectrum(
        self, w: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the power spectrum of the Drude-Lorentz environment.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        sd_derivative = 2 * self.lam / self.gamma
        return self._ps_from_sd(w, None, sd_derivative)

    # --- approximation methods

    @overload
    def approximate(
        self,
        method: Literal['matsubara', 'pade'],
        Nk: int,
        combine: bool = True,
        compute_delta: Literal[False] = False,
        tag: Any = None
    ) -> ExponentialBosonicEnvironment: ...

    @overload
    def approximate(
        self,
        method: Literal['matsubara', 'pade'],
        Nk: int,
        combine: bool = True,
        compute_delta: Literal[True] = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, float]: ...

    # region overloads from parent class
    # unfortunately, @overload definitions must be duplicated in subclasses

    @overload
    def approximate(
        self,
        method: Literal['cf'],
        tlist: ArrayLike,
        target_rmse: float = 2e-5,
        Nr_max: int = 10,
        Ni_max: int = 10,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        full_ansatz: bool = False,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['ps'],
        wlist: ArrayLike,
        target_rmse: float = 5e-6,
        Nmax: int = 5,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal["sd"],
        wlist: ArrayLike,
        Nk: int = 1,
        target_rmse: float = 5e-6,
        Nmax: int = 10,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['aaa'],
        wlist: ArrayLike,
        tol: float = 1e-13,
        Nmax: int = 10,
        combine: bool = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['prony', 'esprit', 'espira-I', 'espira-II'],
        tlist: ArrayLike,
        separate: bool = False,
        Nr: int = 3,
        Ni: int = 3,
        combine: bool = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    # endregion

    @property
    def _approximation_methods(self):
        return {
            **super()._approximation_methods,
            "matsubara": (self._approx_by_matsubara, "Matsubara Truncation"),
            "pade": (self._approx_by_pade, "Pade Truncation")
        }

    def approximate(self, method: str, *args, **kwargs):
        """
        Generates a multi-exponential approximation of this environment.
        The available methods are ``"matsubara"``, ``"pade"``, ``"cf"``,
        ``"ps"``, ``"sd"``, ``"aaa"``, ``"prony"``, ``"esprit"``,
        ``"espira-I"`` and ``"espira-II"``. The methods and the parameters
        required per method are documented in the
        :ref:`Users Guide<environment approximations api>`.
        """
        return super().approximate(method, *args, **kwargs)

    def _approx_by_matsubara(
        self, method, Nk, combine=True, compute_delta=False, tag=None
    ):
        if self.T == 0:
            raise ValueError("The Drude-Lorentz correlation function diverges "
                             "at zero temperature.")
        if tag is None and self.tag is not None:
            tag = (self.tag, "Matsubara Truncation")

        lists = self._matsubara_params(Nk)
        approx_env = ExponentialBosonicEnvironment(
            *lists, T=self.T, combine=combine, tag=tag)

        if not compute_delta:
            return approx_env

        delta = 2 * self.lam * self.T / self.gamma - 1j * self.lam
        for exp in approx_env.exponents:
            delta -= exp.coefficient / exp.exponent

        return approx_env, delta

    def _approx_by_pade(
        self, method, Nk, combine=True, compute_delta=False, tag=None
    ):
        if self.T == 0:
            raise ValueError("The Drude-Lorentz correlation function diverges "
                             "at zero temperature.")
        if tag is None and self.tag is not None:
            tag = (self.tag, "Pade Truncation")

        ck_real, vk_real, ck_imag, vk_imag = self._pade_params(Nk)
        approx_env = ExponentialBosonicEnvironment(
            ck_real, vk_real, ck_imag, vk_imag,
            T=self.T, combine=combine, tag=tag
        )

        if not compute_delta:
            return approx_env

        delta = 2 * self.lam * self.T / self.gamma - 1j * self.lam
        for exp in approx_env.exponents:
            delta -= exp.coefficient / exp.exponent

        return approx_env, delta

    def _pade_params(self, Nk):
        ck_real, vk_real = self._corr(Nk)

        # There is only one term in the expansion of the imaginary part of the
        # Drude-Lorentz correlation function.
        ck_imag = [-self.lam * self.gamma]
        vk_imag = [self.gamma]

        return ck_real, vk_real, ck_imag, vk_imag

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

    # --- Pade approx calculation ---

    def _corr(self, Nk):
        kappa, epsilon = self._kappa_epsilon(Nk)

        eta_p = [self.lam * self.gamma *
                 self._cot(self.gamma / (2 * self.T))]
        gamma_p = [self.gamma]

        for ll in range(1, Nk + 1):
            eta_p.append(
                (kappa[ll] * self.T) * 4 * self.lam *
                self.gamma * (epsilon[ll] * self.T)
                / ((epsilon[ll]**2 * self.T**2) - self.gamma**2)
            )
            gamma_p.append(epsilon[ll] * self.T)

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

    def approx_by_matsubara(self, *args, **kwargs):
        # TODO remove by 5.3
        warnings.warn(
            'The API has changed. Please use approximate("matsubara", ...)'
            ' instead of approx_by_matsubara(...).', FutureWarning)
        return self.approximate("matsubara", *args, **kwargs)

    def approx_by_pade(self, *args, **kwargs):
        # TODO remove by 5.3
        warnings.warn(
            'The API has changed. Please use approximate("pade", ...)'
            ' instead of approx_by_pade(...).', FutureWarning)
        return self.approximate("pade", *args, **kwargs)


class UnderDampedEnvironment(BosonicEnvironment):
    r"""
    Describes an underdamped environment with the spectral density

    .. math::

        J(\omega) = \frac{\lambda^{2} \Gamma \omega}{(\omega_0^{2}-
        \omega^{2})^{2}+ \Gamma^{2} \omega^{2}}

    (see Eq. 16 in [BoFiN23]_).

    Parameters
    ----------
    T : float
        Environment temperature.

    lam : float
        Coupling strength.

    gamma : float
        Spectral density cutoff frequency.

    w0 : float
        Spectral density resonance frequency.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(
        self, T: float, lam: float, gamma: float, w0: float, *, tag: Any = None
    ):
        super().__init__(T, tag)

        self.lam = lam
        self.gamma = gamma
        self.w0 = w0

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        Calculates the underdamped spectral density.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        w = np.asarray(w, dtype=float)
        result = np.zeros_like(w)

        positive_mask = (w > 0)
        w_mask = w[positive_mask]
        result[positive_mask] = (
            self.lam**2 * self.gamma * w_mask / (
                (w_mask**2 - self.w0**2)**2 + (self.gamma * w_mask)**2
            )
        )

        return result.item() if w.ndim == 0 else result

    def power_spectrum(
        self, w: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the power spectrum of the underdamped environment.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        sd_derivative = self.lam**2 * self.gamma / self.w0**4
        return self._ps_from_sd(w, None, sd_derivative)

    def correlation_function(
        self, t: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the two-time auto-correlation function of the underdamped
        environment.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        """
        # we need an wMax so that spectral density is zero for w>wMax, guess:
        wMax = self.w0 + 25 * self.gamma
        return self._cf_from_ps(t, wMax)

    # --- approximation methods

    @overload
    def approximate(
        self,
        method: Literal['matsubara'],
        Nk: int,
        combine: bool = True,
        compute_delta: Literal[False] = False,
        tag: Any = None
    ) -> ExponentialBosonicEnvironment: ...

    @overload
    def approximate(
        self,
        method: Literal['matsubara'],
        Nk: int,
        combine: bool = True,
        compute_delta: Literal[True] = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, float]: ...

    # region overloads from parent class
    # unfortunately, @overload definitions must be duplicated in subclasses

    @overload
    def approximate(
        self,
        method: Literal['cf'],
        tlist: ArrayLike,
        target_rmse: float = 2e-5,
        Nr_max: int = 10,
        Ni_max: int = 10,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        full_ansatz: bool = False,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['ps'],
        wlist: ArrayLike,
        target_rmse: float = 5e-6,
        Nmax: int = 5,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal["sd"],
        wlist: ArrayLike,
        Nk: int = 1,
        target_rmse: float = 5e-6,
        Nmax: int = 10,
        guess: list[float] = None,
        lower: list[float] = None,
        upper: list[float] = None,
        sigma: float | ArrayLike = None,
        maxfev: int = None,
        combine: bool = True,
        tag: Any = None,
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['aaa'],
        wlist: ArrayLike,
        tol: float = 1e-13,
        Nmax: int = 10,
        combine: bool = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    @overload
    def approximate(
        self,
        method: Literal['prony', 'esprit', 'espira-I', 'espira-II'],
        tlist: ArrayLike,
        separate: bool = False,
        Nr: int = 3,
        Ni: int = 3,
        combine: bool = True,
        tag: Any = None
    ) -> tuple[ExponentialBosonicEnvironment, dict[str, Any]]:
        ...

    # endregion

    @property
    def _approximation_methods(self):
        return {
            **super()._approximation_methods,
            "matsubara": (self._approx_by_matsubara, "Matsubara Truncation")
        }

    def approximate(self, method: str, *args, **kwargs):
        """
        Generates a multi-exponential approximation of this environment.
        The available methods are ``"matsubara"``, ``"cf"``,
        ``"ps"``, ``"sd"``, ``"aaa"``, ``"prony"``, ``"esprit"``,
        ``"espira-I"`` and ``"espira-II"``. The methods and the parameters
        required per method are documented in the
        :ref:`Users Guide<environment approximations api>`.
        """
        return super().approximate(method, *args, **kwargs)

    def _approx_by_matsubara(
        self, method, Nk, combine=True, compute_delta=False, tag=None
    ):
        if tag is None and self.tag is not None:
            tag = (self.tag, "Matsubara Truncation")

        lists = self._matsubara_params(Nk)
        approx_env = ExponentialBosonicEnvironment(
            *lists, T=self.T, combine=combine, tag=tag)

        if not compute_delta:
            return approx_env

        delta = self.gamma * self.lam**2 * self.T / self.w0**4
        for exp in approx_env.exponents:
            delta -= np.real(exp.coefficient / exp.exponent)

        return approx_env, delta

    def _matsubara_params(self, Nk):
        """ Calculate the Matsubara coefficients and frequencies. """

        if Nk > 0 and self.T == 0:
            warnings.warn("The Matsubara expansion cannot be performed at "
                          "zero temperature. Use other approaches such as "
                          "fitting the correlation function.")
            Nk = 0

        Om = np.sqrt(self.w0**2 - (self.gamma / 2)**2)
        Gamma = self.gamma / 2

        z = np.inf if self.T == 0 else (Om + 1j * Gamma) / (2 * self.T)
        # we set the argument of the hyperbolic tangent to infinity if T=0
        ck_real = ([
            (self.lam**2 / (4 * Om)) * (1 / np.tanh(z)),
            (self.lam**2 / (4 * Om)) * (1 / np.tanh(np.conjugate(z))),
        ])

        ck_real.extend([
            (-2 * self.lam**2 * self.gamma * self.T) * (2 * np.pi * k * self.T)
            / (
                ((Om + 1j * Gamma)**2 + (2 * np.pi * k * self.T)**2)
                * ((Om - 1j * Gamma)**2 + (2 * np.pi * k * self.T)**2)
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

    def approx_by_matsubara(self, *args, **kwargs):
        # TODO remove by 5.3
        warnings.warn(
            'The API has changed. Please use approximate("matsubara", ...)'
            ' instead of approx_by_matsubara(...).', FutureWarning)
        return self.approximate("matsubara", *args, **kwargs)


class OhmicEnvironment(BosonicEnvironment):
    r"""
    Describes Ohmic environments as well as sub- or super-Ohmic environments
    (depending on the choice of the parameter `s`). The spectral density is

    .. math::

        J(\omega)
        = \alpha \frac{\omega^s}{\omega_c^{s-1}} e^{-\omega / \omega_c} .

    This class requires the `mpmath` module to be installed.

    Parameters
    ----------
    T : float
        Temperature of the environment.

    alpha : float
        Coupling strength.

    wc : float
        Cutoff parameter.

    s : float
        Power of omega in the spectral density.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(
        self, T: float, alpha: float, wc: float, s: float, *, tag: Any = None
    ):
        super().__init__(T, tag)

        self.alpha = alpha
        self.wc = wc
        self.s = s

        if _mpmath_available is False:
            warnings.warn(
                "The mpmath module is required for some operations on "
                "Ohmic environments, but it is not installed.")

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        r"""
        Calculates the spectral density of the Ohmic environment.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        w = np.asarray(w, dtype=float)
        result = np.zeros_like(w)

        positive_mask = (w > 0)
        w_mask = w[positive_mask]
        result[positive_mask] = (
            self.alpha * w_mask ** self.s
            / (self.wc ** (self.s - 1))
            * np.exp(-np.abs(w_mask) / self.wc)
        )

        return result.item() if w.ndim == 0 else result

    def power_spectrum(
        self, w: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the power spectrum of the Ohmic environment.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """
        if self.s > 1:
            sd_derivative = 0
        elif self.s == 1:
            sd_derivative = self.alpha
        else:
            sd_derivative = np.inf
        return self._ps_from_sd(w, None, sd_derivative)

    def correlation_function(
        self, t: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        r"""
        Calculates the correlation function of an Ohmic environment using the
        formula

        .. math::

            C(t)= \frac{1}{\pi} \alpha w_{c}^{1-s} \beta^{-(s+1)} \Gamma(s+1)
            \left[ \zeta\left(s+1,\frac{1+\beta w_{c} -i w_{c} t}{\beta w_{c}}
            \right) +\zeta\left(s+1,\frac{1+ i w_{c} t}{\beta w_{c}}\right)
            \right] ,

        where :math:`\Gamma` is the gamma function, and :math:`\zeta` the
        Riemann zeta function.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        """
        t = np.asarray(t, dtype=float)
        t_was_array = t.ndim > 0
        if not t_was_array:
            t = np.array([t], dtype=float)

        if self.T != 0:
            corr = (self.alpha * self.wc ** (1 - self.s) / np.pi
                    * mp.gamma(self.s + 1) * self.T ** (self.s + 1))
            z1_u = ((1 + self.wc / self.T - 1j * self.wc * t)
                    / (self.wc / self.T))
            z2_u = (1 + 1j * self.wc * t) / (self.wc / self.T)
            result = np.asarray(
                [corr * (mp.zeta(self.s + 1, u1) + mp.zeta(self.s + 1, u2))
                 for u1, u2 in zip(z1_u, z2_u)],
                dtype=np.cdouble
            )
        else:
            corr = (self.alpha * self.wc**2 / np.pi
                    * mp.gamma(self.s + 1)
                    * (1 + 1j * self.wc * t) ** (-self.s - 1))
            result = np.asarray(corr, dtype=np.cdouble)

        if t_was_array:
            return result
        return result[0]


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

        "+" and "-" are fermionic exponents.

    ck : complex
        The coefficient of the excitation term.

    vk : complex
        The frequency of the exponent of the excitation term.

    ck2 : optional, complex
        For exponents of type "RI" this is the coefficient of the term in the
        imaginary expansion (and ``ck`` is the coefficient in the real
        expansion).

    tag : optional, str, tuple or any other object
        A label for the exponent (often the name of the environment). It
        defaults to None.

    Attributes
    ----------
    fermionic : bool
        True if the type of the exponent is a Fermionic type (i.e. either
        "+" or "-") and False otherwise.

    coefficient : complex
        The coefficient of this excitation term in the total correlation
        function (including real and imaginary part).

    exponent : complex
        The frequency of the exponent of the excitation term. (Alias for `vk`.)

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

    def _type_is_fermionic(self, type):
        return type in (self.types["+"], self.types["-"])

    def __init__(
            self, type: str | CFExponent.ExponentType,
            ck: complex, vk: complex, ck2: complex = None, tag: Any = None
    ):
        if not isinstance(type, self.types):
            type = self.types[type]
        self._check_ck2(type, ck2)

        self.type = type
        self.ck = ck
        self.vk = vk
        self.ck2 = ck2

        self.tag = tag
        self.fermionic = self._type_is_fermionic(type)

    def rescale(self, alpha: float) -> CFExponent:
        """Rescale the coefficient of the exponent by a factor of alpha."""
        ck_new = self.ck * alpha
        if self.type == self.types["RI"]:
            ck2_new = self.ck2 * alpha
        else:
            ck2_new = None
        return CFExponent(
            type=self.type, ck=ck_new, vk=self.vk, ck2=ck2_new, tag=self.tag
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} type={self.type.name}"
            f" ck={self.ck!r} vk={self.vk!r} ck2={self.ck2!r}"
            f" fermionic={self.fermionic!r}"
            f" tag={self.tag!r}>"
        )

    @property
    def coefficient(self) -> complex:
        coeff = 0
        if self.type != self.types['I']:
            coeff += self.ck
        else:
            coeff += 1j * self.ck
        if self.type == self.types['RI']:
            coeff += 1j * self.ck2
        return coeff

    @property
    def exponent(self) -> complex:
        return self.vk

    def _can_combine(self, other, rtol, atol):
        if type(self) is not type(other):
            return False
        if self.fermionic or other.fermionic:
            return False
        if not np.isclose(self.vk, other.vk, rtol=rtol, atol=atol):
            return False
        return True

    def _combine(self, other, **init_kwargs):
        # Assumes can combine was checked
        cls = type(self)

        if self.type == other.type and self.type != self.types['RI']:
            # Both R or both I
            return cls(type=self.type, ck=(self.ck + other.ck),
                       vk=self.vk, tag=self.tag, **init_kwargs)

        # Result will be RI
        real_part_coefficient = 0
        imag_part_coefficient = 0
        for exp in [self, other]:
            if exp.type == self.types['RI'] or exp.type == self.types['R']:
                real_part_coefficient += exp.ck
            if exp.type == self.types['I']:
                imag_part_coefficient += exp.ck
            if exp.type == self.types['RI']:
                imag_part_coefficient += exp.ck2

        return cls(type=self.types['RI'], ck=real_part_coefficient, vk=self.vk,
                   ck2=imag_part_coefficient, tag=self.tag, **init_kwargs)


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
        the correlation function. The corresponding coefficients are passed as
        ck_real.

    ck_imag : list of complex
        The coefficients of the expansion terms in the imaginary part of the
        correlation function. The corresponding frequencies are passed as
        vk_imag.

    vk_imag : list of complex
        The frequencies (exponents) of the expansion terms for the imaginary
        part of the correlation function. The corresponding coefficients are
        passed as ck_imag.

    exponents : list of :class:`CFExponent`
        The expansion coefficients and exponents of both the real and the
        imaginary parts of the correlation function as :class:`CFExponent`
        objects.

    combine : bool, default True
        Whether to combine exponents with the same frequency. See
        :meth:`combine` for details.

    T: optional, float
        The temperature of the environment.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    _make_exponent = CFExponent

    def _check_cks_and_vks(self, ck_real, vk_real, ck_imag, vk_imag):
        # all None: returns False
        # all provided and lengths match: returns True
        # otherwise: raises ValueError
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
        self,
        ck_real: ArrayLike = None, vk_real: ArrayLike = None,
        ck_imag: ArrayLike = None, vk_imag: ArrayLike = None,
        *,
        exponents: Sequence[CFExponent] = None,
        combine: bool = True, T: float = None, tag: Any = None
    ):
        super().__init__(T, tag)

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
            exponents.extend(self._make_exponent("R", ck, vk, tag=tag)
                             for ck, vk in zip(ck_real, vk_real))
            exponents.extend(self._make_exponent("I", ck, vk, tag=tag)
                             for ck, vk in zip(ck_imag, vk_imag))

        if combine:
            exponents = self.combine(exponents)
        self.exponents = exponents

    @classmethod
    def combine(
        cls, exponents: Sequence[CFExponent],
        rtol: float = 1e-5, atol: float = 1e-7
    ) -> Sequence[CFExponent]:
        """
        Group bosonic exponents with the same frequency and return a
        single exponent for each frequency present.

        Parameters
        ----------
        exponents : list of :class:`CFExponent`
            The list of exponents to combine.

        rtol : float, default 1e-5
            The relative tolerance to use to when comparing frequencies.

        atol : float, default 1e-7
            The absolute tolerance to use to when comparing frequencies.

        Returns
        -------
        list of :class:`CFExponent`
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

    def correlation_function(
        self, t: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Computes the correlation function represented by this exponential
        decomposition.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        """

        t = np.asarray(t, dtype=float)
        corr = np.zeros_like(t, dtype=complex)

        for exp in self.exponents:
            corr += exp.coefficient * np.exp(-exp.exponent * np.abs(t))
        corr[t < 0] = np.conj(corr[t < 0])

        return corr.item() if t.ndim == 0 else corr

    def power_spectrum(
        self, w: float | ArrayLike, **kwargs
    ) -> (float | ArrayLike):
        """
        Calculates the power spectrum corresponding to the multi-exponential
        correlation function.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        w = np.asarray(w, dtype=float)
        S = np.zeros_like(w)

        for exp in self.exponents:
            S += 2 * np.real(
                exp.coefficient / (exp.exponent - 1j * w)
            )

        return S.item() if w.ndim == 0 else S

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        Calculates the spectral density corresponding to the multi-exponential
        correlation function.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        return self._sd_from_ps(w)

    def rescale(
        self, alpha: float, tag: Any = None
    ) -> ExponentialBosonicEnvironment:
        """
        Returns a new environment where all exponents are scaled by the factor
        alpha. This corresponds to changing the coupling constant. The spectral
        density, the correlation function and the power spectrum are all scaled
        by the same factor.

        Parameters
        ----------
        alpha : float
            Rescaling factor.

        tag : optional, str, tuple or any other object
            An identifier (name) for the rescaled environment.

        Returns
        -------
        ExponentialBosonicEnvironment
            The new environment with the rescaled exponents.
        """

        tag = tag or self.tag

        return ExponentialBosonicEnvironment(
            exponents=[exp.rescale(alpha) for exp in self.exponents],
            combine=False, T=self.T, tag=tag
        )


def system_terminator(Q: Qobj, delta: float) -> Qobj:
    """
    Constructs the terminator for a given approximation discrepancy.

    Parameters
    ----------
    Q : :class:`Qobj`
        The system coupling operator.

    delta : float
        The approximation discrepancy of approximating an environment with a
        finite number of exponentials, see for example the description of the
        :ref:`Matsubara approximation<matsubara approximations api>`.

    Returns
    -------
    terminator : :class:`Qobj`
        A superoperator acting on the system Hilbert space. Liouvillian term
        representing the contribution to the system-environment dynamics of all
        neglected expansion terms. It should be used by adding it to the system
        Liouvillian (i.e. ``liouvillian(H_sys)``).
    """
    op = 2 * spre(Q) * spost(Q.dag()) - spre(Q.dag() * Q) - spost(Q.dag() * Q)
    return delta * op


# --- utility functions ---

def _real_interpolation(fun, xlist, name, args=None):
    args = args or {}
    if callable(fun):
        return lambda w: fun(w, **args)
    else:
        if xlist is None or len(xlist) != len(fun):
            raise ValueError("A list of x-values with the same length must be "
                             f"provided for the discretized function ({name})")
        return CubicSpline(xlist, fun)


def _complex_interpolation(fun, xlist, name, args=None):
    args = args or {}
    if callable(fun):
        return lambda t: fun(t, **args)
    else:
        real_interp = _real_interpolation(np.real(fun), xlist, name)
        imag_interp = _real_interpolation(np.imag(fun), xlist, name)
        return lambda x: real_interp(x) + 1j * imag_interp(x)


def _fft(f, wMax, tMax):
    r"""
    Calculates the Fast Fourier transform of the given function. We calculate
    Fourier transformations via FFT because numerical integration is often
    noisy in the scenarios we are interested in.

    Given a (mathematical) function `f(t)`, this function approximates its
    Fourier transform

    .. math::

        g(\omega) = \int_{-\infty}^\infty dt\, e^{-i\omega t}\, f(t) .

    The function f is sampled on the interval `[-tMax, tMax]`. The sampling
    discretization is chosen as `dt = pi / (4*wMax)` (Shannon-Nyquist + some
    leeway). However, `dt` is always chosen small enough to have at least 500
    samples on the interval `[-tMax, tMax]`.

    Parameters
    ----------
    wMax: float
        Maximum frequency of interest
    tMax: float
        Support of the function f (i.e., f(t) is essentially zero for
        `|t| > tMax`).

    Returns
    -------
    The fourier transform of the provided function as an interpolated function.
    """
    # Code adapted from https://stackoverflow.com/a/24077914

    numSamples = int(
        max(500, np.ceil(2 * tMax * 4 * wMax / np.pi + 1))
    )
    t, dt = np.linspace(-tMax, tMax, numSamples, retstep=True)
    f_values = f(t)

    # Compute Fourier transform by numpy's FFT function
    g = np.fft.fft(f_values)
    # frequency normalization factor is 2 * np.pi / dt
    w = np.fft.fftfreq(numSamples) * 2 * np.pi / dt
    # In order to get a discretisation of the continuous Fourier transform
    # we need to multiply g by a phase factor
    g *= dt * np.exp(1j * w * tMax)

    return _complex_interpolation(
        np.fft.fftshift(g), np.fft.fftshift(w), 'FFT'
    )


def _cf_real_fit_model(tlist, a, b, c, d=0):
    return np.real((a + 1j * d) * np.exp((b + 1j * c) * np.abs(tlist)))


def _cf_imag_fit_model(tlist, a, b, c, d=0):
    return np.sign(tlist) * np.imag(
        (a + 1j * d) * np.exp((b + 1j * c) * np.abs(tlist))
    )


def _default_guess_cfreal(tlist, clist, full_ansatz):
    corr_abs = np.abs(clist)
    corr_max = np.max(corr_abs)
    tc = 2 / np.max(tlist)

    # Checks if constant array, and assigns zero
    if (clist == clist[0]).all():
        if full_ansatz:
            return [[0] * 4] * 3
        return [[0] * 3] * 3

    if full_ansatz:
        lower = [-100 * corr_max, -np.inf, -np.inf, -100 * corr_max]
        guess = [corr_max, -100 * corr_max, 0, 0]
        upper = [100 * corr_max, 0, np.inf, 100 * corr_max]
    else:
        lower = [-20 * corr_max, -np.inf, 0]
        guess = [corr_max, -tc, 0]
        upper = [20 * corr_max, 0.1, np.inf]

    return guess, lower, upper


def _default_guess_cfimag(clist, full_ansatz):
    corr_max = np.max(np.abs(clist))
    # Checks if constant array, and assigns zero
    if (clist == clist[0]).all():
        if full_ansatz:
            return [[0] * 4] * 3
        return [[0] * 3] * 3

    if full_ansatz:
        lower = [-100 * corr_max, -np.inf, -np.inf, -100 * corr_max]
        guess = [0, -10 * corr_max, 0, 0]
        upper = [100 * corr_max, 0, np.inf, 100 * corr_max]
    else:
        lower = [-20 * corr_max, -np.inf, 0]
        guess = [-corr_max, -10 * corr_max, 1]
        upper = [10 * corr_max, 0, np.inf]

    return guess, lower, upper


def _sd_fit_model(wlist, a, b, c):
    return (
        2 * a * b * wlist / ((wlist + c)**2 + b**2) / ((wlist - c)**2 + b**2)
    )


def _default_guess_sd(wlist, jlist):
    sd_abs = np.abs(jlist)
    sd_max = np.max(sd_abs)
    wc = wlist[np.argmax(sd_abs)]

    if sd_max == 0:
        return [0] * 3

    lower = [-100 * sd_max, 0.1 * wc, 0.1 * wc]
    guess = [sd_max, wc, wc]
    upper = [100 * sd_max, 100 * wc, 100 * wc]

    return guess, lower, upper


def _ps_fit_model(wlist, a, b, c, d):
    return (
        2 * (a*c + b*(d-wlist)) / ((wlist - d)**2 + c**2)
    )


def _default_guess_ps(wlist, jlist):
    sd_abs = np.abs(jlist)
    sd_max = np.max(sd_abs)
    wc = np.abs(wlist[np.argmin(sd_abs)])

    if sd_max == 0:
        return [0] * 4
    lower = [-1.5 * sd_max, -1.5 * sd_max, 0.01 * wc, -np.pi]
    guess = [sd_max, sd_max, wc, -np.pi/2]
    upper = [1.5 * sd_max, 1.5 * sd_max, 10 * wc, np.pi]
    return guess, lower, upper


def _fit_summary(time, rmse, N, label, params,
                 columns=['a', 'b', 'c']):
    # Generates summary of fit by nonlinear least squares
    if len(columns) == 3:
        summary = (f"Result of fitting {label} "
                   f"with {N} terms: \n \n {'Parameters': <10}|"
                   f"{columns[0]: ^10}|{columns[1]: ^10}|{columns[2]: >5} \n ")
        for k in range(N):
            summary += (
                f"{k + 1: <10}|{params[k][0]: ^10.2e}|{params[k][1]:^10.2e}|"
                f"{params[k][2]:>5.2e}\n ")
    elif len(columns) == 4:
        summary = (
            f"Result of fitting {label} "
            f"with {N} terms: \n \n {'Parameters': <10}|"
            f"{columns[0]: ^10}|{columns[1]: ^10}|{columns[2]: ^10}"
            f"|{columns[3]: >5} \n ")
        for k in range(N):
            summary += (
                f"{k + 1: <10}|{params[k][0]: ^10.2e}|{params[k][1]:^10.2e}"
                f"|{params[k][2]:^10.2e}|{params[k][3]:>5.2e}\n ")
    else:
        raise ValueError("Unsupported number of columns")
    summary += (f"\nA RMSE of {rmse: .2e}"
                f" was obtained for the {label}.\n")
    summary += f"The current fit took {time: 2f} seconds."
    return summary


def _cf_fit_summary(
    params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
    rmse_real, rmse_imag, n=3
):
    # Generate nicely formatted summary with two columns for CF fit
    columns = ["ckr", "vkr", "vki"]
    if n == 4:
        columns.append("cki")
    summary_real = _fit_summary(
        fit_time_real, rmse_real, Nr,
        "the real part of\nthe correlation function",
        params_real, columns=columns
    )
    summary_imag = _fit_summary(
        fit_time_imag, rmse_imag, Ni,
        "the imaginary part\nof the correlation function",
        params_imag, columns=columns
    )

    full_summary = "Correlation function fit:\n\n"
    lines_real = summary_real.splitlines()
    lines_imag = summary_imag.splitlines()
    max_lines = max(len(lines_real), len(lines_imag))
    # Fill the shorter string with blank lines
    lines_real = (
        lines_real[:-1]
        + (max_lines - len(lines_real)) * [""] + [lines_real[-1]]
    )
    lines_imag = (
        lines_imag[:-1]
        + (max_lines - len(lines_imag)) * [""] + [lines_imag[-1]]
    )
    # Find the maximum line length in each column
    max_length1 = max(len(line) for line in lines_real)
    max_length2 = max(len(line) for line in lines_imag)

    # Print the strings side by side with a vertical bar separator
    for line1, line2 in zip(lines_real, lines_imag):
        formatted_line1 = f"{line1:<{max_length1}} |"
        formatted_line2 = f"{line2:<{max_length2}}"
        full_summary += formatted_line1 + formatted_line2 + "\n"
    return full_summary


# --- fermionic environments ---

class FermionicEnvironment(abc.ABC):
    r"""
    The fermionic environment of an open quantum system. It is characterized by
    its spectral density, temperature and chemical potential or, equivalently,
    by its power spectra or its two-time auto-correlation functions.

    This class is included as a counterpart to :class:`BosonicEnvironment`, but
    it currently does not support all features that the bosonic environment
    does. In particular, fermionic environments cannot be constructed from
    manually specified spectral densities, power spectra or correlation
    functions. The only types of fermionic environment implemented at this time
    are Lorentzian environments (:class:`LorentzianEnvironment`) and
    environments with multi-exponential correlation functions
    (:class:`ExponentialFermionicEnvironment`).

    Parameters
    ----------
    T : optional, float
        The temperature of this environment.
    mu : optional, float
        The chemical potential of this environment.
    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(self, T: float = None, mu: float = None, tag: Any = None):
        self.T = T
        self.mu = mu
        self.tag = tag

    @abc.abstractmethod
    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        r"""
        The spectral density of this environment. See the Users Guide on
        :ref:`fermionic environments <fermionic environments guide>` for
        specifics on the definitions used by QuTiP.

        Parameters
        ----------
        w : array_like or float
            The frequencies at which to evaluate the spectral density.
        """

        ...

    @abc.abstractmethod
    def correlation_function_plus(
        self, t: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        The "+"-branch of the auto-correlation function of this environment.
        See the Users Guide on
        :ref:`fermionic environments <fermionic environments guide>` for
        specifics on the definitions used by QuTiP.

        Parameters
        ----------
        t : array_like or float
            The times at which to evaluate the correlation function.
        """

        ...

    @abc.abstractmethod
    def correlation_function_minus(
        self, t: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        The "-"-branch of the auto-correlation function of this environment.
        See the Users Guide on
        :ref:`fermionic environments <fermionic environments guide>` for
        specifics on the definitions used by QuTiP.

        Parameters
        ----------
        t : array_like or float
            The times at which to evaluate the correlation function.
        """

        ...

    @abc.abstractmethod
    def power_spectrum_plus(self, w: float | ArrayLike) -> (float | ArrayLike):
        r"""
        The "+"-branch of the power spectrum of this environment. See the Users
        Guide on :ref:`fermionic environments <fermionic environments guide>`
        for specifics on the definitions used by QuTiP.

        Parameters
        ----------
        w : array_like or float
            The frequencies at which to evaluate the power spectrum.
        """

        ...

    @abc.abstractmethod
    def power_spectrum_minus(
        self, w: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        The "-"-branch of the power spectrum of this environment. See the Users
        Guide on :ref:`fermionic environments <fermionic environments guide>`
        for specifics on the definitions used by QuTiP.

        Parameters
        ----------
        w : array_like or float
            The frequencies at which to evaluate the power spectrum.
        """

        ...

    # --- user-defined environment creation

    @classmethod
    def from_correlation_functions(cls, **kwargs) -> FermionicEnvironment:
        r"""
        User-defined fermionic environments are currently not implemented.
        """

        raise NotImplementedError("User-defined fermionic environments are "
                                  "currently not implemented.")

    @classmethod
    def from_power_spectra(cls, **kwargs) -> FermionicEnvironment:
        r"""
        User-defined fermionic environments are currently not implemented.
        """

        raise NotImplementedError("User-defined fermionic environments are "
                                  "currently not implemented.")

    @classmethod
    def from_spectral_density(cls, **kwargs) -> FermionicEnvironment:
        r"""
        User-defined fermionic environments are currently not implemented.
        """

        raise NotImplementedError("User-defined fermionic environments are "
                                  "currently not implemented.")


class LorentzianEnvironment(FermionicEnvironment):
    r"""
    Describes a Lorentzian fermionic environment with the spectral density

    .. math::

        J(\omega) = \frac{\gamma W^2}{(\omega - \omega_0)^2 + W^2}.

    (see Eq. 46 in [BoFiN23]_).

    Parameters
    ----------
    T : float
        Environment temperature.

    mu : float
        Environment chemical potential.

    gamma : float
        Coupling strength.

    W : float
        The spectral width of the environment.

    omega0 : optional, float (default equal to ``mu``)
        The resonance frequency of the environment.

    Nk : optional, int, defaults to 10
        The number of Pade exponents to be used for the calculation of the
        correlation functions.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def __init__(
        self, T: float, mu: float, gamma: float, W: float,
        omega0: float = None, *, Nk: int = 10, tag: Any = None
    ):
        super().__init__(T, mu, tag)

        self.gamma = gamma
        self.W = W
        self.Nk = Nk
        if omega0 is None:
            self.omega0 = mu
        else:
            self.omega0 = omega0

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        Calculates the Lorentzian spectral density.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        w = np.asarray(w, dtype=float)
        return self.gamma * self.W**2 / ((w - self.omega0)**2 + self.W**2)

    def correlation_function_plus(
        self, t: float | ArrayLike, Nk: int = None
    ) -> (float | ArrayLike):
        r"""
        Calculates the "+"-branch of the two-time auto-correlation function of
        the Lorentzian environment. The calculation is performed by summing a
        large number of exponents of the Pade expansion.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        Nk : int, optional
            The number of exponents to use. If not provided, then the value
            that was provided when the class was instantiated is used.
        """
        Nk = Nk or self.Nk
        return self._correlation_function(t, Nk, 1)

    def correlation_function_minus(
        self, t: float | ArrayLike, Nk: int = None
    ) -> (float | ArrayLike):
        r"""
        Calculates the "-"-branch of the two-time auto-correlation function of
        the Lorentzian environment. The calculation is performed by summing a
        large number of exponents of the Pade expansion.

        Parameters
        ----------
        t : array_like or float
            The time at which to evaluate the correlation function.
        Nk : int, optional
            The number of exponents to use. If not provided, then the value
            that was provided when the class was instantiated is used.
        """
        Nk = Nk or self.Nk
        return self._correlation_function(t, Nk, -1)

    def _correlation_function(self, t, Nk, sigma):
        if self.T == 0:
            raise NotImplementedError(
                "Calculation of zero-temperature Lorentzian correlation "
                "functions is not implemented yet.")

        t = np.asarray(t, dtype=float)
        abs_t = np.abs(t)
        c, v = self._corr(Nk, sigma)

        result = np.sum([ck * np.exp(-np.asarray(vk * abs_t))
                         for ck, vk in zip(c, v)], axis=0)

        result = np.asarray(result, dtype=complex)
        result[t < 0] = np.conj(result[t < 0])
        return result.item() if t.ndim == 0 else result

    def power_spectrum_plus(self, w: float | ArrayLike) -> (float | ArrayLike):
        r"""
        Calculates the "+"-branch of the power spectrum of the Lorentzian
        environment.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        return self.spectral_density(w) / (np.exp((w - self.mu) / self.T) + 1)

    def power_spectrum_minus(
        self, w: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        Calculates the "-"-branch of the power spectrum of the Lorentzian
        environment.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        return self.spectral_density(w) / (np.exp((self.mu - w) / self.T) + 1)

    def approx_by_matsubara(
        self, Nk: int, tag: Any = None
    ) -> ExponentialFermionicEnvironment:
        """
        Generates an approximation to this environment by truncating its
        Matsubara expansion.

        Parameters
        ----------
        Nk : int
            Number of Matsubara terms to include. In total, the "+" and "-"
            correlation function branches will include `Nk+1` terms each.

        tag : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

        Returns
        -------
        The approximated environment with multi-exponential correlation
        function.
        """
        if self.T == 0:
            raise NotImplementedError(
                "Calculation of zero-temperature Lorentzian correlation "
                "functions is not implemented yet.")

        if tag is None and self.tag is not None:
            tag = (self.tag, "Matsubara Truncation")

        ck_plus, vk_plus = self._matsubara_params(Nk, 1)
        ck_minus, vk_minus = self._matsubara_params(Nk, -1)

        return ExponentialFermionicEnvironment(
            ck_plus, vk_plus, ck_minus, vk_minus, T=self.T, mu=self.mu,
            tag=tag
        )

    def approx_by_pade(
        self, Nk: int, tag: Any = None
    ) -> ExponentialFermionicEnvironment:
        """
        Generates an approximation to this environment by truncating its
        Pade expansion.

        Parameters
        ----------
        Nk : int
            Number of Pade terms to include. In total, the "+" and "-"
            correlation function branches will include `Nk+1` terms each.

        tag : optional, str, tuple or any other object
            An identifier (name) for the approximated environment. If not
            provided, a tag will be generated from the tag of this environment.

        Returns
        -------
        The approximated environment with multi-exponential correlation
        function.
        """
        if self.T == 0:
            raise NotImplementedError(
                "Calculation of zero-temperature Lorentzian correlation "
                "functions is not implemented yet.")

        if tag is None and self.tag is not None:
            tag = (self.tag, "Pade Truncation")

        ck_plus, vk_plus = self._corr(Nk, sigma=1)
        ck_minus, vk_minus = self._corr(Nk, sigma=-1)

        return ExponentialFermionicEnvironment(
            ck_plus, vk_plus, ck_minus, vk_minus, T=self.T, mu=self.mu, tag=tag
        )

    def _matsubara_params(self, Nk, sigma):
        """ Calculate the Matsubara coefficients and frequencies. """
        def f(x):
            return 1 / (np.exp(x / self.T) + 1)

        coeff_list = [(
            self.W * self.gamma / 2 *
            f(sigma * (self.omega0 - self.mu) + 1j * self.W)
        )]
        exp_list = [self.W - sigma * 1j * self.omega0]

        xk_list = [(2 * k - 1) * np.pi * self.T for k in range(1, Nk + 1)]
        for xk in xk_list:
            coeff_list.append(
                1j * self.gamma * self.W**2 * self.T /
                ((sigma * xk - 1j * self.mu + 1j * self.omega0)**2 - self.W**2)
            )
            exp_list.append(
                xk - sigma * 1j * self.mu
            )

        return coeff_list, exp_list

    # --- Pade approx calculation ---

    def _corr(self, Nk, sigma):
        beta = 1 / self.T
        kappa, epsilon = self._kappa_epsilon(Nk)

        def f_approx(x):
            f = 0.5
            for ll in range(1, Nk + 1):
                f = f - 2 * kappa[ll] * x / (x**2 + epsilon[ll]**2)
            return f

        eta_list = [(0.5 * self.gamma * self.W *
                     f_approx(beta * sigma * (self.omega0 - self.mu)
                              + beta * 1j * self.W))]
        gamma_list = [self.W - sigma * 1.0j * self.omega0]

        for ll in range(1, Nk + 1):
            eta_list.append(
                -1.0j * (kappa[ll] / beta) * self.gamma * self.W**2
                / ((self.mu - self.omega0 + sigma * 1j * epsilon[ll] / beta)**2
                   + self.W**2)
            )
            gamma_list.append(epsilon[ll] / beta - sigma * 1.0j * self.mu)

        return eta_list, gamma_list

    def _kappa_epsilon(self, Nk):
        eps = self._calc_eps(Nk)
        chi = self._calc_chi(Nk)

        kappa = [0]
        prefactor = 0.5 * Nk * (2 * (Nk + 1) - 1)
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
            1. / np.sqrt((2 * k + 3) * (2 * k + 1))
            for k in range(2 * Nk - 1)
        ], k=1)
        alpha += alpha.transpose()

        evals = eigvalsh(alpha)
        eps = [-2. / val for val in evals[0: Nk]]
        return eps

    def _calc_chi(self, Nk):
        alpha_p = np.diag([
            1. / np.sqrt((2 * k + 5) * (2 * k + 3))
            for k in range(2 * Nk - 2)
        ], k=1)
        alpha_p += alpha_p.transpose()
        evals = eigvalsh(alpha_p)
        chi = [-2. / val for val in evals[0: Nk - 1]]
        return chi


class ExponentialFermionicEnvironment(FermionicEnvironment):
    """
    Fermionic environment that is specified through an exponential
    decomposition of its correlation function. The list of coefficients and
    exponents in the decomposition may either be passed through the four lists
    `ck_plus`, `vk_plus`, `ck_minus`, `vk_minus`, or as a list of fermionic
    :class:`CFExponent` objects.

    Alternative constructors :meth:`from_plus_exponents` and
    :meth:`from_minus_exponents` are available to compute the "-" exponents
    automatically from the "+" ones, or vice versa.

    Parameters
    ----------
    ck_plus : list of complex
        The coefficients of the expansion terms for the ``+`` part of the
        correlation function. The corresponding frequencies are passed as
        vk_plus.

    vk_plus : list of complex
        The frequencies (exponents) of the expansion terms for the ``+`` part
        of the correlation function. The corresponding coefficients are passed
        as ck_plus.

    ck_minus : list of complex
        The coefficients of the expansion terms for the ``-`` part of the
        correlation function. The corresponding frequencies are passed as
        vk_minus.

    vk_minus : list of complex
        The frequencies (exponents) of the expansion terms for the ``-`` part
        of the correlation function. The corresponding coefficients are passed
        as ck_minus.

    exponents : list of :class:`CFExponent`
        The expansion coefficients and exponents of both parts of the
        correlation function as :class:`CFExponent` objects.

    T: optional, float
        The temperature of the environment.

    mu: optional, float
        The chemical potential of the environment.

    tag : optional, str, tuple or any other object
        An identifier (name) for this environment.
    """

    def _check_cks_and_vks(self, ck_plus, vk_plus, ck_minus, vk_minus):
        # all None: returns False
        # all provided and lengths match: returns True
        # otherwise: raises ValueError
        lists = [ck_plus, vk_plus, ck_minus, vk_minus]
        if all(x is None for x in lists):
            return False
        if any(x is None for x in lists):
            raise ValueError(
                "If any of the exponent lists ck_plus, vk_plus, ck_minus, "
                "vk_minus is provided, all must be provided."
            )
        if len(ck_plus) != len(vk_plus) or len(ck_minus) != len(vk_minus):
            raise ValueError(
                "The exponent lists ck_plus and vk_plus, and ck_minus and "
                "vk_minus must be the same length."
            )
        return True

    def __init__(
        self,
        ck_plus: ArrayLike = None, vk_plus: ArrayLike = None,
        ck_minus: ArrayLike = None, vk_minus: ArrayLike = None,
        *,
        exponents: Sequence[CFExponent] = None,
        T: float = None, mu: float = None, tag: Any = None
    ):
        super().__init__(T, mu, tag)

        lists_provided = self._check_cks_and_vks(
            ck_plus, vk_plus, ck_minus, vk_minus)
        if exponents is None and not lists_provided:
            raise ValueError(
                "Either the parameter `exponents` or the parameters "
                "`ck_plus`, `vk_plus`, `ck_minus`, `vk_minus` must be "
                "provided."
            )
        if (exponents is not None and
                not all(exp.fermionic for exp in exponents)):
            raise ValueError(
                "Bosonic exponent passed to exponential fermionic environment."
            )

        self.exponents = exponents or []
        if lists_provided:
            self.exponents.extend(CFExponent("+", ck, vk, tag=tag)
                                  for ck, vk in zip(ck_plus, vk_plus))
            self.exponents.extend(CFExponent("-", ck, vk, tag=tag)
                                  for ck, vk in zip(ck_minus, vk_minus))

    def spectral_density(self, w: float | ArrayLike) -> (float | ArrayLike):
        """
        Computes the spectral density corresponding to the multi-exponential
        correlation function.

        Parameters
        ----------
        w : array_like or float
            Energy of the mode.
        """

        return self.power_spectrum_minus(w) + self.power_spectrum_plus(w)

    def correlation_function_plus(
        self, t: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        Computes the "+"-branch of the correlation function represented by this
        exponential decomposition.

        Parameters
        ----------
        t : array_like or float
            The times at which to evaluate the correlation function.
        """

        return self._cf(t, CFExponent.types['+'])

    def correlation_function_minus(
        self, t: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        Computes the "-"-branch of the correlation function represented by this
        exponential decomposition.

        Parameters
        ----------
        t : array_like or float
            The times at which to evaluate the correlation function.
        """

        return self._cf(t, CFExponent.types['-'])

    def _cf(self, t, type):
        t = np.asarray(t, dtype=float)
        corr = np.zeros_like(t, dtype=complex)

        for exp in self.exponents:
            if exp.type == type:
                corr += exp.coefficient * np.exp(-exp.exponent * np.abs(t))
        corr[t < 0] = np.conj(corr[t < 0])

        return corr.item() if t.ndim == 0 else corr

    def power_spectrum_plus(
        self, w: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        Calculates the "+"-branch of the power spectrum corresponding to the
        multi-exponential correlation function.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        return self._ps(w, CFExponent.types['+'], 1)

    def power_spectrum_minus(
        self, w: float | ArrayLike
    ) -> (float | ArrayLike):
        r"""
        Calculates the "-"-branch of the power spectrum corresponding to the
        multi-exponential correlation function.

        Parameters
        ----------
        w : array_like or float
            The frequency at which to evaluate the power spectrum.
        """

        return self._ps(w, CFExponent.types['-'], -1)

    def _ps(self, w, type, sigma):
        w = np.asarray(w, dtype=float)
        S = np.zeros_like(w)

        for exp in self.exponents:
            if exp.type == type:
                S += 2 * np.real(
                    exp.coefficient / (exp.exponent + sigma * 1j * w)
                )

        return S.item() if w.ndim == 0 else S

    def rescale(
        self, alpha: float, tag: Any = None
    ) -> ExponentialFermionicEnvironment:
        """
        Returns a new environment where all exponents are scaled by the factor
        alpha. This corresponds to changing the coupling constant. The spectral
        density, the correlation function and the power spectrum are all scaled
        by the same factor.

        Parameters
        ----------
        alpha : float
            Rescaling factor.

        tag : optional, str, tuple or any other object
            An identifier (name) for the rescaled environment.

        Returns
        -------
        ExponentialBosonicEnvironment
            The new environment with the rescaled exponents.
        """

        tag = tag or self.tag

        return ExponentialFermionicEnvironment(
            exponents=[exp.rescale(alpha) for exp in self.exponents],
            T=self.T, mu=self.mu, tag=tag
        )


Environment = Union[BosonicEnvironment, FermionicEnvironment]
