"""
Classes to create baths (reservoirs) for the simulation of open systems,
compatible with the mesolve,bremesolve, HEOMSolver
"""
from time import time
import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from qutip.core import data as _data
from .solver.heom.bofin_baths import (
    UnderDampedBath, DrudeLorentzBath, DrudeLorentzPadeBath, BosonicBath,
    BathExponent)
from .fit_utils import (_run_fit, _gen_summary,
                        _two_column_summary, aaa, filter_poles)


class Reservoir:
    """
    Class that contains the quantities one needs to describe a bath or 
    reservoir

    Parameters
    ----------

    T : float
        Bath temperature.

    x : :obj:`np.array.` (optional)
        The points on which the correlation function is sampled. Does not need
        to be provided if the correlation is passed as function

    J : :obj:`np.array.` or callable
       Rhe function used to create the reservoir

    """

    def __init__(self, T: float, J: callable, x=None):
        self.T = T
        self.set_func(x, J)

    def set_func(self, w, J):
        """
        Sets the function providec. It may be provided either as an
        array of function values or as a python function. For internal reasons,
        it will then be interpolated or discretized as necessary.
        """

        if callable(J):
            self._w = w
            self._func = J
        else:
            if w is None:
                raise ValueError("If the spectra of the bath is provided"
                                 "as a discrete set of points, the points on"
                                 "which it is evaluated must be provided")
            elif len(w) != len(J):
                raise ValueError(f"The spectra o and discrete set of points"
                                 "provided must have the same lenght")
            self._w = w
            self._func_array = J
            real_interp = interp1d(
                w, np.real(J),
                kind='cubic', fill_value='extrapolate')
            imag_interp = interp1d(
                w, np.imag(J),
                kind='cubic', fill_value='extrapolate')

            # Create a complex-valued interpolation function
            self._func = lambda x: real_interp(x) + 1j * imag_interp(x)

    def _fft(self, t0=10, dt=1e-5):
        """
        Calculates the Fast Fourier transform of the correlation function. This
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
        TThe fourier transform of the correlation function
        """
        t = np.arange(-t0, t0, dt)
        # Define function
        f = self.correlation_function(t)

        # Compute Fourier transform by numpy's FFT function
        g = np.fft.fft(f)
        # frequency normalization factor is 2*np.pi/dt
        w = np.fft.fftfreq(f.size)*2*np.pi/dt
        # In order to get a discretisation of the continuous Fourier transform
        # we need to multiply g by a phase factor
        g *= dt*2.5*np.exp(-1j*w*t0)/(np.sqrt(2*np.pi))
        sorted_indices = np.argsort(w)
        zz = interp1d(w[sorted_indices], g[sorted_indices])
        return zz

    def _bose_einstein(self, w):  # TODO: remove
        """
        Calculates the bose einstein distribution for the
        temperature of the bath.

        Parameters
        ----------
        w: float or obj:`np.array.`
            Energy of the mode.

        Returns
        -------
        The population of the mode with energy w.
        """

        if self.T is None:
            raise ValueError(
                "Bath temperature must be specified for this operation")
        if self.T == 0:
            return np.zeros_like(w)

        w = np.array(w, dtype=float)
        result = np.zeros_like(w)
        non_zero = w != 0
        result[non_zero] = 1 / (np.exp(w[non_zero] / self.T) - 1)
        return result


class _BosonicReservoir_fromCF(Reservoir):
    """
    Hiden class that constructs a bosonic reservoir if the correlation function
    and Temperature are provided 

    Parameters
    ----------

    T : float
        Bath temperature.

    x : :obj:`np.array.` (optional)
        The points on which the correlation function is sampled. Does not need
        to be provided if the correlation is passed as function

    C : :obj:`np.array.` or callable
        The correlation function

    """

    def __init__(self, T: float, C: callable, x=None):
        super().__init__(T, C, x)

    def correlation_function(self, t, **kwargs):
        """
        Calculate the correlation function of the bath.

        Returns:
            The correlation function (return type to be determined).
        """
        result = np.zeros_like(t, dtype=complex)
        positive_mask = t > 0
        non_positive_mask = ~positive_mask

        result[positive_mask] = self._func(t[positive_mask])
        result[non_positive_mask] = np.conj(
            self._func(np.abs(t[non_positive_mask])))
        return result

    def spectral_density(self, w):
        """
        Calculate the spectral density of the bath.

        Returns:
            The spectral density (return type to be determined).
        """
        return self.power_spectrum(
            w) / (self._bose_einstein(w) + 1) / 2

    def power_spectrum(self, w, dt=1e-5):
        """
        Calculate the power spectrum of the bath.

        Returns:
            The power spectrum (return type to be determined).
        """
        # Implementation for power_spectrum
        negative = self._fft(w[-1], dt)
        return negative(-w)


class _BosonicReservoir_fromPS(Reservoir):
    """
    Hiden class that constructs a bosonic reservoir if the power spectrum
    and Temperature are provided
    Parameters
    ----------

    T : float
        Bath temperature.

    x : :obj:`np.array.` (optional)
        The points on which the power spectrum is sampled. Does not need
        to be provided if the power spectrum is passed as function

    S: :obj:`np.array.` or callable
        The power spectrum

    """

    def __init__(self, T: float, S: callable, x=None):
        super().__init__(T, S, x)

    def correlation_function(self, t, **kwargs):
        """
        Calculate the correlation function of the bath.
        
        kwargs are the parameters that can be passed to scipy's quad_vec

        Returns:
            The correlation function (return type to be determined).
        """
        def integrand(w, t):
            return self.spectral_density(w) / np.pi * (
                (2 * self._bose_einstein(w) + 1) * np.cos(w * t)
                - 1j * np.sin(w * t)
            )

        result = quad_vec(lambda w: integrand(w, t), 0, np.Inf, **kwargs)
        return result[0]

    def spectral_density(self, w):
        """
        Calculate the spectral density of the bath.

        Returns:
            The spectral density (return type to be determined).
        """
        return self.power_spectrum(
            w) / (self._bose_einstein(w) + 1) / 2

    def power_spectrum(self, w, dt=1e-5):
        """
        Calculate the power spectrum of the bath.

        Returns:
            The power spectrum (return type to be determined).
        """
        return self._func(w)


class _BosonicReservoir_fromSD(Reservoir):
    """
    Hiden class that constructs a bosonic reservoir if the spectral density
    and Temperature are provided

    Parameters
    ----------

    T : float
        Bath temperature.

    x : :obj:`np.array.` (optional)
        The points on which the spectral density is sampled. Does not need
        to be provided if the spectral density is passed as function

    J: :obj:`np.array.` or callable
        The spectral density

    """

    def __init__(self, T: float, J: callable, x=None):
        super().__init__(T, J, x)

    def correlation_function(self, t, **kwargs):
        """
        Calculate the correlation function of the bath.

        Returns:
            The correlation function (return type to be determined).
        """
        def integrand(w, t):
            return self.spectral_density(w) / np.pi * (
                (2 * self._bose_einstein(w) + 1) * np.cos(w * t)
                - 1j * np.sin(w * t)
            )

        result = quad_vec(lambda w: integrand(w, t), 0, np.Inf, **kwargs)
        return result[0]

    def spectral_density(self, w):
        """
        Calculate the spectral density of the bath.

        Returns:
            The spectral density (return type to be determined).
        """
        return self._func(w)

    def power_spectrum(self, w, dt=1e-5):
        """
        Calculate the power spectrum of the bath.

        Returns:
            The power spectrum (return type to be determined).
        """
        w = np.array(w, dtype=float)
        w[w == 0.0] += 1e-6
        if self.T != 0:
            S = (2 * np.sign(w) * self.spectral_density(np.abs(w)) *
                 (self._bose_einstein(w) + 1))
        else:
            S = 2 * np.heaviside(w, 0) * self.spectral_density(w)
        return S


class BosonicReservoir(Reservoir):
    """
    Class that constructs a bosonic reservoir from class methods, temperature 
    and either the spectral density, power spectrum, or correlation function 
    need to be provided
    """
    @classmethod
    def from_SD(self, T, J, w=None):
        return _BosonicReservoir_fromSD(T, J, w)

    @classmethod
    def from_PS(self, T, S, w=None):
        return _BosonicReservoir_fromPS(T, S, w)

    @classmethod
    def from_CF(self, T, C, t=None):
        return _BosonicReservoir_fromCF(T, C, t)


class ExponentialBosonicBath(BosonicBath):
    """
    Hiden class that constructs a bosonic reservoir, from the coefficients and
    exponents

    Parameters
    ----------
    Q : Qobj
        The coupling operator for the bath.

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

    combine : bool, default True
        Whether to combine exponents with the same frequency (and coupling
        operator). See :meth:`combine` for details.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    T: optional, float
        The temperature of the bath.
    """

    def __init__(self, Q, ck_real, vk_real, ck_imag, vk_imag, combine=True,
                 tag=None, T=None):
        super().__init__(Q, ck_real, vk_real, ck_imag, vk_imag, combine,
                         tag, T)

    def _bose_einstein(self, w):
        """
        Calculates the bose einstein distribution for the
        temperature of the bath.

        Parameters
        ----------
        w: float or obj:`np.array.`
            Energy of the mode.

        Returns
        -------
        The population of the mode with energy w.
        """

        if self.T is None:
            raise ValueError(
                "Bath temperature must be specified for this operation")
        if self.T == 0:
            return np.zeros_like(w)

        w = np.array(w, dtype=float)
        result = np.zeros_like(w)
        non_zero = w != 0
        result[non_zero] = 1 / (np.exp(w[non_zero] / self.T) - 1)
        return result

    def correlation_function_approx(self, t):
        """
        Computes the correlation function from the exponents. This is the
        approximation for the correlation function that is used in the HEOM
        construction.

        Parameters
        ----------
        t: float or obj:`np.array.`
            time to compute correlations.

        Returns
        -------
        The correlation function of the bath at time t.
        """

        corr = np.zeros_like(t, dtype=complex)
        for exp in self.exponents:
            if (
                exp.type == BathExponent.types['R'] or
                exp.type == BathExponent.types['RI']
            ):
                corr += exp.ck * np.exp(-exp.vk * t)
            if exp.type == BathExponent.types['I']:
                corr += 1j*exp.ck * np.exp(-exp.vk * t)
            if exp.type == BathExponent.types['RI']:
                corr += 1j*exp.ck2 * np.exp(-exp.vk * t)
        return corr

    def power_spectrum_approx(self, w):
        """
        Calculates the power spectrum from the exponents
        of the bosonic bath.

        Parameters
        ----------
        w: float or obj:`np.array.`
            Energy of the mode.

        Returns
        -------
        The power spectrum of the mode with energy w.
        """

        S = np.zeros_like(w, dtype=float)
        for exp in self.exponents:
            if (
                exp.type == BathExponent.types['R'] or
                exp.type == BathExponent.types['RI']
            ):
                coeff = exp.ck
            if exp.type == BathExponent.types['I']:
                coeff = 1j * exp.ck
            if exp.type == BathExponent.types['RI']:
                coeff += 1j * exp.ck2

            S += 2 * np.real((coeff) / (exp.vk - 1j*w))

        return S

    def spectral_density_approx(self, w):
        """
        Calculates the spectral density from the exponents
        of the bosonic bath.

        Parameters
        ----------
        w: float or obj:`np.array.`
            Energy of the mode.

        Returns
        -------
        The spectral density of the mode with energy w.
        """
        J = self.power_spectrum_approx(w) / (self._bose_einstein(w) + 1) / 2
        return J


class ApproximatedBosonicBath(ExponentialBosonicBath):
    """
    This class allows to construct a reservoir from the correlation function,
    power spectrum or spectral density.
    """
    @classmethod
    def from_sd(self, bath, N, Nk, x, Q):
        """
        Generates a reservoir from the spectral density

        Parameters
        ----------
        bath: obj:`BosonicReservoir`
            The reservoir we want to approximate
        N: int
            The number of modes to use for the fit
        Nk: int
            The number of exponents to use in each mode
        x: obj:`np.ndarray`
            The range on which to perform the fit
        Q: obj:`Qobj`
            The coupling operator to the bath

        Returns
        -------
        A bosonic reservoir
        """
        cls = SpectralFitter(bath.T, Q, x, bath.spectral_density)
        cls.get_fit(N=N, Nk=Nk)
        return cls

    @classmethod
    def from_ps(self, bath, x, tol, max_exponents, Q):
        """
        Generates a reservoir from the power spectrum

        Parameters
        ----------
        bath: obj:`BosonicReservoir`
            The reservoir we want to approximate
        tol: float
            The desired error tolerance
        max_exponents: int
            The maximum number of exponents allowed
        x: obj:`np.ndarray`
            The range on which to perform the fit
        Q: obj:`Qobj`
            The coupling operator to the bath

        Returns
        -------
        A bosonic reservoir
        """
        r, pol, res, zer, _ = aaa(bath.power_spectrum, x,
                                  tol=tol,
                                  max_iter=max_exponents*2)
        new_pols, new_res = filter_poles(pol, res)
        ckAR, ckAI = np.real(-1j*new_res), np.imag(-1j*new_res)
        vkAR, vkAI = np.real(1j*new_pols), np.imag(1j*new_pols)
        cls = ExponentialBosonicBath(
            Q=Q, ck_real=ckAR, vk_real=vkAR + 1j * vkAI, ck_imag=ckAI,
            vk_imag=vkAR + 1j * vkAI, T=bath.T)
        return cls

    @classmethod
    def from_cf(self, Q, x, bath, Nr, Ni, full_ansatz=False):
        """
        Generates a reservoir from the correlation function

        Parameters
        ----------
        bath: obj:`BosonicReservoir`
            The reservoir we want to approximate
        Nr: int
            The number of modes to use for the fit of the real part 
        Ni: int
            The number of modes to use for the fit of the real part 
        x: obj:`np.ndarray`
            The range on which to perform the fit
        Q: obj:`Qobj`
            The coupling operator to the bath
        full_ansatz: bool
            Whether to use a fit of the imaginary and real parts that is 
            complex
        Returns
        -------
        A bosonic reservoir
        """
        cls = CorrelationFitter(Q, bath.T, x, bath.correlation_function)
        cls.get_fit(
            Nr=Nr,
            Ni=Ni,
            full_ansatz=full_ansatz)
        return cls

    @classmethod
    def from_pade_OD(self, Q, T, lam, gamma, Nk, combine=True, tag=None):

        eta_p, gamma_p = self._corr(lam=lam, gamma=gamma, T=T, Nk=Nk)

        ck_real = [np.real(eta) for eta in eta_p]
        vk_real = [gam for gam in gamma_p]

        ck_imag = [np.imag(eta_p[0])]
        vk_imag = [gamma_p[0]]

        cls = _drudepade(
            Q, ck_real, vk_real, ck_imag, vk_imag,
            combine=combine, tag=tag, T=T
        )
        return cls

    @classmethod
    def from_matsubara_UD(
            self, Q, T, lam, gamma, w0, Nk, combine=True, tag=None):

        return _underdamped(
            Q=Q, T=T, lam=lam, gamma=gamma, w0=w0, Nk=Nk,
            combine=combine, tag=tag)

    @classmethod
    def from_matsubara_OD(self, lam, gamma, Nk, Q, T, combine=True, tag=None):
        return _drude(
            lam=lam, gamma=gamma, T=T, Q=Q, Nk=Nk, combine=combine, tag=tag)


class _drudepade(DrudeLorentzPadeBath, ExponentialBosonicBath):
    """
    Hidden class to have DrudeLorentzPadeBath inherit from 
    ExponentialBosonicBath
    """
    def __init__(
            self, Q, ck_real, vk_real, ck_imag, vk_imag, T, combine=True,
            tag=None):
        super().__init__(Q, ck_real, vk_real, ck_imag, vk_imag,
                         combine=combine, tag=tag, T=T)


class _drude(DrudeLorentzBath, ExponentialBosonicBath):
    """
    Hidden class to have DrudeLorentzBath inherit from
    ExponentialBosonicBath
    """
    def __init__(self, Q, lam, gamma, T, Nk, combine=True, tag=None):
        super().__init__(Q=Q, lam=lam, gamma=gamma, T=T, Nk=Nk,
                         combine=combine, tag=tag)


class _underdamped(UnderDampedBath, ExponentialBosonicBath):
    """
    Hidden class to have UnderDampedBath inherit from
    ExponentialBosonicBath
    """
    def __init__(self, Q, lam, gamma, w0, T, Nk, combine=True, tag=None):
        super().__init__(Q=Q, lam=lam, gamma=gamma, w0=w0, T=T, Nk=Nk,
                         combine=combine, tag=tag)


class SpectralFitter(ExponentialBosonicBath):
    """
    A helper class for constructing a Bosonic bath from a fit of the spectral
    density with a sum of underdamped modes.

    Parameters
    ----------
    Q : :obj:`.Qobj`
        Operator describing the coupling between system and bath.

    T : float
        Bath temperature.

    w : :obj:`np.array.`
        The range on which to perform the fit, it is recommended that it covers
        at least twice the cutoff frequency of the desired spectral density.

    J : :obj:`np.array.` or callable
        The spectral density to be fitted as an array or function.
    """

    def __init__(self, T, Q, w, J):
        self.Q = Q
        self.T = T
        self.fitinfo = None
        self.set_spectral_density(w, J)

    def set_spectral_density(self, w, J):
        """
        Sets the spectral density to be fitted. It may be provided either as an
        array of function values or as a python function. For internal reasons,
        it will then be interpolated or discretized as necessary.
        """

        if callable(J):
            self._w = w
            self._J_array = J(w)
            self._J_fun = J
        else:
            self._w = w
            self._J_array = J
            self._J_fun = InterpolatedUnivariateSpline(w, J)

    @classmethod
    def _meier_tannor_SD(cls, w, a, b, c):
        r"""
        Underdamped spectral density used for fitting in Meier-Tannor form
        (see Eq. 38 in the BoFiN paper, DOI: 10.1103/PhysRevResearch.5.013181)
        or the get_fit method.

        Parameters
        ----------
        w : :obj:`np.array.`
            The frequency of the spectral density
        a : :obj:`np.array.`
            Array of coupling constants ($\alpha_i^2$)
        b : :obj:`np.array.`
            Array of cutoff parameters ($\Gamma'_i$)
        c : :obj:`np.array.`
            Array of resonant frequencies ($\Omega_i$)
        """

        return sum((2 * ai * bi * w
                    / ((w + ci) ** 2 + bi ** 2)
                    / ((w - ci) ** 2 + bi ** 2))
                   for ai, bi, ci in zip(a, b, c))

    def get_fit(
        self,
        N=None,
        Nk=1,
        final_rmse=5e-6,
        lower=None,
        upper=None,
        sigma=None,
        guesses=None,
    ):
        r"""
        Provides a fit to the spectral density with N underdamped oscillator
        baths. N can be determined automatically based on reducing the
        normalized root mean squared error below a certain threshold.

        Parameters
        ----------
        N : optional, int
            Number of underdamped oscillators to use.
            If set to None, it is determined automatically.
        Nk : optional, int
            Number of exponential terms used to approximate the bath
            correlation functions, defaults to 1. To approximate the
            correlation function the number of exponents grow as the
            Desired normalized root mean squared error. Defaults to
            :math:`5\times10^{-6}`. Only used if N is set to None.
            Desired normalized root mean squared error. Defaults to
            Lower bounds on the parameters for the fit. A list of size 3,
            containing the lower bounds for :math:`a_i` (coupling constants),
            :math:`b_i` (cutoff frequencies) and :math:`c_i`
            (resonant frequencies) in the following fit function:

            .. math::
                J(\omega) = \sum_{i=1}^{k} \frac{2 a_{i} b_{i} \omega
                }{\left(\left( \omega + c_{i}\right)^{2} + b_{i}^{2}\right)
                \left(\left( \omega - c_{i}\right)^{2} + b_{i}^{2} \right)}

            The lower bounds are considered to be the same for all N modes.
            For example,

            lower=[0,-1,2]

            would bound the coupling to be bigger than 0, the cutoff frequency
            to be higher than 1, and the central frequency to be bigger than 2

        upper : list
            Upper bounds on the parameters for the fit, the structure is the
            same as the lower keyword.
        sigma : float
            Uncertainty in the data considered for the fit, all data points are
            considered to have the same uncertainty.
        guesses : list
            Initial guesses for the parameters. Same structure as lower and
            upper.

        Note: If one of lower, upper, sigma, guesses is None, all are discarded

        Returns
        -------
        1. A Bosonic Bath created with the fit parameters for the original
          spectral density function (that was provided or interpolated)
        2. A dictionary containing the following information about the fit:
            * fit_time:
                The time the fit took in seconds.
            * rsme:
                Normalized mean squared error obtained in the fit.
            * N:
                The number of terms used for the fit.
            * params:
                The fitted parameters (3N parameters), it contains three lists
                one for each parameter, each list containing N terms.
            * Nk:
                The number of exponents used to construct the bosonic bath.
            * summary:
                A string that summarizes the information of the fit.
        """

        start = time()
        rmse, params = _run_fit(
            SpectralFitter._meier_tannor_SD, self._J_array, self._w,
            final_rmse, default_guess_scenario="Spectral Density", N=N,
            sigma=sigma, guesses=guesses, lower=lower, upper=upper)
        end = time()

        fit_time = end - start
        spec_n = len(params[0])
        self._generate_bath(params, Nk)
        summary = _gen_summary(
            fit_time, rmse, N, "The Spectral Density", params)
        fitInfo = {
            "fit_time": fit_time, "rmse": rmse, "N": spec_n, "params": params,
            "Nk": Nk, "summary": summary}
        self.fitinfo = fitInfo

    def _generate_bath(self, params, Nk):
        """
        Obtains the bath exponents from the list of fit parameters. Some
        transformations are done, to reverse the ones in the UnderDampedBath.
        They are done to change the spectral density from eq. 38 to eq. 16
        of the BoFiN paper and vice-versa.

        Parameters
        ----------
        params: list
            The parameters obtained from the fit.

        Returns
        -------
            A Bosonic Bath created with the fit parameters.
        """

        lam, gamma, w0 = params
        w0 = np.array(
            [
                np.sqrt((w0[i] + 0j) ** 2 + (gamma[i] + 0j / 2) ** 2)
                for i in range(len(w0))
            ]
        )
        lam = np.sqrt(
            lam + 0j
        )
        # both w0, and lam modifications are needed to input the
        # right value of the fit into the Underdamped bath
        ckAR = []
        vkAR = []
        ckAI = []
        vkAI = []

        for lamt, Gamma, Om in zip(lam, gamma, w0):
            coeffs = UnderDampedBath._matsubara_params(
                lamt, 2 * Gamma, Om + 0j, self.T, Nk)
            ckAR.extend(coeffs[0])
            vkAR.extend(coeffs[1])
            ckAI.extend(coeffs[2])
            vkAI.extend(coeffs[3])

        super().__init__(self.Q, ckAR, vkAR, ckAI, vkAI, T=self.T)


class CorrelationFitter(ExponentialBosonicBath):
    """
    A helper class for constructing a Bosonic bath from a fit of the
    correlation function with exponential terms.

    Parameters
    ----------
    Q : :obj:`.Qobj`
        Operator describing the coupling between system and bath.
    T : float
        Temperature of the bath.
    t : :obj:`np.array.`
        The range which to perform the fit.
    C : :obj:`np.array.` or callable
        The correlation function to be fitted as an array or function.
    """

    def __init__(self, Q, T, t, C):
        self.Q = Q
        self.T = T
        self.fitinfo = None
        self.set_correlation_function(t, C)

    def set_correlation_function(self, t, C):
        """
        This function creates a discretized version of the correlation function
        if the correlation function is provided, and a function if
        an array is provided.

        The array is needed to run the least squares algorithm, while the
        the function is used to assign a correlation function to the bosonic
        bath object.
        """
        if callable(C):
            self._t = t
            self._C_array = C(t)
            self._C_fun = C
        else:
            self._t = t
            self._C_array = C
            _C_fun_r = InterpolatedUnivariateSpline(t, np.real(C))
            _C_fun_i = InterpolatedUnivariateSpline(t, np.imag(C))
            self._C_fun = lambda t: _C_fun_r(t) + 1j * _C_fun_i(t)

    def _corr_approx(self, t, a, b, c, d=0):
        r"""
        This is the form of the correlation function to be used for fitting.

        Parameters
        ----------
        t : :obj:`np.array.` or float
            The times at which to evaluates the correlation function.
        a : list or :obj:`np.array.`
            A list describing the  real part amplitude of the correlation
            approximation.
        b : list or :obj:`np.array.`
            A list describing the decay of the correlation approximation.
        c : list or :obj:`np.array.`
            A list describing the oscillations of the correlation
            approximation.
        d:  A list describing the imaginary part amplitude of the correlation
            approximation, only used if the user selects if the full_ansatz
            flag from get_fit is True.
        """

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        d = np.array(d)
        if (d == 0).all():
            d = np.zeros(a.shape)

        return np.sum(
            (a[:, None]+1j*d[:, None]) * np.exp(b[:, None] * t[None, :]) *
            np.exp(1j*c[:, None] * t[None, :]),
            axis=0,
        )

    def get_fit(
        self,
        Nr=None,
        Ni=None,
        final_rmse=2e-5,
        lower=None,
        upper=None,
        sigma=None,
        guesses=None,
        full_ansatz=False
    ):
        r"""
        Fit the correlation function with Ni exponential terms
        for the imaginary part of the correlation function and Nr for the real.
        If no number of terms is provided, this function determines the number
        of exponents based on reducing the normalized root mean squared
        error below a certain threshold.

        Parameters
        ----------
        Nr : optional, int
            Number of exponents to use for the real part.
            If set to None it is determined automatically.
        Ni : optional, int
            Number of exponents terms to use for the imaginary part.
            If set to None it is found automatically.
        final_rmse : float
            Desired normalized root mean squared error. Only used if Ni or Nr
            are not specified.
        lower : list
            lower bounds on the parameters for the fit. A list of size 4 when
            full_ansatz is True and of size 3 when it is false,each value
            represents the lower bound for each parameter.

            The first and last terms describe the real and imaginary parts of
            the amplitude, the second the decay rate, and the third one the
            oscillation frequency. The lower bounds are considered to be
            the same for all Nr and Ni exponents. for example

            lower=[0,-1,1,1]

            would bound the real part of the amplitude to be bigger than 0,
            the decay rate to be higher than -1, and the oscillation frequency
            to be bigger than 1, and the imaginary part of the amplitude to
            be greater than 1
        upper : list
            upper bounds on the parameters for the fit, the structure is the
            same as the lower keyword.
        sigma : float
            uncertainty in the data considered for the fit, all data points are
            considered to have the same uncertainty.
        guesses : list
            Initial guesses for the parameters. Same structure as lower and
            upper.
        full_ansatz : bool
            Indicates whether to use the function

            .. math::
                C(t)= \sum_{k}a_{k}e^{-b_{k} t}e^{i c_{k} t}

            for the fitting of the correlation function (when False, the
            default value)  this function gives us
            faster fits,usually it is not needed to tweek
            guesses, sigma, upper and lower as defaults work for most
            situations.  When set to True one uses the function

            .. math::
                C(t)= \sum_{k}(a_{k}+i d_{k})e^{-b_{k} t}e^{i c_{k} t}

            Unfortunately this gives us significantly slower fits and some
            tunning of the guesses,sigma, upper and lower are usually needed.
            On the other hand, it can lead to better fits with lesser exponents
            specially for anomalous spectral densities such that
            $Im(C(0))\neq 0$. When using this with default values if the fit
            takes too long you should input guesses, lower and upper bounds,
            if you are not sure what to set them to it is useful to use the
            output of fitting with the other option as guesses for the fit.



        Note: If one of lower, upper, sigma, guesses is None, all are discarded

        Returns
        -------
        1. A Bosonic Bath created with the fit parameters from the original
          correlation function (that was provided or interpolated).
        2. A dictionary containing the following information about the fit:
            * Nr :
                The number of terms used to fit the real part of the
                correlation function.
            * Ni :
                The number of terms used to fit the imaginary part of the
                correlation function.
            * fit_time_real :
                The time the fit of the real part of the correlation function
                took in seconds.
            * fit_time_imag :
                The time the fit of the imaginary part of the correlation
                function took in seconds.
            * rsme_real :
                Normalized mean squared error obtained in the fit of the real
                part of the correlation function.
            * rsme_imag :
                Normalized mean squared error obtained in the fit of the
                imaginary part of the correlation function.
            * params_real :
                The fitted parameters (3N parameters) for the real part of the
                correlation function, it contains three lists one for each
                parameter, each list containing N terms.
            * params_imag :
                The fitted parameters (3N parameters) for the imaginary part
                of the correlation function, it contains three lists one for
                each parameter, each list containing N terms.
            * summary :
                A string that summarizes the information about the fit.
            """
        if full_ansatz:
            num_params = 4
        else:
            num_params = 3
        # Fit real part
        start_real = time()
        rmse_real, params_real = _run_fit(
            lambda *args: np.real(self._corr_approx(*args)),
            y=np.real(self._C_array), x=self._t, final_rmse=final_rmse,
            default_guess_scenario="correlation_real", N=Nr, sigma=sigma,
            guesses=guesses, lower=lower, upper=upper, n=num_params)
        end_real = time()

        # Fit imaginary part
        start_imag = time()
        rmse_imag, params_imag = _run_fit(
            lambda *args: np.imag(self._corr_approx(*args)),
            y=np.imag(self._C_array), x=self._t, final_rmse=final_rmse,
            default_guess_scenario="correlation_imag", N=Ni, sigma=sigma,
            guesses=guesses, lower=lower, upper=upper, n=num_params)
        end_imag = time()

        # Calculate Fit Times
        fit_time_real = end_real - start_real
        fit_time_imag = end_imag - start_imag

        # Generate summary
        Nr = len(params_real[0])
        Ni = len(params_imag[0])
        full_summary = _two_column_summary(
            params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
            rmse_imag, rmse_real, n=num_params)

        fitInfo = {"Nr": Nr, "Ni": Ni,
                   "fit_time_real": fit_time_real,
                   "fit_time_imag": fit_time_imag,
                   "rmse_real": rmse_real, "rmse_imag": rmse_imag,
                   "params_real": params_real,
                   "params_imag": params_imag, "summary": full_summary}
        self._generate_bath(params_real, params_imag, n=num_params)
        self.fitinfo = fitInfo

    def _generate_bath(self, params_real, params_imag, n=3):
        """
        Calculate the Matsubara coefficients and frequencies for the
        fitted underdamped oscillators and generate the corresponding bosonic
        bath.

        Parameters
        ----------
        params_real : :obj:`np.array.`
            array of shape (N,3) where N is the number of fitted terms
            for the real part.
        params_imag : np.imag
            array of shape (N,3) where N is the number of fitted terms
            for the imaginary part.

        Returns
        -------
        A bosonic Bath constructed from the fitted exponents.
        """
        if n == 4:
            a, b, c, d = params_real
            a2, b2, c2, d2 = params_imag
        else:
            a, b, c = params_real
            a2, b2, c2 = params_imag
            d = np.zeros(a.shape, dtype=int)
            d2 = np.zeros(a2.shape, dtype=int)

        # the 0.5 is from the cosine
        ckAR = [(x + 1j*y)*0.5 for x, y in zip(a, d)]
        # extend the list with the complex conjugates:
        ckAR.extend(np.conjugate(ckAR))
        vkAR = [-x - 1.0j * y for x, y in zip(b, c)]
        vkAR.extend([-x + 1.0j * y for x, y in zip(b, c)])

        # the 0.5 is from the sine
        ckAI = [-1j*(x + 1j*y)*0.5 for x, y in zip(a2, d2)]

        # extend the list with the complex conjugates:
        ckAI.extend(np.conjugate(ckAI))
        vkAI = [-x - 1.0j * y for x, y in zip(b2, c2)]
        vkAI.extend([-x + 1.0j * y for x, y in zip(b2, c2)])

        super().__init__(self.Q, ckAR, vkAR, ckAI, vkAI, T=self.T)
