"""
This module provides utilities for fitting bosonic baths through
the correlation function or spectral density, the fit returns a
HEOM bath object see the ``qutip.nonmarkov.bofin_baths``.

The number of modes for the fit can be indicated by the user or
determined by requiring a normalized root mean squared error for
the fit.
"""

import numpy as np
from time import time
try:
    from mpmath import mp
    _mpmath_available = True
except ModuleNotFoundError:
    _mpmath_available = False
from scipy.optimize import curve_fit
from qutip.solver.heom import UnderDampedBath, BosonicBath
from scipy.interpolate import InterpolatedUnivariateSpline


__all__ = ["SpectralFitter", "CorrelationFitter", "OhmicBath"]


class SpectralFitter:
    """
    A helper class for constructing a Bosonic bath from a fit of the spectral
    density with a sum of underdamped modes.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    T : float
        Bath temperature.

    w : np.array
        The range on which to perform the fit

    J : np.array or callable
        The spectral density to be fitted as an array or function
    """

    def __init__(self, T, Q, w, J):
        self.Q = Q
        self.T = T
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

    def _spectral_density_approx(self, w, a, b, c):
        """
        Underdamped spectral density used for fitting in Meier-Tannor form
        (see Eq. 38 in the BoFiN paper, DOI: 10.1103/PhysRevResearch.5.013181)

        Parameters
        ----------
        w : np.array
            The frequency of the spectral density
        a : np.array
            Array of coupling constant
        b : np.array
            Array of cutoff parameters
        c : np.array
            Array of resonant frequencies
        """

        return sum((2 * ai * bi * w
                    / ((w + ci) ** 2 + bi ** 2)
                    / ((w - ci) ** 2 + bi ** 2))
                   for ai, bi, ci in zip(a, b, c))

    def get_fit(
        self,
        N=None,
        Nk=5,
        final_rmse=5e-6,
        lower=None,
        upper=None,
        sigma=None,
        guesses=None,
    ):
        """
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
            correlation functions, per bath. Defaults to 5.
        final_rmse : float
            Desired normalized root mean squared error.
        lower : list
            lower bounds on the parameters for the fit.
        upper : list
            upper bounds on the parameters for the fit
        sigma : float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guesses for the parameters.

        TODO: describe structure of lower, upper, sigma, guesses
        Note: If one of lower, upper, sigma, guesses is None, all are discarded

        Returns
        -------
        * A Bosonic Bath created with the fit parameters with the original
          spectral density function (that was provided or interpolated)
        * A dictionary containing the following information about the fit:
            fit_time : TODO
            ...
        """

        start = time()
        rmse, params = _run_fit(
            self._spectral_density_approx, self._J_array, self._w, final_rmse,
            default_guess_scenario="Spectral Density", N=N,
            sigma=sigma, guesses=guesses, lower=lower, upper=upper)
        end = time()

        fit_time = end - start
        spec_n = len(params[0])
        bath = self._generate_bath(params, Nk)
        bath.spectral_density = self._J_fun
        summary = _gen_summary(
            fit_time, rmse, N, "The Spectral Density", *params)
        fitInfo = {
            "fit_time": fit_time, "rmse": rmse, "N": spec_n, "params": params,
            "Nk": Nk, "summary": summary}
        return bath, fitInfo

    def _generate_bath(self, params, Nk):
        """
        Obtains the bath exponents from the list of fit parameters. Some
        transformations are done, to reverse the ones in the UnderDampedBath.
        They are done to change the spectral density from eq. 38 to eq. 16
        of the BoFiN paper and vice-versa.

        Parameters
        ----------
        params: list
            The parameters obtained from the fit

        Returns
        -------
            A Bosonic Bath created with the fit parameters
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

        bath = BosonicBath(self.Q, ckAR, vkAR, ckAI, vkAI, T=self.T)
        return bath


class CorrelationFitter:
    """
    A helper class for constructing a Bosonic bath from a fit of the
    correlation function with exponential terms.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.
    T : float
        Temperature of the bath
    t : np.array
        The range which to perform the fit
    C : np.array or callable
        The correlation function to be fitted as an array or function
    """

    def __init__(self, Q, T, t, C):
        self.Q = Q
        self.T = T
        self.set_correlation_function(t, C)

    def set_correlation_function(self, t, C):
        """
        This function creates a discretized version of the correlation function
        if the correlation function is provided, and a function if
        an array is provided.

        The array is needed to run the least squares algorithm, while the
        the function is used to assign a correlation function to the bosonic
        bath object
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

    def _corr_approx(self, t, a, b, c):
        """
        This is the form of the correlation function to be used for fitting

        Parameters
        ----------
        t : np.array or float
            Operator describing the coupling between system and bath.
        a : list or np.array
            A list describing the amplitude of the correlation approximation
        b : list or np.array
            A list describing the decay of the correlation approximation
        c : list or np.array
            A list describing the oscillations of the correlation approximation
        """

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        return np.sum(
            a[:, None] * np.exp(b[:, None] * t) * np.exp(1j * c[:, None] * t),
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
    ):
        """
        Fit the correlation function with Ni exponential terms
        for the imaginary part of the correlation function and Nr for the real.
        If no number of terms is provided, this function determines the number
        of exponentials based on reducing the normalized root mean squared
        error below a certain threshold.

        Parameters
        ----------
        Nr : optional, int
            Number of exponential terms to use for the real part.
            If set to None it is determined automatically.
        Ni : optional, int
            Number of exponential terms to use for the imaginary part.
            If set to None it is found automatically.
        final_rmse : float
            Desired normalized root mean squared error .
        lower : list
            lower bounds on the parameters for the fit.
        upper : list
            upper bounds on the parameters for the fit
        sigma : float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guess for the parameters.

        TODO: as in SpectralFitter
        TODO: Returns
        """

        # Fit real part
        start_real = time()
        rmse_real, params_real = _run_fit(
            lambda *args: np.real(self._corr_approx(*args)),
            y=np.real(self._C_array), x=self._t, final_rmse=final_rmse,
            default_guess_scenario="correlation_real", N=Nr,
            sigma=sigma, guesses=guesses, lower=lower, upper=upper)
        end_real = time()

        # Fit imaginary part
        start_imag = time()
        rmse_imag, params_imag = _run_fit(
            lambda *args: np.imag(self._corr_approx(*args)),
            y=np.imag(self._C_array), x=self._t, final_rmse=final_rmse,
            default_guess_scenario="correlation_imag", N=Ni,
            sigma=sigma, guesses=guesses, lower=lower, upper=upper)
        end_imag = time()

        # Calculate Fit Times
        fit_time_real = end_real - start_real
        fit_time_imag = end_imag - start_imag

        # Generate summary
        full_summary = _two_column_summary(
            params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
            rmse_imag, rmse_real)

        fitInfo = {"Nr": len(params_real[0]), "Ni": len(params_imag[0]),
                   "fit_time_real": fit_time_real,
                   "fit_time_imag": fit_time_imag,
                   "rmse_real": rmse_real, "rmse_imag": rmse_imag,
                   "params_real": params_real,
                   "params_imag": params_imag, "summary": full_summary}
        bath = self._generate_bath(params_real, params_imag)
        bath.correlation_function = self._C_fun
        return bath, fitInfo

    def _generate_bath(self, params_real, params_imag):
        """ Calculate the Matsubara coefficients and frequencies for the
        fitted underdamped oscillators and generate the corresponding bosonic
        bath

        Parameters
        ----------
        params_real : np.array
            array of shape (N,3) where N is the number of fitted terms
            for the real part
        params_imag : np.imag
            array of shape (N,3) where N is the number of fitted terms
            for the imaginary part

        Returns
        -------
        A bosonic Bath constructed from the fitted exponents
        """

        a, b, c = params_real
        a2, b2, c2 = params_imag

        ckAR = [0.5 * x + 0j for x in a]  # the 0.5 is from the cosine
        # extend the list with the complex conjugates:
        ckAR.extend(np.conjugate(ckAR))
        vkAR = [-x - 1.0j * y for x, y in zip(b, c)]
        vkAR.extend([-x + 1.0j * y for x, y in zip(b, c)])

        ckAI = [-0.5j * x for x in a2]  # the 0.5 is from the sine
        # extend the list with the complex conjugates:
        ckAI.extend(np.conjugate(ckAI))
        vkAI = [-x - 1.0j * y for x, y in zip(b2, c2)]
        vkAI.extend([-x + 1.0j * y for x, y in zip(b2, c2)])

        return BosonicBath(
            self.Q, ckAR, vkAR, ckAI, vkAI, T=self.T)


class OhmicBath:
    """
    A helper class for constructing a Bosonic bath from with Ohmic
    spectrum. Requires the mpmath module.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.
    T : Float
        Temperature of the bath
    alpha : float
        Coupling strength
    wc : float
        Cutoff parameter
    s : float
        Power of w in the spectral density
    """

    def __init__(self, T, Q, alpha, wc, s):
        self.alpha = alpha
        self.wc = wc
        self.s = s
        self.Q = Q
        self.T = T
        if _mpmath_available is False:
            print(
                "The mpmath module is needed for the description"
                " of Ohmic baths")

    def spectral_density(self, w):
        """
        Calculates the Ohmic spectral density, see Eq. 36 in the BoFiN
        paper (DOI: 10.1103/PhysRevResearch.5.013181).

        Parameters
        ----------
        w : float or array
            Energy of the mode

        Returns
        -------
        The spectral density of the mode with energy w
        """

        return (self.alpha * w ** (self.s)
                / (self.wc ** (1 - self.s))
                * np.e ** (-abs(w) / self.wc))

    def correlation_function(self, t):
        """
        Calculates the correlation function of an Ohmic bath.

        Parameters
        ----------
        t : float or array
            time

        Returns
        -------
        The correlation function at time t
        """

        if self.T != 0:
            corr = (
                (1 / np.pi)
                * self.alpha
                * self.wc ** (1 - self.s)
                * (1/self.T) ** (-(self.s + 1))
                * mp.gamma(self.s + 1)
            )
            z1_u = (1 + (1/self.T) * self.wc - 1.0j *
                    self.wc * t) / ((1/self.T) * self.wc)
            z2_u = (1 + 1.0j * self.wc * t) / ((1/self.T) * self.wc)
            return np.array(
                [
                    complex(
                        corr * (mp.zeta(self.s + 1, u1) +
                                mp.zeta(self.s + 1, u2)))
                    for u1, u2 in zip(z1_u, z2_u)],
                dtype=np.complex128,)
        else:
            corr = (1 / np.pi)*self.alpha*self.wc**(self.s+1) * \
                mp.gamma(self.s+1)*(1+1j*self.wc*t)**(-(self.s+1))
            return np.array(corr, dtype=np.complex128)

    def make_correlation_fit(
            self, x, rmse=1e-5, lower=None, upper=None,
            sigma=None, guesses=None, Nr=None, Ni=None):
        """
        Provides a fit to the spectral density or corelation function
        with N underdamped oscillators baths, This function gets the
        number of harmonic oscillators based on reducing the normalized
        root mean squared error below a certain threshold

        TODO please make parameters below like actual parameters

        Parameters
        ----------
        J : np.array
            Spectral density to be fit.
        w : np.array
            range of frequencies for the fit.
        N : optional,tuple
            Number of underdamped oscillators and exponents to use
            (N,Nk) if the the method is spectral
            Number of underdamped oscillators for the real and imaginary
            part if the method is correlation.
            when set to None the number of oscillators is found according to
            the rmse, and the Nk is set to 1
        rmse : float
            Desired normalized root mean squared error. Only used if N is
            not provided
        lower : list
            lower bounds on the parameters for the fit.
        upper: list
            upper bounds on the parameters for the fit
        sigma: float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guess for the parameters.
        """
        fc = CorrelationFitter(self.Q, self.T, x, self.correlation_function)
        bath, fitInfo = fc.get_fit(final_rmse=rmse,
                                   lower=lower, upper=upper,
                                   sigma=sigma, guesses=guesses,
                                   Nr=Nr, Ni=Ni)
        return bath, fitInfo

    def make_spectral_fit(self, x, rmse=1e-5, lower=None, upper=None,
                          sigma=None, guesses=None, N=None, Nk=5):
        """
        Provides a fit to the spectral density or corelation function
        with N underdamped oscillators baths, This function gets the
        number of harmonic oscillators based on reducing the normalized
        root mean squared error below a certain threshold

        TODO please make parameters below like actual parameters

        Parameters
        ----------
        J : np.array
            Spectral density to be fit.
        w : np.array
            range of frequencies for the fit.
        N : optional,tuple
            Number of underdamped oscillators and exponents to use
            (N,Nk) if the the method is spectral
            Number of underdamped oscillators for the real and imaginary
            part if the method is correlation.
            when set to None the number of oscillators is found according to
            the rmse, and the Nk is set to 1
        rmse : float
            Desired normalized root mean squared error. Only used if N is
            not provided
        lower : list
            lower bounds on the parameters for the fit.
        upper: list
            upper bounds on the parameters for the fit
        sigma: float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guess for the parameters.
        """
        fs = SpectralFitter(T=self.T, Q=self.Q, w=x, J=self.spectral_density)
        bath, fitInfo = fs.get_fit(N=N, final_rmse=rmse, lower=lower,
                                   upper=upper, Nk=Nk,
                                   sigma=sigma, guesses=guesses)
        return bath, fitInfo


# Utility functions


def _pack(a, b, c):
    """
    Pack parameter lists for fitting. In both use cases (spectral fit,
    correlation fit), the fit parameters are three arrays of equal length.
    """
    return np.concatenate((a, b, c))


def _unpack(params):
    """
    Unpack parameter lists for fitting. In both use cases (spectral fit,
    correlation fit), the fit parameters are three arrays of equal length.
    """
    N = len(params) // 3
    a = params[:N]
    b = params[N: 2 * N]
    c = params[2 * N:]
    return a, b, c


def _leastsq(func, y, x, guesses=None, lower=None, upper=None, sigma=None):
    """
    Performs nonlinear least squares to fit the function func to x and y

    Parameters
    ----------
    func : function
        The function we wish to fit.
    x : np.array
        a numpy array containing the independent variable used for the fit
    y : np.array
        a numpy array containing the dependent variable we use for the fit
    guesses : list
        Initial guess for the parameters.
    lower : list
        lower bounds on the parameters for the fit.
    upper : list
        upper bounds on the parameters for the fit
    sigma : float
        uncertainty in the data considered for the fit

    Returns
    -------
    params: list
        It returns the fitted parameters.
    """

    sigma = [sigma] * len(x)
    params, _ = curve_fit(
        lambda x, *params: func(x, *_unpack(params)),
        x,
        y,
        p0=guesses,
        bounds=(lower, upper),
        sigma=sigma,
        maxfev=int(1e9),
        method="trf",
    )

    return _unpack(params)


def _rmse(func, x, y, a, b, c):
    """
    Calculates the normalized root mean squared error for fits
    from the fitted parameters a, b, c

    Parameters
    ----------
    func : function
        The approximated function for which we want to compute the rmse.
    x: np.array
        a numpy array containing the independent variable used for the fit
    y: np.array
        a numpy array containing the dependent variable used for the fit
    a, b, c : list
        fitted parameters

    Returns
    -------
    rmse: float
        The normalized root mean squared error for the fit, the closer
        to zero the better the fit.
    """
    yhat = func(x, a, b, c)
    rmse = np.sqrt(np.mean((yhat - y) ** 2) / len(y)) / \
        (np.max(y) - np.min(y))
    return rmse


def _fit(func, C, t, N, default_guess_scenario='',
         guesses=None, lower=None, upper=None, sigma=None):
    """
    Performs a fit the function func to t and C, with N number of
    terms in func, the guesses,bounds and uncertainty can be determined
    by the user.If none is provided it constructs default ones according
    to the label.

    Parameters
    ----------
    func : function
        The function we wish to fit.
    C : np.array
        a numpy array containing the dependent variable used for the fit
    t : np.array
        a numpy array containing the independent variable used for the fit
    N : int
        The number of modes / baths used for the fitting.
    default_guess_scenario : str
        Determines how the default guesses and bounds are chosen (in the case
        guesses or bounds are not specified). May be 'correlation_real',
        'correlation_imag' or any other string. Any other string will use
        guesses and bounds designed for the fitting of spectral densities.
    guesses : list
        Initial guess for the parameters.
    lower : list
        lower bounds on the parameters for the fit.
    upper: list
        upper bounds on the parameters for the fit
    sigma: float
        uncertainty in the data considered for the fit

    Returns
    -------
    params:
        It returns the fitted parameters as a list.
    rmse:
        It returns the normalized mean squared error from the fit
    """

    C_max = abs(max(C, key=abs))
    if C_max == 0:
        # When the target function is zero
        rmse = 0
        params = [0, 0, 0]
        return rmse, params

    if None in [guesses, lower, upper, sigma]:
        # No parameters for the fit provided, using default ones
        sigma = 1e-4
        wc = t[np.argmax(C)]
        if default_guess_scenario == "correlation_real":
            guesses = _pack([C_max] * N, [-wc] * N, [wc] * N)
            lower = _pack([-20 * C_max] * N, [-np.inf] * N, [0.0] * N)
            upper = _pack([20 * C_max] * N, [0.1] * N, [np.inf] * N)
        elif default_guess_scenario == "correlation_imag":
            guesses = _pack([-C_max] * N, [-2] * N, [1] * N)
            lower = _pack([-5 * C_max] * N, [-100] * N, [0.0] * N)
            upper = _pack([5 * C_max] * N, [0.01] * N, [100] * N)
        else:
            guesses = _pack([C_max] * N, [wc] * N, [wc] * N)
            lower = _pack([-100 * C_max] * N,
                          [0.1 * wc] * N, [0.1 * wc] * N)
            upper = _pack([100 * C_max] * N,
                          [100 * wc] * N, [100 * wc] * N)

    a, b, c = _leastsq(func, C, t,
                       sigma=sigma, guesses=guesses, lower=lower, upper=upper)
    rmse = _rmse(func, t, C, a, b, c)
    return rmse, [a, b, c]


def _run_fit(funcx, y, x, final_rmse, default_guess_scenario=None, N=None,
             sigma=None, guesses=None, lower=None, upper=None):
    """
    It iteratively tries to fit the funcx to y on the interval x.
    If N is provided the fit is done with N modes, if it is
    None then this automatically finds the smallest number of modes that
    whose mean squared error is smaller than final_rmse

    Parameters
    ----------
    funcx : function
        The function we wish to fit.
    y : np.array
        The function used for the fitting
    x : np.array
        a numpy array containing the independent variable used for the fit
    final_rmse : float
        Desired normalized root mean squared error
    default_guess_scenario : str
        Determines how the default guesses and bounds are chosen (in the case
        guesses or bounds are not specified). May be 'correlation_real',
        'correlation_imag' or any other string. Any other string will use
        guesses and bounds designed for the fitting of spectral densities.
    N : optional , int
        The number of modes used for the fitting, if not provided starts at
        1 and increases until a desired RMSE is satisfied
    sigma: float
        uncertainty in the data considered for the fit
    guesses : list
        Initial guess for the parameters.
    lower : list
        lower bounds on the parameters for the fit.
    upper: list
        upper bounds on the parameters for the fit

    Returns
    -------
    params:
        It returns the fitted parameters as a list.
    rmse:
        It returns the normalized mean squared error from the fit
    """

    if N is None:
        N = 1
        flag = False
    else:
        flag = True
    rmse1 = np.inf
    while rmse1 > final_rmse:
        N += 1
        rmse1, params = _fit(
            funcx, y, x, N, default_guess_scenario,
            sigma=sigma, guesses=guesses, lower=lower, upper=upper)
        if flag is True:
            break
    return rmse1, params


def _gen_summary(time, rmse, N, label, lam, gamma, w0):
    summary = (f"Result of fitting {label} "
               f"with {N} terms: \n \n {'Parameters': <10}|"
               f"{'lam': ^10}|{'gamma': ^10}|{'w0': >5} \n ")
    for k in range(len(gamma)):
        summary += (
            f"{k+1: <10}|{lam[k]: ^10.2e}|{gamma[k]:^10.2e}|"
            f"{w0[k]:>5.2e}\n ")
    summary += (f"\nA  normalized RMSE of {rmse: .2e}"
                f" was obtained for the {label}\n")
    summary += f" The current fit took {time: 2f} seconds"
    return summary


def _two_column_summary(
        params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
        rmse_imag, rmse_real):
    lam, gamma, w0 = params_real
    lam2, gamma2, w02 = params_imag

    # Generate nicely formatted summary
    summary_real = _gen_summary(
        fit_time_real,
        rmse_real,
        Nr,
        "The Real Part Of  \n the Correlation Function", lam, gamma,
        w0)
    summary_imag = _gen_summary(
        fit_time_imag,
        rmse_imag,
        Ni,
        "The Imaginary Part \n Of the Correlation Function", lam2,
        gamma2, w02)

    full_summary = "Fit correlation class instance: \n \n"
    lines_real = summary_real.splitlines()
    lines_imag = summary_imag.splitlines()
    max_lines = max(len(lines_real), len(lines_imag))
    # Fill the shorter string with blank lines
    lines_real = lines_real[:-1] + (max_lines - len(lines_real)
                                    ) * [""] + [lines_real[-1]]
    lines_imag = lines_imag[:-1] + (max_lines - len(lines_imag)
                                    ) * [""] + [lines_imag[-1]]
    # Find the maximum line length in each column
    max_length1 = max(len(line) for line in lines_real)
    max_length2 = max(len(line) for line in lines_imag)

    # Print the strings side by side with a vertical bar separator
    for line1, line2 in zip(lines_real, lines_imag):
        formatted_line1 = f"{line1:<{max_length1}} |"
        formatted_line2 = f"{line2:<{max_length2}}"
        full_summary += formatted_line1 + formatted_line2 + "\n"
    return full_summary
