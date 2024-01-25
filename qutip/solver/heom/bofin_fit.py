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
from qutip.core.superoperator import spre, spost
from qutip.solver.heom import UnderDampedBath, BosonicBath
from scipy.interpolate import InterpolatedUnivariateSpline


class SpectralFitter:
    """
    A helper class for constructing a Bosonic bath from a spectral density fit.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    T : float
        Bath temperature.

    w : np.array
        The range on which to perform the fit

    J : np.array or func
        The spectral density to be fitted as an array or function
    """

    def __init__(self, T, Q, w, J):
        self.Q = Q
        self.T = T
        self.set_function(w, J)

    def set_function(self, w, J):
        """
        This function creates a discretized version of the spectral density
        if the spectral density is provided as a function, and a function if
        an array is provided.

        The array is needed to run the least squares algorithm, while the
        the function is used to assign a spectral density to the bosonic bath
        object
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
        (https://doi.org/10.1103/PhysRevResearch.5.013181 see eq 38)

        Parameters
        ----------
        w : np.array
            The frequency of the spectral density
        a: np .array
            Array of coupling constant
        b :s np.array
            Array of cutoff parameters
        c : np.array
            Array of is  resonant frequencies
        """
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
        Provides a fit to the spectral density with N underdamped
        oscillators baths, This function gets the number of harmonic
        oscillators based on reducing the normalized root mean
        squared error below a certain threshold

        Parameters
        ----------
        J : np.array
            Spectral density to be fit.
        w : np.array
            range of frequencies for the fit.
        N : optional,int
            Number of underdamped oscillators to use,
            if set to False it is found automatically.
        
        Nk : optional,int
            Number of exponential terms used to approximate the bath correlation
            functions. Defaults to 5
        final_rmse : float
            Desired normalized root mean squared error .
        lower : list
            lower bounds on the parameters for the fit.
        upper: list
            upper bounds on the parameters for the fit
        sigma: float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guess for the parameters.

        Returns
        ----------
            A Bosonic Bath created with the fit parameters with the original
            spectral density function (that was provided or interpolated)
            A dictionary containing the fit information
        """

        start = time()
        rmse, params = _run_fit(
            self._spectral_density_approx, self._J_array, self._w, final_rmse,
            N=N, sigma=sigma, label="Spectral Density", guesses=guesses,
            lower=lower, upper=upper)
        spec_n = len(params[0])
        end = time()
        fit_time = end - start
        bath = self._generate_bath(params,Nk)
        bath.spectral_density = self._J_fun
        summary = gen_summary(
            fit_time, rmse, N, "The Spectral Density", *params)
        self.fitInfo = {
            "fit_time": fit_time, "rmse": rmse, "N": spec_n, "params": params,
            "Nk": Nk, "summary": summary}
        return bath, self.fitInfo

    def _generate_bath(self, params, Nk):
        """
        Obtains the bath exponents from the list of fit parameters some
        transformations are done, to reverse the ones in the UnderDampedBath
        They are done to change the spectral density from eq.38 to eq.16
        (https://doi.org/10.1103/PhysRevResearch.5.013181) and vice-versa

        Parameters
        ----------
        params: list
            The parameters obtained from the fit

        Returns
        ----------
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
        # both w0, and lam  modifications are needed to input the
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
    A helper class for constructing a Bosonic bath from the
    correlation function fit.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.
    T: Float
        Temperature of the bath
    """

    def __init__(self, Q, T, t, C):
        self.Q = Q
        self.T = T
        self.set_function(t, C)

    def set_function(self, t, C):
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
            self._C_fun_r = InterpolatedUnivariateSpline(
                t, np.real(C))
            self._C_fun_i = InterpolatedUnivariateSpline(t, np.imag(C))
            self._C_fun = lambda t: self._C_fun_r(t)+1j*self._C_fun_i(t)

    def _corr_approx(self, t, a, b, c):
        """This is the form of the correlation function to be used for fitting

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
        """Fit the correlation function with Ni underdamped oscillators baths
        for the imaginary part of the correlation function and Nr for the real.
        Provides a fit to the with N underdamped oscillators. If no number of
        terms is provided This function gets the number of harmonic oscillators
        based on reducing the normalized root mean squared error below a
        certain threshold.

        Parameters
        ----------
        t : np.array
            range of frequencies for the fit.
        C : np.array
            Correlation function to be fit.

        Nr : optional,int
            Number of underdamped oscillators to use for the real part,
            if set to None it is found automatically.
        Ni : optional,int
            Number of underdamped oscillators to use for the imaginary part,
            if set to None it is found automatically.
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
        """

        # Fit real part
        start_real = time()
        rmse_real, params_real = _run_fit(
            funcx=lambda * args: np.real(self._corr_approx(*args)),
            funcy=np.real(self._C_array),
            x=self._t, final_rmse=final_rmse, label="correlation_real",
            sigma=sigma, N=Nr, guesses=guesses, lower=lower, upper=upper)
        end_real = time()

        # Fit imaginary part
        start_imag = time()
        rmse_imag, params_imag = _run_fit(
            lambda *args: np.imag(self._corr_approx(*args)),
            np.imag(self._C_array), self._t, final_rmse,
            label="correlation_imag", sigma=sigma, N=Ni,
            guesses=guesses, lower=lower, upper=upper)
        end_imag = time()

        # Calculate Fit Times
        fit_time_real = end_real - start_real
        fit_time_imag = end_imag - start_imag

        # Generate summary
        full_summary = two_column_summary(
            params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
            rmse_imag, rmse_real)

        self.fitInfo = {"Nr": len(params_real[0]), "Ni": len(params_imag[0]),
                        "fit_time_real": fit_time_real,
                        "fit_time_imag": fit_time_imag,
                        "rmse_real": rmse_real, "rmse_imag": rmse_imag,
                        "params_real": params_real,
                        "params_imag": params_imag, "summary": full_summary}
        bath = self._generate_bath(params_real, params_imag)
        bath.correlation_function = self._C_fun
        return bath, self.fitInfo

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
        ----------
            A bosonic Bath constructed from the fitted exponents
        """

        lam, gamma, w0 = params_real
        lam2, gamma2, w02 = params_imag
        ckAR = [0.5 * x + 0j for x in lam]  # the 0.5 is from the cosine
        # extend the list with the complex conjugates:
        ckAR.extend(np.conjugate(ckAR))
        vkAR = [-x - 1.0j * y for x, y in zip(gamma, w0)]
        vkAR.extend([-x + 1.0j * y for x, y in zip(gamma, w0)])
        ckAI = [-0.5j * x for x in lam2]  # the 0.5 is from the sine
        # extend the list with the complex conjugates:
        ckAI.extend(np.conjugate(ckAI))

        vkAI = [-x - 1.0j * y for x, y in zip(gamma2, w02)]
        vkAI.extend([-x + 1.0j * y for x, y in zip(gamma2, w02)])
        return BosonicBath(
            self.Q, ckAR, vkAR, ckAI, vkAI, T=self.T)


class OhmicBath:
    """
    A helper class for constructing a Bosonic bath from with Ohmic
    spectrum.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.
    T : Float
        Temperature of the bath
    alpha : float
        Coupling strenght
    wc : float
        Cutoff parameter
    s : float
        power of w in the spectral density
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
        """Calculates the Ohmic spectral density
        Parameters
        ----------
        w: float or array
            Energy of the mode
        Returns
        ----------
            The spectral density of the mode with energy w"""
        return (
            self.alpha
            * w ** (self.s)
            / (self.wc ** (1 - self.s))
            * np.e ** (-abs(w) / self.wc)
        )

    def correlation_function(self, t):
        """Calculates the Ohmic spectral density
        Parameters
        ----------
        t: float or array
            time
        Returns
        ----------
            The correlation function at time t"""
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
        C = self.correlation_function(x)
        fc = CorrelationFitter(self.Q, self.T, x, C)
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
        J = self.spectral_density(x)
        fs = SpectralFitter(T=self.T, Q=self.Q, w=x, J=J, Nk=Nk)
        bath, fitInfo = fs.get_fit(N=N, final_rmse=rmse, lower=lower,
                                   upper=upper,
                                   sigma=sigma, guesses=guesses)
        return bath, fitInfo

# Utility functions


def pack(a, b, c):
    """Pack parameter lists for fitting."""
    return np.concatenate((a, b, c))


def unpack(params):
    """Unpack parameter lists for fitting."""
    N = len(params) // 3
    a = params[:N]
    b = params[N: 2 * N]
    c = params[2 * N:]
    return a, b, c


def _leastsq(
    func, y, x, guesses=None,
    lower=None, upper=None, sigma=None
):
    """
    Performs nonlinear least squares  to fit the function func to x and y

    Parameters
    ----------
    func : function
        The function we wish to fit.
    x: np.array
        a numpy array containing the independent variable used for the fit
    y: np.array
        a numpy array containing the dependent variable we use for the fit
    guesses : list
        Initial guess for the parameters.
    lower : list
        lower bounds on the parameters for the fit.
    upper: list
        upper bounds on the parameters for the fit
    sigma:
        uncertainty in the data considered for the fit
    Returns
    -------
    params: list
        It returns the fitted parameters.
    """
    try:
        sigma = [sigma] * len(x)
        if None in [guesses, lower, upper, sigma]:
            raise Exception(
                "Fit parameters were not provided"
            )  # maybe unnecessary and can let it go to scipy defaults
    except Exception:
        pass
    params, _ = curve_fit(
        lambda x, *params: func(x, *unpack(params)),
        x,
        y,
        p0=guesses,
        bounds=(lower, upper),
        sigma=sigma,
        maxfev=int(1e9),
        method="trf",
    )

    return unpack(params)


def _rmse(func, x, y, lam, gamma, w0):
    """
    Calculates the normalized root Mean squared error for fits
    from the fitted parameters lam,gamma,w0

    Parameters
    ----------
    func : function
        The approximated function for which we want to compute the rmse.
    x: np.array
        a numpy array containing the independent variable used for the fit
    y: np.array
        a numpy array containing the dependent variable used for the fit
    lam : list
        a listed containing fitted couplings strength.
    gamma : list
        a list containing fitted cutoff frequencies.
    w0s:
        a list containing fitted resonance frequencies
    Returns
    -------
    rmse: float
        The normalized root mean squared error for the fit, the closer
        to zero the better the fit.
    """
    yhat = func(x, lam, gamma, w0)
    rmse = np.sqrt(np.mean((yhat - y) ** 2) / len(y)) / \
        (np.max(y) - np.min(y))
    return rmse


def _fit(
    func, C, t, N=4, label="correlation_real",
    guesses=None, lower=None, upper=None, sigma=None
):
    """
    Performs a fit the function func to t and C, with N number of
    terms in func, the guesses,bounds and uncertainty can be determined
    by the user.If none is provided it constructs default ones according
    to the label.

    Parameters
    ----------
    func : function
        The function we wish to fit.
    x: np.array
        a numpy array containing the independent variable used for the fit
    y: np.array
        a numpy array containing the dependent variable used for the fit
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
    try:
        if None in [guesses, lower, upper, sigma]:
            raise Exception(
                "No parameters for the fit provided, using default ones"
            )
    except Exception:
        sigma = 1e-4
        C_max = abs(max(C, key=abs))
        if C_max == 0:  ## When The correlation does not have imaginary or real part
            rmse = 0
            params = [0, 0, 0]
            return rmse, params
        wc = t[np.argmax(C)]
        if label == "correlation_real":
            guesses = pack([C_max] * N, [-wc] * N, [wc] * N)
            lower = pack([-20 * C_max] * N, [-np.inf] * N, [0.0] * N)
            upper = pack([20 * C_max] * N, [0.1] * N, [np.inf] * N)
        elif label == "correlation_imag":
            guesses = pack([-C_max] * N, [-2] * N, [1] * N)
            lower = pack([-5 * C_max] * N, [-100] * N, [0.0] * N)
            upper = pack([5 * C_max] * N, [0.01] * N, [100] * N)

        else:
            guesses = pack([C_max] * N, [wc] * N, [wc] * N)
            lower = pack([-100 * C_max] * N,
                         [0.1 * wc] * N, [0.1 * wc] * N)
            upper = pack([100 * C_max] * N,
                         [100 * wc] * N, [100 * wc] * N)

    lam, gamma, w0 = _leastsq(
        func,
        C,
        t,
        sigma=sigma,
        guesses=guesses,
        lower=lower,
        upper=upper,
    )
    rmse = _rmse(func, t, C, lam=lam, gamma=gamma, w0=w0)
    params = [lam, gamma, w0]
    return rmse, params


def _run_fit(funcx, funcy, x, final_rmse, label=None, N=None,
             sigma=None, guesses=None, lower=None, upper=None):
    """
    It iteratively tries to fit the fucx to funcy on the interval x,
    if the Ns are provided the fit is done with N modes, if they are
    None then One automatically finds the smallest number of modes that
    whose mean squared error is smaller than final_rmse

    Parameters
    ----------
    funcx : function
        The function we wish to fit.
    funcy : function
        The function used for the fitting
    x : np.array
        a numpy array containing the independent variable used for the fit
    N : optional , int
        The number of modes used for the fitting, if not provided starts at
        1 and increases until a desired RMSE is satisfied
    label : str
        Denotes the options for the different default guesses and bounds if
        they are not provided it can be
        - correlation_real
        - correlation_imag
        Any other string will use guesses and bounds designed for spectral
        densities
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
    if N is None:
        N = 1
        flag = False
    else:
        flag = True
        N = N-1
    rmse1 = 8
    while rmse1 > final_rmse:
        N += 1
        rmse1, params = _fit(
            funcx,
            funcy,
            x,
            N=N,
            sigma=sigma,
            guesses=guesses,
            lower=lower,
            upper=upper,
            label=label,
        )
        if flag is True:
            break
    return rmse1, params


def gen_summary(time, rmse, N, label, lam, gamma, w0):
    summary = f"Result of fitting {label} "\
        f"with {N} terms: \n \n {'Parameters': <10}|"\
        f"{'lam': ^10}|{'gamma': ^10}|{'w0': >5} \n "
    for k in range(len(gamma)):
        summary += (
            f"{k+1: <10}|{lam[k]: ^10.2e}|{gamma[k]:^10.2e}|"
            f"{w0[k]:>5.2e}\n "
        )
    summary += f"\nA  normalized RMSE of {rmse: .2e}"\
        f" was obtained for the {label}\n"
    summary += f" The current fit took {time: 2f} seconds"
    return summary


def two_column_summary(
        params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
        rmse_imag, rmse_real):
    lam, gamma, w0 = params_real
    lam2, gamma2, w02 = params_imag

    # Generate nicely formatted summary
    summary_real = gen_summary(
        fit_time_real,
        rmse_real,
        Nr,
        "The Real Part Of  \n the Correlation Function", lam, gamma,
        w0)
    summary_imag = gen_summary(
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