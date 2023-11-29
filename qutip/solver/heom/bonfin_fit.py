import numpy as np
from time import time
try:
    from mpmath import mp
    _mpmath_available = True
except ModuleNotFoundError:
    _mpmath_available = False
from scipy.optimize import curve_fit
from qutip.visualization import gen_spectral_plots, gen_corr_plots
from qutip.core.superoperator import spre, spost
from qutip.solver.heom import UnderDampedBath, BosonicBath, BathExponent


class SpectralFitter:
    """
    A helper class for constructing a Bosonic bath from a spectral density fit.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.
    """

    def __init__(self, T, Q, Nk=14):
        self.Q = Q
        self.T = T
        self.Nk = Nk

    def __str__(self):
        try:
            lam, gamma, w0 = self.fitInfo['params']
            summary = gen_summary(
                self.fitInfo['fit_time'],
                self.fitInfo['rmse'],
                self.fitInfo['N'],
                "The Spectral Density", lam, gamma, w0)
            return summary
        except NameError:
            return "Fit correlation instance: \n No fit has been performed yet"

    def summary(self):
        print(self.__str__())

    def _spectral_density_approx(self, w, a, b, c):
        """Calculate the fitted value of the function for the parameters."""
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

    def fitted_spectral_density(self, w):
        lam, gamma, w0 = self.fitInfo['params']
        return self._spectral_density_approx(w, lam, gamma, w0)

    def fitted_power_spectrum(self, w):
        return self.fitted_spectral_density(w) * 2 * (
            (1 / (np.e ** (w * (1/self.T)) - 1)) + 1)

    def get_fit(
        self,
        J,
        w,
        N=None,
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
        """
        # Check if input is function if it is turn into array
        # on the range
        if callable(J):
            J = J(w)
        start = time()
        rmse, params = _run_fit(self._spectral_density_approx, J, w,
                                final_rmse, N=N, sigma=sigma,
                                label="Spectral Density", guesses=guesses,
                                lower=lower, upper=upper)
        params_spec = params
        spec_n = len(params[0])
        end = time()
        fit_time = end - start
        self.fitInfo = {"fit_time": fit_time, "rmse": rmse,
                        "N": spec_n, "params": params_spec, "Nk": self.Nk}
        bath = self._matsubara_spectral_fit()
        self.summary()
        return bath, self.fitInfo

    def fit_plots(self, w, J, t, C, w2, S):
        gen_spectral_plots(self, w, J, t, C, w2, S)

    def _matsubara_spectral_fit(self):
        """
        Obtains the bath exponents from the list of fit parameters
        """
        lam, gamma, w0 = self.fitInfo['params']
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
        terminator = 0. * spre(self.Q)
        # the number of matsubara expansion terms to include in the terminator:
        terminator_max_k = 1000

        for lamt, Gamma, Om in zip(lam, gamma, w0):
            coeffs = UnderDampedBath._matsubara_params(
                lamt, 2 * Gamma, Om + 0j, self.T, self.Nk)
            ckAR.append(coeffs[0])
            vkAR.append(coeffs[1])
            ckAI.append(coeffs[2])
            vkAI.append(coeffs[3])
            terminator_factor = 0
            for k in range(self.Nk + 1, terminator_max_k):
                ek = 2 * np.pi * k / (1/self.T)
                ck = (
                    (-2 * lamt * 2 * Gamma / (1/self.T)) * ek /
                    (
                        ((Om + 1.0j * Gamma)**2 + ek**2) *
                        ((Om - 1.0j * Gamma)**2 + ek**2)
                    )
                )
                terminator_factor += ck / ek
            terminator += terminator_factor * (
                2 * spre(self.Q) * spost(self.Q.dag())
                - spre(self.Q.dag() * self.Q)
                - spost(self.Q.dag() * self.Q)
            )

        ckAR = np.array(ckAR).flatten()
        ckAI = np.array(ckAI).flatten()
        vkAR = np.array(vkAR).flatten()
        vkAI = np.array(vkAI).flatten()
        self.terminator = terminator
        return BosonicBath(self.Q, ckAR, vkAR, ckAI, vkAI)

    def correlation_function(self, t):
        """Computes the correlation function from 
         the exponents"""
        corr = 0+0j
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


class CorrelationFitter:
    """
    A helper class for constructing a Bosonic bath from the
    correlation function fit.

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.
    """

    def __init__(self, Q):
        self.Q = Q

    def _corr_approx(self, t, a, b, c):
        """Calculate the fitted value of the function for the parameters."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        return np.sum(
            a[:, None] * np.exp(b[:, None] * t) * np.exp(1j * c[:, None] * t),
            axis=0,
        )

    def fitted_correlation(self, t):
        """Calculate the fitted value of the function for the parameters."""
        lam, gamma, w0 = self.fitInfo['params_real']
        lam2, gamma2, w02 = self.fitInfo['params_imag']
        return np.real(
            self._corr_approx(t, lam, gamma, w0)) + 1j * np.imag(
            self._corr_approx(t, lam2, gamma2, w02))

    def __str__(self):
        try:
            lam, gamma, w0 = self.fitInfo['params_real']
            lam2, gamma2, w02 = self.fitInfo['params_imag']
            summary = gen_summary(
                self.fitInfo['fit_time_real'],
                self.fitInfo['rmse_real'],
                self.fitInfo['Nr'],
                "The Real Part Of  \n the Correlation Function", lam, gamma,
                w0)
            summary2 = gen_summary(
                self.fitInfo['fit_time_imag'],
                self.fitInfo['rmse_imag'],
                self.fitInfo['Ni'],
                "The Imaginary Part \n Of the Correlation Function", lam2,
                gamma2, w02)

            return summary, summary2

        except NameError:
            return "Fit correlation instance: \n No fit has been performed yet"

    def summary(self):
        print("Fit correlation class instance: \n \n")
        string1, string2 = self.__str__()
        lines1 = string1.splitlines()
        lines2 = string2.splitlines()
        max_lines = max(len(lines1), len(lines2))
        # Fill the shorter string with blank lines
        lines1 = lines1[:-1] + (max_lines - len(lines1)) * [""] + [lines1[-1]]
        lines2 = lines2[:-1] + (max_lines - len(lines2)) * [""] + [lines2[-1]]
        # Find the maximum line length in each column
        max_length1 = max(len(line) for line in lines1)
        max_length2 = max(len(line) for line in lines2)

        # Print the strings side by side with a vertical bar separator
        for line1, line2 in zip(lines1, lines2):
            formatted_line1 = f"{line1:<{max_length1}} |"
            formatted_line2 = f"{line2:<{max_length2}}"
            print(formatted_line1 + formatted_line2)

    def fit_plots(self, w, J, t, C, w2, S, beta):
        gen_corr_plots(self, w, J, t, C, w2, S, beta)

    def get_fit(
        self,
        t,
        C,
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
        upper: list
            upper bounds on the parameters for the fit
        sigma: float
            uncertainty in the data considered for the fit
        guesses : list
            Initial guess for the parameters.
        """
        # Check if input is function if it is turn into array
        # on the range
        if callable(C):
            C = C(t)
        start = time()
        rmse_real, params_real = _run_fit(
            lambda *args: np.real(self._corr_approx(*args)),
            np.real(C), t, final_rmse,
            label="correlation_real", sigma=sigma, N=Nr,
            guesses=guesses, lower=lower, upper=upper)
        end = time()
        fit_time_real = end - start
        start = time()
        rmse_imag, params_imag = _run_fit(
            lambda *args: np.imag(self._corr_approx(*args)),
            np.imag(C), t, final_rmse,
            label="correlation_imag", sigma=sigma, N=Ni,
            guesses=guesses, lower=lower, upper=upper)
        end = time()
        fit_time_imag = end - start
        self.fitInfo = {"Nr": len(params_real[0]), "Ni": len(params_imag[0]),
                        "fit_time_real": fit_time_real,
                        "fit_time_imag": fit_time_imag,
                        "rmse_real": rmse_real, "rmse_imag": rmse_imag,
                        "params_real": params_real,
                        "params_imag": params_imag}
        bath = self._matsubara_coefficients()
        self.exponents = bath.exponents
        return bath, self.fitInfo

    def _matsubara_coefficients(self):
        lam, gamma, w0 = self.fitInfo['params_real']
        lam2, gamma2, w02 = self.fitInfo['params_imag']
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
        return BosonicBath(self.Q, ckAR, vkAR, ckAI, vkAI)

    def power_spectrum_approx(self, w):
        S = np.zeros(len(w), dtype=np.complex128)

        def s(x, y): return 2 * x * np.real(y) / (
            (w - np.imag(y)) ** 2 + (np.real(y)) ** 2)
        for exp in self.exponents:
            if (
                exp.type == BathExponent.types['R'] or
                exp.type == BathExponent.types['RI']
            ):
                S += s(exp.ck, exp.vk)
            elif exp.type == BathExponent.types['RI']:
                S += s(1j*exp.ck2, exp.vk)
            else:
                S += s(1j*exp.ck, exp.vk)
        return S

    def spectral_density_approx(self, w, beta):
        J = np.real(
            self.power_spectrum_approx(w) /
            (((1 / (np.e**(w * beta) - 1)) + 1) * 2)
        )
        return J


class OhmicBath(BosonicBath):
    def __init__(self, T, Q, alpha, wc, s, Nk=4, method="spectral", rmse=7e-5):
        self.alpha = alpha
        self.wc = wc
        self.s = s
        self.method = method
        self.Q = Q
        self.T = T
        self.Nk = Nk
        if _mpmath_available is False:
            print(
                "The mpmath module is needed for the description"
                " of Ohmic baths")
        if method == "correlation":
            self.fit = CorrelationFitter(self.Q)
            t = np.linspace(0, 15 / self.wc, 1000)
            C = self.correlation(t, self.s)
            bath, _ = self.fit.get_fit(t, C, final_rmse=rmse)
            self.exponents = bath.exponents
        else:
            self.fit = SpectralFitter(self.T, self.Q, self.Nk)
            w = np.linspace(0, 15 * self.wc, 20000)
            J = self.spectral_density(w)
            bath, _ = self.fit.get_fit(J, w, final_rmse=rmse)
            self.terminator = self.fit.terminator
            self.exponents = bath.exponents

    def summary(self):
        self.fit.summary()

    def spectral_density(self, w):
        """The Ohmic bath spectral density as a function of w
        (and the bath parameters).
        """
        return (
            self.alpha
            * w ** (self.s)
            / (self.wc ** (1 - self.s))
            * np.e ** (-abs(w) / self.wc)
        )

    def correlation(self, t, s=1):
        """The Ohmic bath correlation function as a function of t
        (and the bath parameters).
        """
        corr = (
            (1 / np.pi)
            * self.alpha
            * self.wc ** (1 - s)
            * (1/self.T) ** (-(s + 1))
            * mp.gamma(s + 1)
        )
        z1_u = (1 + (1/self.T) * self.wc - 1.0j *
                self.wc * t) / ((1/self.T) * self.wc)
        z2_u = (1 + 1.0j * self.wc * t) / ((1/self.T) * self.wc)
        # Note: the arguments to zeta should be in as high precision
        # as possible.
        # See http://mpmath.org/doc/current/basics.html#providing-correct-input
        return np.array(
            [
                complex(corr * (mp.zeta(s + 1, u1) + mp.zeta(s + 1, u2)))
                for u1, u2 in zip(z1_u, z2_u)
            ],
            dtype=np.complex128,
        )

    def power_spectrum(self, w):
        """The Ohmic bath power spectrum as a function of w
        (and the bath parameters).
        """
        return (
            self.spectral_density(w)
            * ((1 / (np.e ** (w * (1/self.T)) - 1)) + 1)
            * 2
        )

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
        a list containing fitted resonance frequecies
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
