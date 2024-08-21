import scipy as sp
import numpy as np
from scipy.optimize import curve_fit



def _pack(*args):
    """
    Pack parameter lists for fitting. In both use cases (spectral fit,
    correlation fit), the fit parameters are three arrays of equal length.
    """
    return np.concatenate(tuple(args))


def _unpack(params, n=3):
    """
    Unpack parameter lists for fitting. In the use cases (spectral fit/
    correlation fit), the fit parameters are three/four arrays of equal length.
    """
    N = len(params) // n
    zz = []
    for i in range(n):
        zz.append(params[i*N:(i+1)*N])
    return zz


def _leastsq(func, y, x, guesses=None, lower=None,
             upper=None, sigma=None, n=3):
    """
    Performs nonlinear least squares to fit the function func to x and y.

    Parameters
    ----------
    func : function
        The function we wish to fit.
    x : np.array
        a numpy array containing the independent variable used for the fit.
    y : :obj:`np.array.`
        a numpy array containing the dependent variable we use for the fit.
    guesses : list
        Initial guess for the parameters.
    lower : list
        lower bounds on the parameters for the fit.
    upper : list
        upper bounds on the parameters for the fit.
    sigma : float
        uncertainty in the data considered for the fit
    n: int
        number of free parameters to be fitted, used for reshaping of the
        parameters array across the different functions
    Returns
    -------
    params: list
        It returns the fitted parameters.
    """

    sigma = [sigma] * len(x)
    params, _ = curve_fit(
        lambda x, *params: func(x, *_unpack(params, n)),
        x,
        y,
        p0=guesses,
        bounds=(lower, upper),
        sigma=sigma,
        maxfev=int(1e9),
        method="trf",
    )

    return _unpack(params, n)


def _rmse(func, x, y, *args):
    """
    Calculates the normalized root mean squared error for fits
    from the fitted parameters a, b, c.

    Parameters
    ----------
    func : function
        The approximated function for which we want to compute the rmse.
    x: :obj:`np.array.`
        a numpy array containing the independent variable used for the fit.
    y: :obj:`np.array.`
        a numpy array containing the dependent variable used for the fit.
    a, b, c : list
        fitted parameters.

    Returns
    -------
    rmse: float
        The normalized root mean squared error for the fit, the closer
        to zero the better the fit.
    """
    yhat = func(x, *args)
    rmse = np.sqrt(np.mean((yhat - y) ** 2) / len(y)) / \
        (np.max(y) - np.min(y))
    return rmse


def _fit(func, C, t, N, default_guess_scenario='',
         guesses=None, lower=None, upper=None, sigma=None, n=3):
    """
    Performs a fit the function func to t and C, with N number of
    terms in func, the guesses,bounds and uncertainty can be determined
    by the user.If none is provided it constructs default ones according
    to the label.

    Parameters
    ----------
    func : function
        The function we wish to fit.
    C : :obj:`np.array.`
        a numpy array containing the dependent variable used for the fit.
    t : :obj:`np.array.`
        a numpy array containing the independent variable used for the fit.
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
        upper bounds on the parameters for the fit.
    sigma: float
        uncertainty in the data considered for the fit.
    n: int
        The Number of variables used in the fit
    Returns
    -------
    params:
        It returns the fitted parameters as a list.
    rmse:
        It returns the normalized mean squared error from the fit
    """

    if None not in (guesses, lower, upper, sigma):
        guesses = _reformat(guesses, N)
        lower = _reformat(lower, N)
        upper = _reformat(upper, N)
    else:
        tempguess, templower, tempupper, tempsigma = _default_guess_scenarios(
            C, t, default_guess_scenario, N, n)
        guesses = tempguess
        lower = templower
        upper = tempupper
        sigma = tempsigma
        if (tempupper == templower).all() and (tempguess == tempupper).all():
            return 0, _unpack(templower, n)
    if not ((len(guesses) == len(lower)) and (len(guesses) == len(upper))):
        raise ValueError("The shape of the provided fit parameters is \
                         not consistent")
    args = _leastsq(func, C, t, sigma=sigma, guesses=guesses,
                    lower=lower, upper=upper, n=n)
    rmse = _rmse(func, t, C, *args)
    return rmse, args


def _default_guess_scenarios(C, t, default_guess_scenario, N, n):
    C_max = abs(max(C, key=np.abs))
    tempsigma = 1e-2

    if C_max == 0:
        # When the target function is zero
        tempguesses = _pack(
            [0] * N, [0] * N, [0] * N, [0] * N)
        templower = tempguesses
        tempupper = tempguesses
        return tempguesses, templower, tempupper, tempsigma
    wc = t[np.argmax(C)]

    if "correlation" in default_guess_scenario:
        if n == 4:
            templower = _pack([-100*C_max] * N, [-np.inf] * N, [-1]
                              * N, [-100*C_max] * N)
        else:
            templower = _pack([-20 * C_max] * N, [-np.inf] * N, [0.0] * N)

    if default_guess_scenario == "correlation_real":
        if n == 4:
            wc = np.inf
            tempguesses = _pack([C_max] * N, [-100*C_max]
                                * N, [0] * N, [0] * N)
            tempupper = _pack([100*C_max] * N, [0] * N,
                              [1] * N, [100*C_max] * N)
        else:
            tempguesses = _pack([C_max] * N, [-wc] * N, [wc] * N)
            tempupper = _pack([20 * C_max] * N, [0.1] * N, [np.inf] * N)
    elif default_guess_scenario == "correlation_imag":
        if n == 4:
            wc = np.inf
            tempguesses = _pack([0] * N, [-10*C_max] * N, [0] * N, [0] * N)
            tempupper = _pack([100*C_max] * N, [0] * N,
                              [2] * N, [100*C_max] * N)
        else:
            tempguesses = _pack([-C_max] * N, [-10*C_max] * N, [1] * N)
            tempupper = _pack([10 * C_max] * N, [0] * N, [np.inf] * N)
    else:
        tempguesses = _pack([C_max] * N, [wc] * N, [wc] * N)
        templower = _pack([-100 * C_max] * N,
                          [0.1 * wc] * N, [0.1 * wc] * N)
        tempupper = _pack([100 * C_max] * N,
                          [100 * wc] * N, [100 * wc] * N)
    return tempguesses, templower, tempupper, tempsigma


def _reformat(guess, N):
    """
    This function reformats the user provided guesses into the format
    appropiate for fitting, if the user did not provide it the defaults are
    assigned
    """
    guesses = [[i]*N for i in guess]
    guesses = [x for xs in guesses for x in xs]
    guesses = _pack(guesses)
    return guesses


def _run_fit(funcx, y, x, final_rmse, default_guess_scenario='', N=None, n=3,
             **kwargs):
    """
    It iteratively tries to fit the funcx to y on the interval x.
    If N is provided the fit is done with N modes, if it is
    None then this automatically finds the smallest number of modes that
    whose mean squared error is smaller than final_rmse.

    Parameters
    ----------
    funcx : function
        The function we wish to fit.
    y : :obj:`np.array.`
        The function used for the fitting.
    x : :obj:`np.array.`
        a numpy array containing the independent variable used for the fit.
    final_rmse : float
        Desired normalized root mean squared error.
    default_guess_scenario : str
        Determines how the default guesses and bounds are chosen (in the case
        guesses or bounds are not specified). May be 'correlation_real',
        'correlation_imag' or any other string. Any other string will use
        guesses and bounds designed for the fitting of spectral densities.
    N : optional , int
        The number of modes used for the fitting, if not provided starts at
        1 and increases until a desired RMSE is satisfied.
    sigma: float
        uncertainty in the data considered for the fit.
    guesses : list
        Initial guess for the parameters.
    lower : list
        lower bounds on the parameters for the fit.
    upper: list
        upper bounds on the parameters for the fit.

    Returns
    -------
    params:
        It returns the fitted parameters as a list.
    rmse:
        It returns the normalized mean squared error from the fit
    """

    if N is None:
        N = 2
        iterate = True
    else:
        iterate = False
    rmse1 = np.inf

    while rmse1 > final_rmse:
        rmse1, params = _fit(
            funcx, y, x, N, default_guess_scenario, n=n, **kwargs)
        N += 1
        if not iterate:
            break

    return rmse1, params


def _gen_summary(time, rmse, N, label, params,
                 columns=['lam', 'gamma', 'w0']):
    if len(columns) == 3:
        summary = (f"Result of fitting {label} "
                   f"with {N} terms: \n \n {'Parameters': <10}|"
                   f"{columns[0]: ^10}|{columns[1]: ^10}|{columns[2]: >5} \n ")
        for k in range(len(params[0])):
            summary += (
                f"{k+1: <10}|{params[0][k]: ^10.2e}|{params[1][k]:^10.2e}|"
                f"{params[2][k]:>5.2e}\n ")
    else:
        summary = (
            f"Result of fitting {label} "
            f"with {N} terms: \n \n {'Parameters': <10}|"
            f"{columns[0]: ^10}|{columns[1]: ^10}|{columns[2]: ^10}"
            f"|{columns[3]: >5} \n ")
        for k in range(len(params[0])):
            summary += (
                f"{k+1: <10}|{params[0][k]: ^10.2e}|{params[1][k]:^10.2e}"
                f"|{params[2][k]:^10.2e}|{params[3][k]:>5.2e}\n ")
    summary += (f"\nA  normalized RMSE of {rmse: .2e}"
                f" was obtained for the {label}\n")
    summary += f" The current fit took {time: 2f} seconds"
    return summary


def _two_column_summary(
        params_real, params_imag, fit_time_real, fit_time_imag, Nr, Ni,
        rmse_imag, rmse_real, n=3):
    # Generate nicely formatted summary
    columns = ["a", "b", "c"]
    if n == 4:
        columns.append("d")
    summary_real = _gen_summary(
        fit_time_real,
        rmse_real,
        Nr,
        "The Real Part Of  \n the Correlation Function", params_real,
        columns=columns)
    summary_imag = _gen_summary(
        fit_time_imag,
        rmse_imag,
        Ni,
        "The Imaginary Part \n Of the Correlation Function", params_imag,
        columns=columns)

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


def aaa(F, Z, tol=1e-13, mmax=100):

    if ~ (type(F) == np.array):
        F = F(Z)
    M = len(Z)
    J = list(range(0, M))
    z = np.empty(0)
    f = np.empty(0)
    C = []
    errors = []
    R = np.mean(F) * np.ones_like(F)
    for m in range(mmax):
        # find largest residual
        j = np.argmax(abs(F - R))
        z = np.append(z, Z[j])
        f = np.append(f, F[j])
        try:
            J.remove(j)
        except:
            pass

        # Cauchy matrix containing the basis functions as columns
        C = 1.0 / (Z[J, None] - z[None, :])
        # Loewner matrix
        A = (F[J, None] - f[None, :]) * C

        # compute weights as right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(A)
        wj = Vh[-1, :].conj()

        # approximation: numerator / denominator
        N = C.dot(wj * f)
        D = C.dot(wj)

        # update residual
        R = F.copy()
        R[J] = N / D

        # check for convergence
        errors.append(np.linalg.norm(F - R, np.inf))
        if errors[-1] <= tol * np.linalg.norm(F, np.inf):
            break

    def r(x): return approximated_function(x, z, f, wj)
    # return z,f,wj
    pol, res, zer = prz(z, f, wj)
    return r, pol, res, zer


def approximated_function(zz, z, f, w, need=False):
    # evaluate r at zz
    zv = np.ravel(zz)  # vectorize zz

    # Cauchy matrix
    CC = 1 / (np.subtract.outer(zv, z))

    # AAA approximation as vector
    r = np.dot(CC, w * f) / np.dot(CC, w)
    if need is True:
        return np.dot(CC, w * f), np.dot(CC, w * f)
    # Find values NaN = Inf/Inf if any
    ii = np.isnan(r)

    # Force interpolation at NaN points
    for j in np.where(ii)[0]:
        r[j] = f[np.where(zv[j] == z)[0][0]]

    # Reshape the result to match the shape of zz
    r = r.reshape(zz.shape)
    return r


def prz(z, f, w):
    m = len(w)
    B = np.eye(m+1)
    B[0, 0] = 0
    E = np.block([[0, w], [np.ones((m, 1)), np.diag(z)]])
    eigvals = sp.linalg.eig(E, B)[0]
    # eigvals[~np.isinf(eigvals)] #remove singularities
    pol = np.real_if_close(eigvals[np.isfinite(eigvals)])
    # Get residues from quotients, in the paper they use a little shift
    # but I coudn't broadcast it correctly
    C = 1.0/(pol[:, None]-z[None, :])
    N = C.dot(f*w)
    # Derivative, formula for simple poles see Zill complex analysis
    D = (-C**2).dot(w)
    res = N/D
    ez = np.block([[0, w], [f[:, None], np.diag(z)]])
    eigvals_zeros = sp.linalg.eig(ez, B)[0]
    zer = eigvals_zeros[~np.isinf(eigvals_zeros)]
    return pol, res, zer


def filter_poles(pol, res):
    pols = []
    ress = []
    for i in range(len(pol)):
        if (np.imag(pol[i]) < 0):
            pols.append(pol[i])
            ress.append(res[i])
    return np.array(pols), np.array(ress)
