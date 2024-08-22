"""
Set of utility functions to obtain a decaying exponential representation of
correlation functions via fitting
"""
import scipy.linalg as sp
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
    num_params = len(params) // n
    zz = []
    for i in range(n):
        zz.append(params[i*num_params:(i+1)*num_params])
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


def _fit(func, corr, t, N, default_guess_scenario='',
         guesses=None, lower=None, upper=None, sigma=None, n=3):
    """
    Performs a fit the function func to t and corr, with N number of
    terms in func, the guesses,bounds and uncertainty can be determined
    by the user.If none is provided it constructs default ones according
    to the label.

    Parameters
    ----------
    func : function
        The function we wish to fit.
    corr : :obj:`np.array.`
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
            corr, t, default_guess_scenario, N, n)
        guesses = tempguess
        lower = templower
        upper = tempupper
        sigma = tempsigma
        if (tempupper == templower).all() and (tempguess == tempupper).all():
            return 0, _unpack(templower, n)
    if not ((len(guesses) == len(lower)) and (len(guesses) == len(upper))):
        raise ValueError("The shape of the provided fit parameters is \
                         not consistent")
    args = _leastsq(func, corr, t, sigma=sigma, guesses=guesses,
                    lower=lower, upper=upper, n=n)
    rmse = _rmse(func, t, corr, *args)
    return rmse, args


def _default_guess_scenarios(corr, t, default_guess_scenario, N, n):
    corr_max = abs(max(corr, key=np.abs))
    tempsigma = 1e-2

    if corr_max == 0:
        # When the target function is zero
        tempguesses = _pack(
            [0] * N, [0] * N, [0] * N, [0] * N)
        templower = tempguesses
        tempupper = tempguesses
        return tempguesses, templower, tempupper, tempsigma
    wc = t[np.argmax(corr)]

    if "correlation" in default_guess_scenario:
        if n == 4:
            templower = _pack([-100*corr_max] * N, [-np.inf] * N, [-1]
                              * N, [-100*corr_max] * N)
        else:
            templower = _pack([-20 * corr_max] * N, [-np.inf] * N, [0.0] * N)

    if default_guess_scenario == "correlation_real":
        if n == 4:
            wc = np.inf
            tempguesses = _pack([corr_max] * N, [-100*corr_max]
                                * N, [0] * N, [0] * N)
            tempupper = _pack([100*corr_max] * N, [0] * N,
                              [1] * N, [100*corr_max] * N)
        else:
            tempguesses = _pack([corr_max] * N, [-wc] * N, [wc] * N)
            tempupper = _pack([20 * corr_max] * N, [0.1] * N, [np.inf] * N)
    elif default_guess_scenario == "correlation_imag":
        if n == 4:
            wc = np.inf
            tempguesses = _pack([0] * N, [-10*corr_max] * N, [0] * N, [0] * N)
            tempupper = _pack([100*corr_max] * N, [0] * N,
                              [2] * N, [100*corr_max] * N)
        else:
            tempguesses = _pack([-corr_max] * N, [-10*corr_max] * N, [1] * N)
            tempupper = _pack([10 * corr_max] * N, [0] * N, [np.inf] * N)
    else:
        tempguesses = _pack([corr_max] * N, [wc] * N, [wc] * N)
        templower = _pack([-100 * corr_max] * N,
                          [0.1 * wc] * N, [0.1 * wc] * N)
        tempupper = _pack([100 * corr_max] * N,
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
    """Generates a summary of fits by nonlinear least squares"""
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
    # Generate nicely formatted summary with two columns for correlations
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


def aaa(func, z, tol=1e-13, max_iter=100):
    """
    Computes a rational approximation of the function according to the AAA 
    algorithm as explained in https://doi.org/10.1137/16M1106122 . This
    implementation is a python translation of the one in that paper

    Parameters:
    -----------
    func : callable or np.ndarray

    z : np.ndarray
        The sampling points on which to perform the rational approximation. 
        Even though linearly spaced sample points will yield good 
        approximations, logarithmicly spaced points will usually give better
        exponent approximations

    tol : float, optional
        Relative tolerance of the approximation
    max_iter : int, optional
        Maximum number of support points ~2*n where n is the number of bath 
        exponents

    Returns:
    --------
    r : callable
        rational approximation of the function
    pol : np.ndarray
        poles of the approximant function
    res : np.ndarray
        residues of the approximant function
    zer : np.ndarray
        zeros of the approximant function
    errors : np.ndarray
        Error by iteration
    """
    func = func(z) if callable(func) else func
    indices = np.arange(len(z))
    support_points = np.empty(0, dtype=z.dtype)
    values = np.empty(0, dtype=func.dtype)
    errors = np.zeros(max_iter)
    rational_approx = np.full_like(func, np.mean(func))

    for k in range(max_iter):
        j = np.argmax(np.abs(func - rational_approx))  # next support time
        support_points = np.append(support_points, z[j])
        values = np.append(values, func[j])
        indices = indices[indices != j]

        cauchy = compute_cauchy_matrix(z[indices], support_points)
        loewner = np.subtract.outer(func[indices], values) * cauchy
        _, _, vh = np.linalg.svd(loewner)
        weights = vh[-1, :].conj()
        rational_approx = get_rational_approx(
            cauchy, weights, values, indices, func)
        errors[k] = np.linalg.norm(
            func - rational_approx, np.inf)  # compute error
        if errors[k] <= tol * np.linalg.norm(func, np.inf):
            break

    def r(x):
        return approximated_function(x, support_points, values, weights)
    pol, res, zer = prz(support_points, values, weights)
    return r, pol, res, zer, errors[:k+1]


def compute_cauchy_matrix(z, support_points):
    r"""
    Computes the `Cauchy matrix <https://en.wikipedia.org/wiki/Cauchy_matrix>`
    for the AAA rational approximation

    ..math::
    a_{ij}={\frac {1}{x_{i}-y_{j}}};\quad x_{i}-y_{j}\neq 0
    ,\quad 1\leq i\leq m,\quad 1\leq j\leq n}

    Parameters:
    -----------
    z : np.ndarray
        sample points x
    support_points : np.ndarray
        suuport points y 

    Returns:
    --------
    cauchy : np.ndarray
        The cauchy matrix from the sample and support points
    """
    return 1 / np.subtract.outer(z, support_points)


def get_rational_approx(cauchy, weights, values, indices=None, func=None):
    """
    Gets the rational approximation of the function. The approximation is of 
    the form 

    ..math::
        r(z) = \frac{w_{j} f_{j}}{z-z_{j}}/\frac{w_{j}}{z-z_{j}}
    Parameters:
    -----------
    cauchy : np.ndarray
        The cauchy matrix
    values : np.ndarray
        The data used for the approximation
    weights : np.ndarray
        The weights used for the approximation

    Returns:
    --------
    r : np.ndarray
        The rational approximation of the function
    """

    numerator = cauchy @ (weights * values)
    denominator = cauchy @ weights
    if func is None:
        rational_approx = numerator / denominator
    else:
        rational_approx = func.copy()
        rational_approx[indices] = numerator / denominator
    return rational_approx


def approximated_function(z, support_points, values, weights):
    """
    It computes the rational approximation 
    ..math::
        r(z) = \frac{w_{j} f_{j}}{z-z_{j}}/\frac{w_{j}}{z-z_{j}}
    and interpolates its poles naively
    Parameters:
    -----------
    z : np.ndarray
        sample points for the approximation
    support_points : np.ndarray
        the support points for the cauchy matrix
    values : np.ndarray
        the data use for the approximation
    weights : np.ndarray
        the weight vector w

    Returns:
    --------
    r : np.ndarray
        The rational approximation of the function smoothed out
    """
    zv = np.ravel(z)  # flatten with c order
    cauchy = compute_cauchy_matrix(z, support_points)
    r = get_rational_approx(cauchy, weights, values)
    mask = np.isnan(r)  # removing the nans in the poles
    if np.any(mask):
        closest_indices = np.argmin(
            np.abs(zv[mask, None] - support_points), axis=1)
        r[mask] = values[closest_indices]

    return r.reshape(z.shape)


def prz(support_points, values, weights):
    r"""
    prz stands for poles, residues and zeros. It calculates and returns the
    poles, residue and zeros of the rational approximation. Using the
    generalized eigenvalue problem

    ..math::
       geig = \begin{pmatrix}0 & \omega_{2} & \dots& \omega_{m} \\
           1& z_{1} & 0 & \dots \\
           1 & 0 & z_{2} & \dots \\
            \vdots   & \vdots & \vdots & \vdots \\
             1   & \dots  & \dots &z_{m}\end{pmatrix} = \lambda L

    where B is like a mxm identity matrix, except its first element is 0

    Parameters:
    -----------
    support_points : np.ndarray
        The support points of the rational approximation
    values : np.ndarray
        Data values on which the approximation is performed
    weights :np.ndarray
        The weight vector

    Returns:
    --------
    pol : np.ndarray
        The poles of the rational approximation
    res : np.ndarray
        The residues of the rational approximation
    zer : np.ndarray
        The zeros of the rational approximation
    """
    m = len(weights)
    geye = np.eye(m+1)
    geye[0, 0] = 0
    geig = np.block([[0, weights], [np.ones((m, 1)), np.diag(support_points)]])
    eigvals = sp.linalg.eig(geig, geye)[0]
    # removing spurious values
    pol = np.real_if_close(eigvals[np.isfinite(eigvals)])

    cauchy = compute_cauchy_matrix(pol, support_points)
    # this is not the same formula they use, instead of a
    # phase perturbation I use  1/cauchy' weights (simple quotient formula)
    # for the residue

    numerator = cauchy @ (values * weights)
    denominator = (-cauchy**2) @ weights
    res = numerator / denominator
    ez = np.block([[0, weights], [values[:, None], np.diag(support_points)]])
    zeros = sp.linalg.eig(ez, geye)[0]
    zeros = zeros[~np.isinf(zeros)]
    return pol, res, zeros


def filter_poles(pol, res):
    """
    The rational approximation gives poles both in the upper and lower 
    plane, which translates into poles describing :math:`C(t)` and
    :math:`C(-t)`. We filter the poles to get the ones for :math:`C(t)`

    Parameters:
    -----------
    pol : np.ndarray
        poles obtained from calling aaa
    res : np.ndarray
        residues obtained from calling aaa

    Returns:
    --------
    filtered_pol : np.ndarray
        The poles in the lower half plane
    filtered_res : np.ndarray
        The residues in the lower half plane
    """
    mask = np.imag(pol) < 0
    return pol[mask], res[mask]
