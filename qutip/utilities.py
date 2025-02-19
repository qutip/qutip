"""
This module contains utility functions that are commonly needed in other
qutip modules.
"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['n_thermal', 'clebsch', 'convert_unit', 'iterated_fit']

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.linalg import qr, hankel, lstsq, eigvals, svd, eig
from scipy.fft import fft


def n_thermal(w, w_th):
    """
    Return the number of photons in thermal equilibrium for an harmonic
    oscillator mode with frequency 'w', at the temperature described by
    'w_th' where :math:`\\omega_{\\rm th} = k_BT/\\hbar`.

    Parameters
    ----------

    w : float or ndarray
        Frequency of the oscillator.

    w_th : float
        The temperature in units of frequency (or the same units as `w`).


    Returns
    -------

    n_avg : float or array

        Return the number of average photons in thermal equilibrium for a
        an oscillator with the given frequency and temperature.


    """

    w = np.array(w, dtype=float)
    result = np.zeros_like(w)

    if w_th <= 0:
        result[w < 0] = -1
        return result.item() if w.ndim == 0 else result

    non_zero = w != 0
    result[non_zero] = 1 / (np.exp(w[non_zero] / w_th) - 1)

    return result.item() if w.ndim == 0 else result


def _factorial_prod(N, arr):
    arr[:int(N)] += 1


def _factorial_div(N, arr):
    arr[:int(N)] -= 1


def _to_long(arr):
    prod = 1
    for i, v in enumerate(arr):
        prod *= (i + 1)**int(v)
    return prod


def clebsch(j1, j2, j3, m1, m2, m3):
    """Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).

    Parameters
    ----------
    j1 : float
        Total angular momentum 1.

    j2 : float
        Total angular momentum 2.

    j3 : float
        Total angular momentum 3.

    m1 : float
        z-component of angular momentum 1.

    m2 : float
        z-component of angular momentum 2.

    m3 : float
        z-component of angular momentum 3.

    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.

    """
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    c_factor = np.zeros((int(j1 + j2 + j3 + 1)), np.int32)
    _factorial_prod(j3 + j1 - j2, c_factor)
    _factorial_prod(j3 - j1 + j2, c_factor)
    _factorial_prod(j1 + j2 - j3, c_factor)
    _factorial_prod(j3 + m3, c_factor)
    _factorial_prod(j3 - m3, c_factor)
    _factorial_div(j1 + j2 + j3 + 1, c_factor)
    _factorial_div(j1 - m1, c_factor)
    _factorial_div(j1 + m1, c_factor)
    _factorial_div(j2 - m2, c_factor)
    _factorial_div(j2 + m2, c_factor)
    C = np.sqrt((2.0 * j3 + 1.0) * _to_long(c_factor))

    s_factors = np.zeros(((vmax + 1 - vmin), (int(j1 + j2 + j3))), np.int32)
    # `S` and `C` are large integer,s if `sign` is a np.int32 it could oveflow
    sign = int((-1) ** (vmin + j2 + m2))
    for i, v in enumerate(range(vmin, vmax + 1)):
        factor = s_factors[i, :]
        _factorial_prod(j2 + j3 + m1 - v, factor)
        _factorial_prod(j1 - m1 + v, factor)
        _factorial_div(j3 - j1 + j2 - v, factor)
        _factorial_div(j3 + m3 - v, factor)
        _factorial_div(v + j1 - j2 - m3, factor)
        _factorial_div(v, factor)
    common_denominator = -np.min(s_factors, axis=0)
    numerators = s_factors + common_denominator
    S = sum([(-1)**i * _to_long(vec) for i, vec in enumerate(numerators)]) * \
        sign / _to_long(common_denominator)
    return C * S


# -----------------------------------------------------------------------------
# Functions for unit conversions
#
_e = 1.602176565e-19  # C
_kB = 1.3806488e-23   # J/K
_h = 6.62606957e-34   # Js

_unit_factor_tbl = {
    #   "unit": "factor that convert argument from unit 'unit' to Joule"
    "J": 1.0,
    "eV": _e,
    "meV": 1.0e-3 * _e,
    "GHz": 1.0e9 * _h,
    "mK": 1.0e-3 * _kB,
}


def convert_unit(value, orig="meV", to="GHz"):
    """
    Convert an energy from unit `orig` to unit `to`.

    Parameters
    ----------
    value : float / array
        The energy in the old unit.

    orig : str, {"J", "eV", "meV", "GHz", "mK"}, default: "meV"
        The name of the original unit.

    to : str, {"J", "eV", "meV", "GHz", "mK"}, default: "GHz"
        The name of the new unit.

    Returns
    -------
    value_new_unit : float / array
        The energy in the new unit.
    """
    if orig not in _unit_factor_tbl:
        raise TypeError("Unsupported unit %s" % orig)

    if to not in _unit_factor_tbl:
        raise TypeError("Unsupported unit %s" % to)

    return value * (_unit_factor_tbl[orig] / _unit_factor_tbl[to])


def convert_GHz_to_meV(w):
    """
    Convert an energy from unit GHz to unit meV.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.
    """
    # 1 GHz = 4.1357e-6 eV = 4.1357e-3 meV
    w_meV = w * 4.1357e-3
    return w_meV


def convert_meV_to_GHz(w):
    """
    Convert an energy from unit meV to unit GHz.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.
    """
    # 1 meV = 1.0/4.1357e-3 GHz
    w_GHz = w / 4.1357e-3
    return w_GHz


def convert_J_to_meV(w):
    """
    Convert an energy from unit J to unit meV.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.
    """
    # 1 eV = 1.602e-19 J
    w_meV = 1000.0 * w / _e
    return w_meV


def convert_meV_to_J(w):
    """
    Convert an energy from unit meV to unit J.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.
    """
    # 1 eV = 1.602e-19 J
    w_J = 0.001 * w * _e
    return w_J


def convert_meV_to_mK(w):
    """
    Convert an energy from unit meV to unit mK.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.
    """
    # 1 mK = 0.0000861740 meV
    w_mK = w / 0.0000861740
    return w_mK


def convert_mK_to_meV(w):
    """
    Convert an energy from unit mK to unit meV.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.
    """
    # 1 mK = 0.0000861740 meV
    w_meV = w * 0.0000861740
    return w_meV


def convert_GHz_to_mK(w):
    """
    Convert an energy from unit GHz to unit mK.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.
    """
    # h v [Hz] = kB T [K]
    # h 1e9 v [GHz] = kB 1e-3 T [mK]
    # T [mK] = 1e12 * (h/kB) * v [GHz]
    w_mK = w * 1.0e12 * (_h / _kB)
    return w_mK


def convert_mK_to_GHz(w):
    """
    Convert an energy from unit mK to unit GHz.

    Parameters
    ----------
    w : float / array
        The energy in the old unit.

    Returns
    -------
    w_new_unit : float / array
        The energy in the new unit.

    """
    w_GHz = w * 1.0e-12 * (_kB / _h)
    return w_GHz


def _version2int(version_string):
    str_list = version_string.split(
        "-dev")[0].split("rc")[0].split("a")[0].split("b")[0].split(
        "post")[0].split('.')
    return sum([int(d if len(d) > 0 else 0) * (100 ** (3 - n))
                for n, d in enumerate(str_list[:3])])


# -----------------------------------------------------------------------------
# Fitting utilities
#

def iterated_fit(
    fun: Callable[..., complex],
    num_params: int,
    xdata: ArrayLike,
    ydata: ArrayLike,
    target_rmse: float = 1e-5,
    Nmin: int = 1,
    Nmax: int = 10,
    guess: ArrayLike | Callable[[int], ArrayLike] = None,
    lower: ArrayLike = None,
    upper: ArrayLike = None,
    sigma: float | ArrayLike = None,
    maxfev: int = None
) -> tuple[float, ArrayLike]:
    r"""
    Iteratively tries to fit the given data with a model of the form

    .. math::
        y = \sum_{k=1}^N f(x; p_{k,1}, \dots, p_{k,n})

    where `f` is a model function depending on `n` parameters, and the number
    `N` of terms is increased until the normalized rmse (root mean square
    error) falls below the target value.

    Parameters
    ----------
    fun : callable
        The model function. Its first argument is the array `xdata`, its other
        arguments are the fitting parameters.
    num_params : int
        The number of fitting parameters per term (`n` in the equation above).
        The function `fun` must take `num_params+1` arguments.
    xdata : array_like
        The independent variable.
    ydata : array_like
        The dependent data.
    target_rmse : optional, float
        Desired normalized root mean squared error (default `1e-5`).
    Nmin : optional, int
        The minimum number of terms to be used for the fit (default 1).
    Nmax : optional, int
        The maximum number of terms to be used for the fit (default 10).
        If the number `Nmax` of terms is reached, the function returns even if
        the target rmse has not been reached yet.
    guess : optional, array_like or callable
        This can be either a list of length `n`, with the i-th entry being the
        guess for the parameter :math:`p_{k,i}` (for all terms :math:`k`), or a
        function that provides different initial guesses for each term.
        Specifically, given a number `N` of terms, the function returns an
        array `[[p11, ..., p1n], [p21, ..., p2n], ..., [pN1, ..., pNn]]` of
        initial guesses.
    lower : optional, list of length `num_params`
        Lower bounds on the parameters for the fit.
    upper : optional, list of length `num_params`
        Upper bounds on the parameters for the fit.
    sigma : optional, float or array_like
        The uncertainty in the dependent data, see the documentation of
        ``scipy.optimize.curve_fit``.
    maxfev : optional, int
        The maximum number of function evaluations (per value of ``N``).

    Returns
    -------
    rmse : float
        The normalized mean squared error of the fit
    params : array_like
        The model parameters in the form
        `[[p11, ..., p1n], [p21, ..., p2n], ..., [pN1, ..., pNn]]`.
    """

    if len(xdata) != len(ydata):
        raise ValueError(
            "The shape of the provided fit data is not consistent")

    if lower is None:
        lower = np.full(num_params, -np.inf)
    if upper is None:
        upper = np.full(num_params, np.inf)
    if not (len(lower) == num_params and len(upper) == num_params):
        raise ValueError(
            "The shape of the provided fit bounds is not consistent")

    N = Nmin
    rmse1 = np.inf

    while rmse1 > target_rmse and N <= Nmax:
        if guess is None:
            guesses = np.ones((N, num_params), dtype=float)
        elif callable(guess):
            guesses = np.array(guess(N))
            if guesses.shape != (N, num_params):
                raise ValueError(
                    "The shape of the provided fit guesses is not consistent")
        else:
            guesses = np.tile(guess, (N, 1))

        lower_repeat = np.tile(lower, N)
        upper_repeat = np.tile(upper, N)
        rmse1, params, r2 = _fit(fun, num_params, xdata, ydata,
                                 guesses, lower_repeat,
                                 upper_repeat, sigma, maxfev)
        N += 1

    return rmse1, params, r2


def _pack(params):
    # Pack parameter lists for fitting.
    # Input: array of parameters like `[[p11, ..., p1n], ..., [pN1, ..., pNn]]`
    # Output: packed parameters like `[p11, ..., p1n, p21, ..., p2n, ...]`
    return params.ravel()  # like flatten, but doesn't copy data


def _unpack(params, num_params):
    # Inverse of _pack, `num_params` is "n"
    N = len(params) // num_params
    return np.reshape(params, (N, num_params))


def _evaluate(fun, x, params):
    result = 0
    for term_params in params:
        result += fun(x, *term_params)
    return result


def _rmse(fun, xdata, ydata, params):
    """
    The normalized root mean squared error for the fit with the given
    parameters. (The closer to zero = the better the fit.)
    """
    if params is None:
        yhat = fun
    else:
        yhat = _evaluate(fun, xdata, params)

    if (yhat == ydata).all():
        return 0
    return (
        np.sqrt(np.mean((yhat - ydata) ** 2) / len(ydata))
        / (np.max(ydata) - np.min(ydata))
    )


def _r2(fun, xdata, ydata, params):
    """
    The 1-r2 coefficient serves to evaluate the goodness of fit 
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    it normally ranges from zero to one the closer to one the better. 
    if negative it means the model is worst than the worse possible least 
    squares predictor (basically a line over the mean of the signal)
    1-r2 is chosen instead of r2 because fits using our methods are typically 
    good, so it is hard to show in summary as everything is almost one, so the
    summary shows 1. this way it shows small numbers and fits can be compared 
    easily
    """
    if params is None:
        yhat = fun
    else:
        yhat = _evaluate(fun, xdata, params)

    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    ss_res = np.sum((ydata - yhat) ** 2)

    if ss_tot == 0:
        return 1 if ss_res == 0 else 0  # Handle constant ydata case

    return (ss_res / ss_tot)


def _fit(fun, num_params, xdata, ydata, guesses, lower, upper, sigma,
         maxfev, method='trf'):
    # fun: model function
    # num_params: number of parameters in fun
    # xdata, ydata: data to be fit
    # N: number of terms
    # guesses: initial guesses [[p11, ..., p1n],..., [pN1, ..., pNn]]
    # lower, upper: parameter bounds
    # sigma: data uncertainty (useful to control when values are small)
    # maxfev: how many times the parameters can be altered, lower is faster but
    #         less accurate
    if (upper <= lower).all():
        return _rmse(fun, xdata, ydata, guesses), guesses

    # Depending on the method, scipy uses leastsq or least_squares, and the
    # `maxfev` parameter has different names in the two functions
    if method == 'lm':
        maxfev_arg = {'maxfev': maxfev}
    else:
        maxfev_arg = {'max_nfev': maxfev}

    # Scipy only supports scalar sigma since 1.12
    if sigma is not None and not hasattr(sigma, "__len__"):
        sigma = [sigma] * len(xdata)

    packed_params, _ = curve_fit(
        lambda x, *packed_params: _evaluate(
            fun, x, _unpack(packed_params, num_params)
        ),
        xdata, ydata, p0=_pack(guesses), bounds=(lower, upper),
        method=method, sigma=sigma, **maxfev_arg
    )
    params = _unpack(packed_params, num_params)
    rmse = _rmse(fun, xdata, ydata, params)
    r2 = _r2(fun, xdata, ydata, params)
    return rmse, params, r2


# AAA Fitting

def aaa(func, z, tol=1e-13, max_iter=100):
    """
    Computes a rational approximation of the function according to the AAA
    algorithm as explained in https://doi.org/10.1137/16M1106122 . This
    implementation is a python adaptation of the matlab version in that paper
    NOTE: I am not sure if this is necessary anymore as scipy 1.15 includes AAA

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
        j = np.argmax(np.abs(func - rational_approx))  # next support index
        support_points = np.append(support_points, z[j])
        values = np.append(values, func[j])  # function evaluated at support
        indices = indices[indices != j]  # Remaining indices

        cauchy = compute_cauchy_matrix(z[indices], support_points)
        # Then we construct the Loewner Matrix
        loewner = np.subtract.outer(func[indices], values) * cauchy
        # Perform SVD only d is needed
        _, _, vh = svd(loewner)
        # Compute weights
        weights = vh[-1, :].conj()
        # Obtain the rational Approximation of the function with these support
        # points
        rational_approx = get_rational_approx(
            cauchy, weights, values, indices, func)
        errors[k] = np.linalg.norm(
            func - rational_approx, np.inf)  # compute error
        if errors[k] <= tol * np.linalg.norm(func, np.inf):
            # if contributions are smaller than the tolerance, then stop the
            # loop
            break
    # Define the function approximation as a callable for output

    def r(z):
        cauchy = compute_cauchy_matrix(z, support_points)
        r = get_rational_approx(cauchy, weights, values)
        return r.reshape(z.shape)
    # Obtain poles residies and zeros
    pol, res, zer = prz(support_points, values, weights)
    rmse = _rmse(r(z), z, func, None)
    r2 = _r2(r(z), z, func, None)

    return {
        "function": r,
        "poles": pol,
        "residues": res,
        "zeros": zer,
        "errors": errors[:k + 1],
        "rmse": rmse,
        "r2": r2,
        "support points": support_points,
        "values at support": values,
        "indices": indices,
    }


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
        support points y

    Returns:
    --------
    cauchy : np.ndarray
        The cauchy matrix from the sample and support points
    """
    epsilon = 1e-15  # Small constant to avoid division by zero
    # Prevent division by zero on poles
    denominator = np.subtract.outer(z, support_points)
    denominator[denominator == 0] = epsilon
    cauchy = 1 / denominator
    return cauchy


def get_rational_approx(cauchy, weights, values, indices=None, func=None):
    """
    Gets the rational approximation of the function. The approximation is of
    the form

    ..math::
        r(z) = \frac{w_{j} f_{j}}{z-z_{j}}/\frac{w_{j}}{z-z_{j}}

    where w is the cauchy matrix
    Parameters:
    -----------
    cauchy : np.ndarray
        The cauchy matrix
    values : np.ndarray
        The data used for the approximation
    weights : np.ndarray
        The weights used for the approximation
    indices: np.ndarray
        The support points to be avoided
    func:
        The function evaluated in the range of the fit


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
        # This bit is to Avoid the support points in the AAA iterations
        # as they shouldn't be included
        rational_approx = func.copy()
        rational_approx[indices] = numerator / denominator
    return rational_approx


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

    where B is like a mxm identity matrix, except its first element is 0.

    Unlike the implementation in the reference we use the simple quotient
    formula for the residue (https://math.stackexchange.com/questions/2202129/
    residue-for-quotient-of-functions)


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
    geye = np.eye(m + 1)
    geye[0, 0] = 0
    geig = np.block([[0, weights], [np.ones((m, 1)), np.diag(support_points)]])
    eigvals = eig(geig, geye)[0]
    # removing spurious values
    pol = np.real_if_close(eigvals[np.isfinite(eigvals)])

    cauchy = compute_cauchy_matrix(pol, support_points)

    numerator = cauchy @ (values * weights)
    denominator = (-cauchy**2) @ weights  # Quotient rule f=1/cauchy
    res = numerator / denominator
    ez = np.block([[0, weights], [values[:, None], np.diag(support_points)]])
    zeros = eig(ez, geye)[0]
    zeros = zeros[~np.isinf(zeros)]
    return pol, res, zeros


def prony_model(orig, amp, phase):
    # It serves to compute rmse
    return amp * np.power(phase, np.arange(len(orig)))


def prony_methods(method: str, C: np.ndarray, n: int):
    num_freqs = len(C) - n
    hankel0 = hankel(c=C[:num_freqs], r=C[num_freqs - 1: -1])
    hankel1 = hankel(c=C[1: num_freqs + 1], r=C[num_freqs:])
    if method == "mp":
        _, R = qr(hankel0)
        pencil_matrix = np.linalg.pinv(R.T @ hankel0) @ (R.T @ hankel1)
        phases = eigvals(pencil_matrix)
    elif method == "prony":
        shift_matrix = lstsq(hankel0.T, hankel1.T)[0]
        phases = eigvals(shift_matrix.T)
    elif method == "esprit":
        U1, _, _ = svd(hankel0)
        pencil_matrix = np.linalg.pinv(U1.T @ hankel0) @ (U1.T @ hankel1)
        phases = eigvals(pencil_matrix)
    generation_matrix = np.array(
        [[phase**k for phase in phases] for k in range(len(C))])
    amplitudes = lstsq(generation_matrix, C)[0]
    params = _unpack(
        np.array([val for pair in zip(amplitudes, phases) for val in pair]), 2)

    rmse = _rmse(prony_model, C, C, params)
    r2 = _r2(prony_model, C, C, params)
    return params, rmse, r2


# def matrix_pencil(C: np.ndarray, n: int) -> tuple:
#     """
#     Estimate amplitudes and frequencies using the Matrix Pencil Method.
#     Based on the description in https://doi.org/10.1093/imanum/drab108

#     Args:
#         signal (np.ndarray): The input signal (1D complex array).
#         n (int): The number of modes to estimate (rank of the signal).

#     Returns:
#         tuple: A tuple containing:
#             - amplitudes (np.ndarray):
#                 The estimated amplitudes.
#             - phases (np.ndarray):
#                 The estimated complex exponential frequencies.
#     """
#     num_freqs = len(C) - n
#     hankel0 = hankel(c=C[:num_freqs], r=C[num_freqs - 1: -1])
#     hankel1 = hankel(c=C[1: num_freqs + 1], r=C[num_freqs:])

#     _, R = qr(hankel0)

#     pencil_matrix = np.linalg.pinv(R.T @ hankel0) @ (R.T @ hankel1)
#     phases = eigvals(pencil_matrix)
#     generation_matrix = np.array(
#         [[phase**k for phase in phases] for k in range(len(C))])
#     amplitudes = lstsq(generation_matrix, C)[0]
#     params = _unpack(
#         np.array([val for pair in zip(amplitudes, phases) for val in pair]), 2)

#     rmse = _rmse(prony_model, C, C, params)
#     r2 = _r2(prony_model, C, C, params)
#     return params, rmse, r2


# def prony(signal: np.ndarray, n):
#     """
#     Estimate amplitudes and frequencies using the prony Method.
#     Based on the description in https://doi.org/10.1093/imanum/drab108

#     Args:
#         signal (np.ndarray): The input signal (1D complex array).
#         t (np.ndarray): The points where the signal was sampled
#         n (int): The number of modes to estimate (rank of the signal).

#     Returns:
#         tuple: A tuple containing:
#             - amplitudes (np.ndarray):
#                 The estimated amplitudes.
#             - phases (np.ndarray):
#                 The estimated complex exponential frequencies.
#             - Normalized Root Mean Squared Error (float)
#                 The error commited by approximating the signal
#     """
#     num_freqs = n
#     hankel0 = hankel(c=signal[:num_freqs], r=signal[num_freqs - 1: -1])
#     hankel1 = hankel(c=signal[1: num_freqs + 1], r=signal[num_freqs:])
#     shift_matrix = lstsq(hankel0.T, hankel1.T)[0]
#     phases = eigvals(shift_matrix.T)

#     generation_matrix = np.array(
#         [[phase**k for phase in phases] for k in range(len(signal))])
#     amplitudes = lstsq(generation_matrix, signal)[0]

#     amplitudes, phases = zip(
#         *sorted(zip(amplitudes, phases), key=lambda x: np.abs(x[0]),
#                 reverse=True)
#     )
#     params = _unpack(
#         np.array([val for pair in zip(amplitudes, phases) for val in pair]), 2)

#     rmse = _rmse(prony_model, signal, signal, params)
#     r2 = _r2(prony_model, signal, signal, params)

#     return params, rmse, r2


# def esprit(C: np.ndarray, n: int) -> tuple:
#     """
#     Estimate amplitudes and frequencies using the ESPRIT Method.
#     Based on the description in https://doi.org/10.1093/imanum/drab108

#     Args:
#         signal (np.ndarray): The input signal (1D complex array).
#         n (int): The number of modes to estimate (rank of the signal).

#     Returns:
#         tuple: A tuple containing:
#             - amplitudes (np.ndarray):
#                 The estimated amplitudes.
#             - phases (np.ndarray):
#                 The estimated complex exponential frequencies.
#     """
#     # Step 1: Create the Hankel matrices
#     num_freqs = len(C) - n
#     hankel0 = hankel(c=C[:num_freqs], r=C[num_freqs - 1: -1])
#     hankel1 = hankel(c=C[1: num_freqs + 1], r=C[num_freqs:])

#     # Step 2: Perform SVD on the first Hankel matrix
#     U1, _, _ = svd(hankel0)
#     pencil_matrix = np.linalg.pinv(U1.T @ hankel0) @ (U1.T @ hankel1)
#     phases = eigvals(pencil_matrix)
#     generation_matrix = np.array(
#         [[phase**k for phase in phases] for k in range(len(C))])
#     amplitudes = lstsq(generation_matrix, C)[0]
#     params = _unpack(
#         np.array([val for pair in zip(amplitudes, phases) for val in pair]), 2)

#     rmse = _rmse(prony_model, C, C, params)
#     r2 = _r2(prony_model, C, C, params)

#     return params, rmse, r2

# ESPIRA I


def espira1(y, Nexp, tol=1e-16):
    # Compute FFT
    F = fft(y)
    M = len(F)  # number of modified DFT values

    # Set knots on the unit circle
    Z = np.exp(2j * np.pi * np.arange(M) / M)
    # Modify the DFT values
    F = F * Z**(-1)
    result = aaa(F, Z, max_iter=Nexp+1, tol=tol)
    CC = (-1) / np.subtract.outer(result['support points'], result['poles'])
    AB, _, _, _ = np.linalg.lstsq(
        CC, result['values at support'], rcond=None)  # Least squares solution
    g = -AB / (1 - result['poles']**M)  # Element-wise division
    params = _unpack(
        np.array([val for pair in zip(g, result['poles']) for val in pair]), 2)
    rmse = _rmse(prony_model, y, y, params)
    r2 = _r2(prony_model, y, y, params)

    return params, rmse, r2
