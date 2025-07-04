"""
This module contains utility functions that are commonly needed in other
qutip modules.
"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['n_thermal', 'clebsch', 'convert_unit', 'iterated_fit']

from typing import Callable, Literal, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.linalg import hankel, lstsq, eigvals, svd, eig
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
        prod *= (i+1)**int(v)
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
    C = np.sqrt((2.0 * j3 + 1.0)*_to_long(c_factor))

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
        rmse1, params = _fit(fun, num_params, xdata, ydata,
                             guesses, lower_repeat,
                             upper_repeat, sigma, maxfev)
        N += 1

    return rmse1, params


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
    return rmse, params


# AAA Fitting
# CHEBFUN attribution for AAA
# Copyright (c) 2017, The Chancellor, Masters and Scholars of the University
# of Oxford, and the Chebfun Developers. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Oxford nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

def aaa(func: Callable[..., complex] | ArrayLike, z: ArrayLike,
        tol: float = 1e-13, max_iter: int = 100) -> dict[str, Any]:
    """
    Computes a rational approximation of the function according to the AAA
    algorithm as explained in [AAA]_ . This
    implementation is a Python adaptation of the Chebfun version in that paper.

    Parameters
    ----------
    func : callable or np.ndarray
        The function to be approximated
    z : np.ndarray
        The sampling points on which to perform the rational approximation.
        Even though linearly spaced sample points will yield good
        approximations, logarithmically spaced points will usually give better
        exponent approximations.
    tol : float, optional
        Relative tolerance of the approximation
    max_iter : int, optional
        Maximum number of support points ~2*n where n is the number of bath
        exponents

    Returns
    -------
    A dictionary containing:
        "function" : callable
            Rational approximation of the function
        "poles" : np.ndarray
            Poles of the approximant function
        "residues" : np.ndarray
            Residues of the approximant function
        "zeros" : np.ndarray
            Zeros of the approximant function
        "errors" : np.ndarray
            Error by iteration
        "rmse" : float
            Normalized root mean squared error from the fit
        "support points" : np.ndarray
            The values used as the support points for the approximation
        "indices" : np.ndarray
            The indices of the support point values
        "indices ordered" : np.ndarray
            The indices of the support point values, sorted by importance
    """
    func = func(z) if callable(func) else func
    indices = np.arange(len(z))
    support_points = np.empty(0, dtype=z.dtype)
    values = np.empty(0, dtype=func.dtype)
    errors = np.zeros(max_iter)
    rational_approx = np.full_like(func, np.mean(func))
    iindices = np.empty(0, dtype=int)
    for k in range(max_iter):
        j = np.argmax(np.abs(func - rational_approx))  # next support index
        iindices = np.append(iindices, j)
        support_points = np.append(support_points, z[j])
        values = np.append(values, func[j])  # function evaluated at support
        indices = indices[indices != j]  # Remaining indices

        cauchy = _compute_cauchy_matrix(z[indices], support_points)
        # Then we construct the Loewner Matrix
        loewner = np.subtract.outer(func[indices], values) * cauchy
        # Perform SVD only d is needed
        _, _, vh = svd(loewner)
        # Compute weights
        weights = vh[-1, :].conj()
        # Obtain the rational Approximation of the function with these support
        # points
        rational_approx = _get_rational_approx(
            cauchy, weights, values, indices, func)
        errors[k] = np.linalg.norm(
            func - rational_approx, np.inf)  # compute error
        if errors[k] <= tol * np.linalg.norm(func, np.inf):
            # if contributions are smaller than the tolerance, then stop the
            # loop
            break
    # Define the function approximation as a callable for output

    def r(z):
        cauchy = _compute_cauchy_matrix(z, support_points)
        r = _get_rational_approx(cauchy, weights, values)
        return r.reshape(z.shape)
    # Obtain poles residues and zeros
    pol, res, zer = _prz(support_points, values, weights)
    rmse = _rmse(r(z), z, func, None)

    return {
        "function": r,
        "poles": pol,
        "residues": res,
        "zeros": zer,
        "errors": errors[:k + 1],
        "rmse": rmse,
        "support points": support_points,
        "values at support": values,
        "indices": indices,
        "indices ordered": iindices
    }


def _compute_cauchy_matrix(z, support_points):
    r"""
    Computes the `Cauchy matrix <https://en.wikipedia.org/wiki/Cauchy_matrix>`
    for the AAA rational approximation

    .. math::

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


def _get_rational_approx(cauchy, weights, values, indices=None, func=None):
    """
    Gets the rational approximation of the function. The approximation is of
    the form

    .. math::

        r(z) = \frac{w_{j} f_{j}}{z-z_{j}}/\frac{w_{j}}{z-z_{j}}

    where w is the cauchy matrix

    Parameters
    ----------
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

    Returns
    -------
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


def _prz(support_points, values, weights):
    r"""
    prz stands for poles, residues and zeros. It calculates and returns the
    poles, residue and zeros of the rational approximation. Using the
    generalized eigenvalue problem

    .. math::

       geig = \begin{pmatrix}0 & \omega_{2} & \dots& \omega_{m} \\
           1& z_{1} & 0 & \dots \\
           1 & 0 & z_{2} & \dots \\
            \vdots   & \vdots & \vdots & \vdots \\
             1   & \dots  & \dots &z_{m}\end{pmatrix} = \lambda L

    where B is like a mxm identity matrix, except its first element is 0.

    Unlike the implementation in the reference we use the `simple quotient
    formula for the residue
    <https://math.stackexchange.com/questions/2202129/
    residue-for-quotient-of-functions>`


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

    cauchy = _compute_cauchy_matrix(pol, support_points)

    numerator = cauchy @ (values * weights)
    denominator = (-cauchy**2) @ weights  # Quotient rule f=1/cauchy
    res = numerator / denominator
    ez = np.block([[0, weights], [values[:, None], np.diag(support_points)]])
    zeros = eig(ez, geye)[0]
    zeros = zeros[~np.isinf(zeros)]
    return pol, res, zeros


# --- Prony methods fitting ---

def _prony_model(n, amp, phase):
    # It serves to compute rmse, a single term of the prony
    # polynomial form [ESPIRAvsESPRIT]_ using phases
    return amp * np.power(phase, np.arange(n))


def prony_methods(method: Literal["prony", "esprit"],
                  signal: ArrayLike, n: int) -> tuple[float, ArrayLike]:
    """
    Estimate amplitudes and frequencies using prony methods.
    Based on the description in [ESPIRAvsESPRIT]_
    and their matlab implementation.

    Parameters
    ----------
    method: str
        The method to obtain the roots of the prony polynomial
        can be prony, and Estimation of signal parameters
        via rotational invariant techniques (ESPRIT)
    signal: n.ndarray
        The input signal (1D complex array).
    n: int
        Desired number of modes to  use as estimation (rank of the signal).

    Returns
    -------
    rmse:
        Normalized mean squared error
    params:
        A list of tuples containing the amplitudes and phases
        of our approximation
    """
    if method != "prony":
        n = len(signal)-n
    hankel0 = hankel(c=signal[:n], r=signal[n - 1: -1])
    hankel1 = hankel(c=signal[1: n + 1], r=signal[n:])
    if method == "prony":
        pencil_matrix = lstsq(hankel0.T, hankel1.T)[0]
        phases = eigvals(pencil_matrix.T)
    elif method == "esprit":
        U1, _, _ = svd(hankel0)
        pencil_matrix = np.linalg.pinv(U1.T @ hankel0) @ (U1.T @ hankel1)
        phases = eigvals(pencil_matrix)
    vandermonte = np.array(
        [[phase**k for phase in phases] for k in range(len(signal))])
    amplitudes = lstsq(vandermonte, signal)[0]
    params = _unpack(
        np.array([val for pair in zip(amplitudes, phases) for val in pair]), 2)

    rmse = _rmse(_prony_model, len(signal), signal, params)
    return rmse, params


# ESPIRA I and II, ESPIRA 2 based on SVD not QR

def espira1(signal: ArrayLike, n: int,
            tol: float = 1e-13) -> tuple[float, ArrayLike]:
    """
    Estimate amplitudes and frequencies using ESPIRA-I.
    Based on the description in [ESPIRAvsESPRIT]_
    and their matlab implementation.

    Parameters
    ----------
    signal: np.ndarray
        The input signal (1D complex array).
    n: int
        Desired number of modes to  use as estimation (rank of the signal).
    tol: float
        Tolerance used in the AAA algorithm. If it is not low enough, the
        desired number of exponents may not be reached, as AAA converges in
        less iterations

    Returns
    -------
    rmse:
        Normalized mean squared error
    params:
        A list of tuples containing the amplitudes and phases
        of our approximation
    """
    # Compute FFT
    F = fft(signal)
    M = len(F)   # lenght of the signal

    # Set knots on the unit circle
    Z = np.exp(2j * np.pi * np.arange(M) / M)
    # Modify the DFT values
    F = F * Z**(-1)
    # Use AAA
    result = aaa(F, Z, max_iter=n+1, tol=tol)  # One extra iteration so n
    # coincides with the number of exponents
    # Construct Cauchy matrix
    CC = (-1) / np.subtract.outer(result['support points'], result['poles'])
    ck, _, _, _ = np.linalg.lstsq(
        CC, result['values at support'], rcond=None)  # Solve by lstsq
    amplitudes = -ck / (1 - result['poles']**M)  # Calculate proper amplitudes

    # pack and calculate goodness of fit
    params = _unpack(
        np.array([val for pair in zip(amplitudes, result['poles'])
                  for val in pair]), 2)
    rmse = _rmse(_prony_model, len(signal), signal, params)

    return rmse, params


def espira2(signal: ArrayLike, n: int,
            tol: float = 1e-13) -> tuple[float, ArrayLike]:
    """
    Estimate amplitudes and frequencies using ESPIRA-II.
    Based on the description in [ESPIRAvsESPRIT]_
    and their matlab implementation

    Parameters
    ----------
    signal: np.ndarray
        The input signal (1D complex array).
    n: int
        Desired number of modes to  use as estimation (rank of the signal).
    tol: float
        Tolerance used in the AAA algorithm. If it is not low enough, the
        desired number of exponents may not be reached, as AAA converges in
        less iterations

    Returns
    -------
    rmse:
        Normalized mean squared error
    params:
        A list of tuples containing the amplitudes and phases
        of our approximation
    """
    # Compute FFT
    F1 = fft(signal)
    M = len(F1)  # lenght of the signal

    # Set knots on the unit circle
    Z = np.exp(2j * np.pi * np.arange(M) / M)
    # Modify the DFT values
    F = F1 * Z**(-1)
    # Run AAA
    result = aaa(F, Z, max_iter=n+1, tol=tol)
    # Use results from AAA to construct lowener and cauchy matrices for the
    # FFT and modified FFT values
    indices = result["indices"]
    support_points = result["support points"]
    values = result['values at support']
    cauchy = _compute_cauchy_matrix(Z[indices], support_points)
    loewner = np.subtract.outer(F[indices], values) * cauchy
    loewner = loewner[::-1].conj()  # invert rows and conjugate
    loewner2 = np.subtract.outer(
        F1[indices], F1[result['indices ordered']]) * cauchy
    loewner2 = loewner2[::-1].conj()  # invert rows and conjugate
    _, N2 = loewner2.shape
    A1 = np.hstack((loewner, loewner2))
    _, _, Vt = np.linalg.svd(A1)
    V = Vt[:n, :]  # Reduce rows in V matrix
    V1 = V[:, :N2]  # First matrix for matrix pencil
    V2 = V[:, N2:2*N2]  # Second matrix for matrix pencil
    # obtain phases
    phases = np.linalg.eigvals(np.linalg.pinv(V1.T) @ V2.T)
    # Initialize Vandermonde matrix
    vd = np.zeros((M, len(phases)), dtype=np.complex128)
    for i in range(len(phases)):
        vd[:, i] = phases[i] ** np.arange(M)
    # Solve for amplitudes, by solving the overdetermined problem with
    # the Vandermonde matrix
    amp = np.linalg.lstsq(vd, signal, rcond=None)[0]
    # calculate and pack stuff similarly to other fitting methods
    params = _unpack(
        np.array([val for pair in zip(amp, phases) for val in pair]), 2)
    rmse = _rmse(_prony_model, len(signal), signal, params)

    return rmse, params
