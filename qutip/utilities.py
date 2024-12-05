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
