# This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
This module contains utility functions that are commonly needed in other
qutip modules.
"""

import numpy as np

try:  # for scipy v <= 0.90
    from scipy import factorial
except:  # for scipy v >= 0.10
    from scipy.misc import factorial


def n_thermal(w, w_th):
    """
    Return the number of photons in thermal equilibrium for an harmonic
    oscillator mode with frequency 'w', at the temperature described by
    'w_th' where :math:`\\omega_{\\rm th} = k_BT/\\hbar`.

    Parameters
    ----------

    w : *float* or *array*
        Frequency of the oscillator.

    w_th : *float*
        The temperature in units of frequency (or the same units as `w`).


    Returns
    -------

    n_avg : *float* or *array*

        Return the number of average photons in thermal equilibrium for a
        an oscillator with the given frequency and temperature.


    """

    if type(w) is np.ndarray:
        return 1.0 / (np.exp(w / w_th) - 1.0)

    else:
        if (w_th > 0) and np.exp(w / w_th) != 1.0:
            return 1.0 / (np.exp(w / w_th) - 1.0)
        else:
            return 0.0


def linspace_with(start, stop, num=50, elems=[]):
    """
    Return an array of numbers sampled over specified interval
    with additional elements added.

    Returns `num` spaced array with elements from `elems` inserted
    if not already included in set.

    Returned sample array is not evenly spaced if addtional elements
    are added.

    Parameters
    ----------
    start : int
        The starting value of the sequence.
    stop : int
        The stoping values of the sequence.
    num : int, optional
        Number of samples to generate.
    elems : list/ndarray, optional
        Requested elements to include in array

    Returns
    -------
    samples : ndadrray
        Original equally spaced sample array with additional
        elements added.
    """
    elems = np.array(elems)
    lspace = np.linspace(start, stop, num)
    return np.union1d(lspace, elems)


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

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
                factorial(j3 + m3) * factorial(j3 - m3) /
               (factorial(j1 + j2 + j3 + 1) *
                factorial(j1 - m1) * factorial(j1 + m1) *
                factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C



#------------------------------------------------------------------------------
# Functions for unit conversions
#
_e = 1.602176565e-19  # C
_kB = 1.3806488e-23   # J/K
_h = 6.62606957e-34   # Js

_unit_factor_tbl = {
#   "unit": "factor that convert argument from unit 'unit' to Joule"
    "J" :   1.0,
    "eV" :  _e,
    "meV" : 1.0e-3 * _e,
    "GHz" : 1.0e9 * _h,
    "mK" :  1.0e-3 * _kB,
}

def convert_unit(value, orig="meV", to="GHz"):
    """
    Convert an energy from unit `orig` to unit `to`.

    Parameters
    ----------
    value : float / array
        The energy in the old unit.

    orig : string
        The name of the original unit ("J", "eV", "meV", "GHz", "mK")

    to : string
        The name of the new unit ("J", "eV", "meV", "GHz", "mK")
 
    Returns
    -------
    value_new_unit : float / array
        The energy in the new unit.
    """
    if not orig in _unit_factor_tbl:
        raise TypeError("Unsupported unit %s" % orig)

    if not to in _unit_factor_tbl:
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
    w_mK = w * 1.0e12 * (_h/_kB)
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
    w_GHz = w * 1.0e-12 * (_kB/_h)
    return w_GHz



