# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

"""
This module contains utility functions that are commonly needed in other
qutip modules.
"""

__all__ = ['n_thermal', 'linspace_with', 'clebsch', 'convert_unit',
           'view_methods']

import numpy as np

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
    from scipy.special import factorial

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

    orig : string
        The name of the original unit ("J", "eV", "meV", "GHz", "mK")

    to : string
        The name of the new unit ("J", "eV", "meV", "GHz", "mK")

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


def view_methods(Q):
    """
    View the methods and corresponding doc strings
    for a Qobj class.

    Parameters
    ----------
    Q : Qobj
        Input Quantum object.

    """
    meth = dir(Q)
    qobj_props = ['data', 'dims', 'isherm', 'shape', 'type']
    pub_meth = [x for x in meth if x.find('_') and x not in qobj_props]
    ml = max([len(x) for x in pub_meth])
    nl = len(Q.__class__.__name__ + 'Class Methods:')
    print(Q.__class__.__name__ + ' Class Methods:')
    print('-' * nl)
    for ii in range(len(pub_meth)):
        m = getattr(Q, pub_meth[ii])
        meth_str = m.__doc__
        ind = meth_str.find('\n')
        pub_len = len(pub_meth[ii] + ': ')
        print(pub_meth[ii] + ':' + ' ' * (ml+3-pub_len) + meth_str[:ind])


def _version2int(version_string):
    str_list = version_string.split(
        "-dev")[0].split("rc")[0].split("a")[0].split("b")[0].split(
        "post")[0].split('.')
    return sum([int(d if len(d) > 0 else 0) * (100 ** (3 - n))
                for n, d in enumerate(str_list[:3])])


def _blas_info():
    config = np.__config__
    blas_info = config.blas_opt_info
    _has_lib_key = 'libraries' in blas_info.keys()
    blas = None
    if hasattr(config,'mkl_info') or \
            (_has_lib_key and any('mkl' in lib for lib in blas_info['libraries'])):
        blas = 'INTEL MKL'
    elif hasattr(config,'openblas_info') or \
            (_has_lib_key and any('openblas' in lib for lib in blas_info['libraries'])):
        blas = 'OPENBLAS'
    elif 'extra_link_args' in blas_info.keys() and ('-Wl,Accelerate' in blas_info['extra_link_args']):
        blas = 'Accelerate'
    else:
        blas = 'Generic'
    return blas
