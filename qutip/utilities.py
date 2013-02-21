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

def linspace_with(start,stop,num=50,elems=[]):
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
    elems=np.array(elems)
    lspace=np.linspace(start,stop,num)
    return np.union1d(lspace,elems)
    
