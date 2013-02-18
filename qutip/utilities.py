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

import numpy


def n_thermal(w, w_th):
    """
    Return the number of photons in thermal equilibrium for an harmonic 
    oscillator mode with frequency 'w', at the temperature described by
    'w_th' where :math:`\\omega_{\\rm th} = k_BT/\\hbar`.
    """
    if (w_th > 0) and exp(w / w_th) != 1.0:
        return 1.0 / (exp(w / w_th) - 1.0)
    else:
        return 0.0 * numpy.ones(numpy.shape(w))
