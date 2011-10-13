#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>. 
#
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################

#import numpy as np
cimport numpy as np
cimport cython
#from cython.parallel cimport prange

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

from scipy import *

# to be replaced by cython implementation
def cyq_ode_rhs_rho(t, rho, L):
    return L*rho

#def cyq_ode_rhs_rho(float t, np.ndarray[CTYPE_t, ndim=2] rho, np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr):
#    ...




