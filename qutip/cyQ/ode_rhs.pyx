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

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cyq_ode_rhs(float t, np.ndarray[CTYPE_t, ndim=1] rho, np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr):
    cdef int row, jj, row_start, row_end
    cdef int num_rows=len(rho)
    cdef complex dot
    cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)
    for row from 0 <= row < num_rows:
        dot = 0.0
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            dot = dot + data[jj] * rho[idx[jj]]
        out[row,0] = dot
    return out

