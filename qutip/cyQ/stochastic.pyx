# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np
cimport numpy as np
cimport cython
cimport libc.math

from qutip.cyQ.spmatfuncs cimport spmv

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_rhs_psi_deterministic(object H,
                               np.ndarray[CTYPE_t, ndim=1] state,
                               double t,
                               double dt,
                               object args):
    """
    Deterministic contribution to the density matrix change; cython
    implementation.
    """
    cdef np.ndarray[CTYPE_t, ndim=1] dpsi_t 

    dpsi_t = (-1.0j * dt) * spmv(H.data, H.indices, H.indptr, state)

    return dpsi_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[CTYPE_t, ndim=1] cy_rhs_rho_deterministic(object L,
                               np.ndarray[CTYPE_t, ndim=1] rho_t,
                               double t,
                               double dt,
                               object args):
    """
    Deterministic contribution to the density matrix change; cython
    implementation.
    """
    cdef np.ndarray[CTYPE_t, ndim=1] drho_t

    drho_t = spmv(L.data, L.indices, L.indptr, rho_t) * dt

    return drho_t



