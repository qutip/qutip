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

from qutip.cyQ.spmatfuncs cimport spmv_csr, cy_expect_rho_vec_csr

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

    dpsi_t = (-1.0j * dt) * spmv_csr(H.data, H.indices, H.indptr, state)

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
    return spmv_csr(L.data, L.indices, L.indptr, rho_t) * dt


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_d1_rho_photocurrent(object A, np.ndarray[CTYPE_t, ndim=1] rho_vec):
    """
    Photo-current D1 function
    """
    n = A[4] + A[5]
    e1 = cy_expect_rho_vec_csr(n.data, n.indices, n.indptr, rho_vec, 0)
    return e1 * rho_vec - spmv_csr(n.data, n.indices, n.indptr, rho_vec)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_d2_rho_photocurrent(object A, np.ndarray[CTYPE_t, ndim=1] rho_vec):
    """
    Photo-current D2 function
    """
    a = A[6]
    e1 = cy_expect_rho_vec_csr(a.data, a.indices, a.indptr, rho_vec, 0)
    if e1.real > 1e-12:
        return [spmv_csr(a.data, a.indices, a.indptr, rho_vec) / e1 - rho_vec]
    else:
        return [- rho_vec]

