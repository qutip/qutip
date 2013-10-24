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

from qutip.cyQ.spmatfuncs cimport (spmv_csr,
                                   cy_expect_rho_vec_csr, cy_expect_psi_csr)

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


# -----------------------------------------------------------------------------
# Wave function d1 and d2 functions
#

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_d1_psi_photocurrent(object A, np.ndarray[CTYPE_t, ndim=1] psi):
    """
    Cython version of d1_psi_photocurrent. See d1_psi_photocurrent for docs.
    """
    a0 = A[0]
    a3 = A[3]
    return (-0.5 * (spmv_csr(a3.data, a3.indices, a3.indptr, psi)
            - psi * np.linalg.norm(spmv_csr(a0.data, a0.indices, a0.indptr, psi)) ** 2))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_d2_psi_photocurrent(object A, np.ndarray[CTYPE_t, ndim=1] psi):
    """
    Cython version of d2_psi_photocurrent. See d2_psi_photocurrent for docs.
    """
    cdef np.ndarray[CTYPE_t, ndim=1] psi1 

    psi1 = spmv_csr(A[0].data, A[0].indices, A[0].indptr, psi)
    n1 = np.linalg.norm(psi1)
    if n1 != 0:
        return psi1 / n1 - psi
    else:
        return - psi


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef d1_psi_homodyne(A, psi):
    """
    Cython version of d2_psi_homodyne. See d2_psi_homodyne for docs.
    """
    e1 = cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    return 0.5 * (e1 * spmv_csr(A[0].data, A[0].indices, A[0].indptr, psi) -
                  spmv_csr(A[3].data, A[3].indices, A[3].indptr, psi) -
                  0.25 * e1 ** 2 * psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef d2_psi_homodyne(A, psi):
    """
    Cython version of d2_psi_homodyne. See d2_psi_homodyne for docs.
    """
    e1 = cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    return [spmv_csr(A[0].data, A[0].indices, A[0].indptr, psi) - 0.5 * e1 * psi]

# -----------------------------------------------------------------------------
# density matric (vector form) d1 and d2 functions
#

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_d1_rho_photocurrent(object A, np.ndarray[CTYPE_t, ndim=1] rho_vec):
    """
    Photo-current D1 function
    """
    n = A[4] + A[5]
    e1 = cy_expect_rho_vec_csr(n.data, n.indices, n.indptr, rho_vec, 0)
    return 0.5 * (e1 * rho_vec - spmv_csr(n.data, n.indices, n.indptr, rho_vec))


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


