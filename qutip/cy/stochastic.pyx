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
import numpy as np
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.spmatfuncs cimport (spmv_csr,
                                  cy_expect_rho_vec_csr, cy_expect_psi_csr)

include "parameters.pxi"

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
cpdef cy_d1_psi_photocurrent(double t, np.ndarray[CTYPE_t, ndim=1] psi,
                             object A, object args):
    """
    Cython version of d1_psi_photocurrent. See d1_psi_photocurrent for docs.
    """
    a0 = A[0]
    a3 = A[3]
    return (-0.5 * (spmv_csr(a3.data, a3.indices, a3.indptr, psi)
            - psi * np.linalg.norm(spmv_csr(a0.data, a0.indices, a0.indptr, psi)) ** 2))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_d2_psi_photocurrent(double t, np.ndarray[CTYPE_t, ndim=1] psi,
                             object A, object args):
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
cpdef d1_psi_homodyne(t, psi, A, args):
    """
    Cython version of d2_psi_homodyne. See d2_psi_homodyne for docs.
    """
    e1 = cy_expect_psi_csr(A[1].data, A[1].indices, A[1].indptr, psi, 0)
    return 0.5 * (e1 * spmv_csr(A[0].data, A[0].indices, A[0].indptr, psi) -
                  spmv_csr(A[3].data, A[3].indices, A[3].indptr, psi) -
                  0.25 * e1 ** 2 * psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef d2_psi_homodyne(t, psi, A, args):
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
cpdef cy_d1_rho_photocurrent(double t, np.ndarray[CTYPE_t, ndim=1] rho_vec,
                             object A, object args):
    """
    Photo-current D1 function
    """
    n = A[4] + A[5]
    e1 = cy_expect_rho_vec_csr(n.data, n.indices, n.indptr, rho_vec, 0)
    return 0.5 * (e1 * rho_vec - spmv_csr(n.data, n.indices, n.indptr, rho_vec))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_d2_rho_photocurrent(double t, np.ndarray[CTYPE_t, ndim=1] rho_vec,
                             object A, object args):
    """
    Photo-current D2 function
    """
    a = A[6]
    e1 = cy_expect_rho_vec_csr(a.data, a.indices, a.indptr, rho_vec, 0)
    if e1.real > 1e-12:
        return [spmv_csr(a.data, a.indices, a.indptr, rho_vec) / e1 - rho_vec]
    else:
        return [- rho_vec]


