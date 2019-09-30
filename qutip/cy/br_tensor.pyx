#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
import warnings
import numpy as np
import qutip.settings as qset
from qutip.qobj import Qobj
cimport numpy as np
cimport cython
from libcpp cimport bool
from qutip.cy.brtools cimport (vec2mat_index, dense_to_eigbasis,
                              ZHEEVR, skew_and_dwmin)
from qutip.cy.brtools import (liou_from_diag_ham, cop_super_term)
from libc.math cimport fabs

include "sparse_routines.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def _br_term(complex[::1,:] A, complex[::1,:] evecs,
                double[:,::1] skew, double dw_min, object spectral,
                unsigned int nrows, int use_secular, double sec_cutoff,
                double atol):

    cdef size_t kk
    cdef size_t I, J # vector index variables
    cdef int[2] ab, cd #matrix indexing variables
    cdef complex[::1,:] A_eig = dense_to_eigbasis(A, evecs, nrows, atol)
    cdef complex elem, ac_elem, bd_elem
    cdef vector[int] coo_rows, coo_cols
    cdef vector[complex] coo_data
    cdef unsigned int nnz
    cdef COO_Matrix coo
    cdef CSR_Matrix csr

    for I in range(nrows**2):
        vec2mat_index(nrows, I, ab)
        for J in range(nrows**2):
            vec2mat_index(nrows, J, cd)

            if (not use_secular) or (fabs(skew[ab[0],ab[1]]-skew[cd[0],cd[1]]) < (dw_min * sec_cutoff)):
                elem = (A_eig[ab[0],cd[0]]*A_eig[cd[1],ab[1]]) * 0.5
                elem *= (spectral(skew[cd[0],ab[0]])+spectral(skew[cd[1],ab[1]]))

                if (ab[0]==cd[0]):
                    ac_elem = 0
                    for kk in range(nrows):
                        ac_elem += A_eig[cd[1],kk]*A_eig[kk,ab[1]] * spectral(skew[cd[1],kk])
                    elem -= 0.5*ac_elem

                if (ab[1]==cd[1]):
                    bd_elem = 0
                    for kk in range(nrows):
                        bd_elem += A_eig[ab[0],kk]*A_eig[kk,cd[0]] * spectral(skew[cd[0],kk])
                    elem -= 0.5*bd_elem

                if (elem != 0):
                    coo_rows.push_back(I)
                    coo_cols.push_back(J)
                    coo_data.push_back(elem)

    PyDataMem_FREE(&A_eig[0,0])

    #Number of elements in BR tensor
    nnz = coo_rows.size()
    coo.nnz = nnz
    coo.rows = coo_rows.data()
    coo.cols = coo_cols.data()
    coo.data = coo_data.data()
    coo.nrows = nrows**2
    coo.ncols = nrows**2
    coo.is_set = 1
    coo.max_length = nnz
    COO_to_CSR(&csr, &coo)
    return CSR_to_scipy(&csr)


@cython.boundscheck(False)
@cython.wraparound(False)
def bloch_redfield_tensor(object H, list a_ops, spectra_cb=None,
                 list c_ops=[], bool use_secular=True,
                 double sec_cutoff=0.1,
                 double atol = qset.atol):
     """
     Calculates the time-independent Bloch-Redfield tensor for a system given
     a set of operators and corresponding spectral functions that describes the
     system's couplingto its environment.

     Parameters
     ----------

     H : :class:`qutip.qobj`
         System Hamiltonian.

     a_ops : list
         Nested list of system operators that couple to the environment,
         and the corresponding bath spectra represented as Python
         functions.

     spectra_cb : list
         Depreciated.

     c_ops : list
         List of system collapse operators.

     use_secular : bool {True, False}
         Flag that indicates if the secular approximation should
         be used.

     sec_cutoff : float {0.1}
         Threshold for secular approximation.

    atol : float {qutip.settings.atol}
        Threshold for removing small parameters.

     Returns
     -------

     R, kets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

         R is the Bloch-Redfield tensor and kets is a list eigenstates of the
         Hamiltonian.

     """
     cdef list _a_ops
     cdef object a, cop, L
     cdef int K, kk
     cdef int nrows = H.shape[0]
     cdef list op_dims = H.dims
     cdef list sop_dims = [[op_dims[0], op_dims[0]], [op_dims[1], op_dims[1]]]
     cdef list ekets, ket_dims

     ket_dims = [op_dims[0], [1] * len(op_dims[0])]

     if not (spectra_cb is None):
         warnings.warn("The use of spectra_cb is depreciated.", DeprecationWarning)
         _a_ops = []
         for kk, a in enumerate(a_ops):
             _a_ops.append([a,spectra_cb[kk]])
         a_ops = _a_ops

     K = len(a_ops)

     # Sanity checks for input parameters
     if not isinstance(H, Qobj):
         raise TypeError("H must be an instance of Qobj")

     for a in a_ops:
         if not isinstance(a[0], Qobj) or not a[0].isherm:
             raise TypeError("Operators in a_ops must be Hermitian Qobj.")

     cdef complex[::1,:] H0 = H.full('F')
     cdef complex[::1,:] evecs = np.zeros((nrows,nrows), dtype=complex, order='F')
     cdef double[::1] evals = np.zeros(nrows, dtype=float)

     ZHEEVR(H0, &evals[0], evecs, nrows)
     L = liou_from_diag_ham(evals)

     for cop in c_ops:
         L = L + cop_super_term(cop.full('F'), evecs, 1,
                               nrows, atol)

     #only lindblad collapse terms
     if K == 0:
         ekets = [Qobj(np.asarray(evecs[:,k]),
               dims=ket_dims) for k in range(nrows)]

         return Qobj(L, dims=sop_dims, copy=False), ekets

     #has some br operators and spectra
     cdef double[:,::1] skew = np.zeros((nrows,nrows), dtype=float)
     cdef double dw_min = skew_and_dwmin(&evals[0], skew, nrows)

     for a in a_ops:
         L = L + _br_term(a[0].full('F'), evecs, skew, dw_min, a[1],
                        nrows, use_secular, sec_cutoff, atol)

     ekets = [Qobj(np.asarray(evecs[:,k]),
                   dims=ket_dims) for k in range(nrows)]


     return Qobj(L, dims=sop_dims, copy=False), ekets
