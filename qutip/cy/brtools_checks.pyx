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
import numpy as np
cimport numpy as np
cimport cython
import qutip.settings as qset
from qutip.cy.brtools cimport (ZHEEVR, diag_liou_mult, dense_to_eigbasis,
                            vec_to_eigbasis, vec_to_fockbasis,
                            cop_super_mult, br_term_mult, skew_and_dwmin)

include "sparse_routines.pxi"

@cython.boundscheck(False)
def _test_zheevr(complex[::1,:] H, double[::1] evals):
    cdef np.ndarray[complex, ndim=2, mode='fortran'] Z = np.zeros((H.shape[0],H.shape[0]),dtype=complex, order='f')
    ZHEEVR(H, &evals[0], Z, H.shape[0])
    return Z


@cython.boundscheck(False)
def _test_diag_liou_mult(double[::1] evals, complex[::1] vec,
                            complex[::1] out, int nrows):
    diag_liou_mult(&evals[0], &vec[0], &out[0], nrows)


@cython.boundscheck(False)
def _test_dense_to_eigbasis(complex[::1,:] A, complex[::1,:] evecs,
                        unsigned int nrows, double atol):
    cdef complex[::1,:] out = dense_to_eigbasis(A, evecs, nrows, atol)
    cdef np.ndarray[complex, ndim=2] mat
    cdef np.npy_intp dims[2]
    dims[:] = [A.shape[0], A.shape[1]]
    #We cannot build simple in fortran-order, so build c-order and return transpose
    mat = np.PyArray_SimpleNewFromData(2, <np.npy_intp *>dims, np.NPY_COMPLEX128, &out[0,0])
    PyArray_ENABLEFLAGS(mat, np.NPY_OWNDATA)
    return mat.T


@cython.boundscheck(False)
def _test_vec_to_eigbasis(complex[::1,:] H, complex[::1] vec):
    cdef np.ndarray[complex, ndim=2, mode='fortran'] Z = np.zeros((H.shape[0],H.shape[0]),
                                                                  dtype=complex, order='f')
    cdef double[::1] evals = np.zeros(H.shape[0],dtype=float)
    ZHEEVR(H, &evals[0], Z, H.shape[0])

    cdef double complex * eig_vec = vec_to_eigbasis(vec,Z, H.shape[0])
    cdef np.npy_intp dim = H.shape[0]**2
    cdef np.ndarray[complex, ndim=1, mode='c'] out
    out = np.PyArray_SimpleNewFromData(1, &dim, np.NPY_COMPLEX128, eig_vec)
    PyArray_ENABLEFLAGS(out, np.NPY_OWNDATA)
    return out


@cython.boundscheck(False)
def _test_eigvec_to_fockbasis(complex[::1] eig_vec, complex[::1,:] evecs, int nrows):
    cdef np.ndarray[complex, ndim=1, mode='c'] out
    out = vec_to_fockbasis(&eig_vec[0], evecs, nrows)
    return out


@cython.boundscheck(False)
def _test_vector_roundtrip(complex[::1,:] H, complex[::1] vec):
    cdef np.ndarray[complex, ndim=2, mode='fortran'] Z = np.zeros((H.shape[0],H.shape[0]),
                                                                  dtype=complex, order='f')
    cdef double[::1] evals = np.zeros(H.shape[0],dtype=float)
    ZHEEVR(H, &evals[0], Z, H.shape[0])
    cdef double complex * eig_vec = vec_to_eigbasis(vec, Z, H.shape[0])
    cdef np.ndarray[complex, ndim=1, mode='c'] out
    out = vec_to_fockbasis(eig_vec, Z, H.shape[0])
    PyDataMem_FREE(eig_vec)
    return out

@cython.boundscheck(False)
def _cop_super_mult(complex[::1,:] cop, complex[::1,:] evecs, complex[::1] vec,
                    double complex alpha,
                    complex[::1] out,
                    unsigned int nrows,
                    double atol):
    cop_super_mult(cop, evecs, &vec[0], alpha, &out[0], nrows, atol)


#Test spectral function
cdef complex spectral(double w, double t): return 1.0

def _test_br_term_mult(double t, complex[::1,:] A, complex[::1, :] evecs,
            double[::1] evals, complex[::1] vec, complex[::1] out,
            int use_secular, double sec_cutoff, double atol):

    cdef unsigned int nrows = A.shape[0]
    cdef double * _temp = <double *>PyDataMem_NEW((nrows**2) * sizeof(double))
    cdef double[:,::1] skew = <double[:nrows,:nrows]> _temp
    cdef double dw_min = skew_and_dwmin(&evals[0], skew, nrows)
    br_term_mult(t, A, evecs, skew, dw_min, spectral, &vec[0], &out[0],
                nrows, use_secular, sec_cutoff, atol)
    PyDataMem_FREE(&skew[0,0])
