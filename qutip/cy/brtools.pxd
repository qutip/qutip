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
cimport numpy as np

#Spectral function with signature (w,t)
ctypedef complex (*spec_func)(double, double)

cdef complex[::1,:] farray_alloc(int nrows)

cpdef void dense_add_mult(complex[::1,:] A, complex[::1,:] B,
                  double complex alpha) nogil

cdef void ZHEEVR(complex[::1,:] H, double * eigvals,
                complex[::1,:] Z, int nrows)

cdef complex[::1,:] dense_to_eigbasis(complex[::1,:] A, complex[::1,:] evecs,
                                    unsigned int nrows, double atol)

cdef void diag_liou_mult(double * diags, double complex * vec,
                        double complex * out, unsigned int nrows) nogil

cdef double complex * vec_to_eigbasis(complex[::1] vec, complex[::1,:] evecs,
                                    unsigned int nrows)

cdef np.ndarray[complex, ndim=1, mode='c'] vec_to_fockbasis(double complex * eig_vec,
                                                complex[::1,:] evecs,
                                                unsigned int nrows)

cdef void cop_super_mult(complex[::1,:] cop, complex[::1,:] evecs,  double complex * vec,
                    double complex alpha,
                    double complex * out,
                    unsigned int nrows,
                    double atol)

cdef void vec2mat_index(int nrows, int index, int[2] out) nogil

cdef double skew_and_dwmin(double * evals, double[:,::1] skew,
                                unsigned int nrows) nogil


cdef void br_term_mult(double t, complex[::1,:] A, complex[::1,:] evecs,
                double[:,::1] skew, double dw_min, spec_func spectral,
                double complex * vec, double complex * out,
                unsigned int nrows, int use_secular, double sec_cutoff,
                double atol)
