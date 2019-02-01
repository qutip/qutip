#!python
#cython: language_level=3
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

cimport numpy as cnp
cimport cython
from libcpp cimport bool

include "parameters.pxi"

cpdef cnp.ndarray[CTYPE_t, ndim=1, mode="c"] spmv_csr(complex[::1] data,
                int[::1] ind, int[::1] ptr, complex[::1] vec)


cdef void spmvpy(complex * data,
                int * ind,
                int *  ptr,
                complex * vec,
                complex a,
                complex * out,
                unsigned int nrows)


cpdef cy_expect_rho_vec_csr(complex[::1] data,
                            int[::1] idx,
                            int[::1] ptr,
                            complex[::1] rho_vec,
                            int herm)


cpdef cy_expect_psi(object A,
                    complex[::1] vec,
                    bool isherm)


cpdef cy_expect_psi_csr(complex[::1] data,
                        int[::1] ind,
                        int[::1] ptr,
                        complex[::1] vec,
                        bool isherm)


cdef void _spmm_c_py(complex * data,
                     int * ind,
                     int * ptr,
                     complex * mat,
                     complex a,
                     complex * out,
                     unsigned int sp_rows,
                     unsigned int nrows,
                     unsigned int ncols)

cpdef void spmmpy_c(complex[::1] data,
                    int[::1] ind,
                    int[::1] ptr,
                    complex[:,::1] M,
                    complex a,
                    complex[:,::1] out)

cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmmc(object sparse,
                                                   complex[:,::1] mat)

cdef void _spmm_f_py(complex * data,
                     int * ind,
                     int * ptr,
                     complex * mat,
                     complex a,
                     complex * out,
                     unsigned int sp_rows,
                     unsigned int nrows,
                     unsigned int ncols)

cpdef void spmmpy_f(complex[::1] data,
                    int[::1] ind,
                    int[::1] ptr,
                    complex[::1,:] mat,
                    complex a,
                    complex[::1,:] out)

cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmmf(object sparse,
                                                   complex[::1,:] mat)

cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmm(object sparse,
                                            cnp.ndarray[complex, ndim=2] mat)
