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

cimport numpy as np
cimport cython

include "parameters.pxi"

cpdef np.ndarray[CTYPE_t, ndim=1, mode="c"] spmv_csr(
    np.ndarray[CTYPE_t, ndim=1, mode="c"] data,
    np.ndarray[ITYPE_t, ndim=1, mode="c"] idx,
    np.ndarray[ITYPE_t, ndim=1, mode="c"] ptr,
    np.ndarray[CTYPE_t, ndim=1, mode="c"] vec)

cpdef cy_expect_rho_vec_csr(np.ndarray[CTYPE_t, ndim=1, mode="c"] data,
                            np.ndarray[ITYPE_t, ndim=1, mode="c"] idx,
                            np.ndarray[ITYPE_t, ndim=1, mode="c"] ptr,
                            np.ndarray[CTYPE_t, ndim=1, mode="c"] rho_vec,
                            int herm)

cpdef cy_expect_psi(object op,
                    np.ndarray[CTYPE_t, ndim=1, mode="c"] state,
                    int isherm)

cpdef cy_expect_psi_csr(np.ndarray[CTYPE_t, ndim=1, mode="c"] data,
                        np.ndarray[ITYPE_t, ndim=1, mode="c"] idx,
                        np.ndarray[ITYPE_t, ndim=1, mode="c"] ptr, 
                        np.ndarray[CTYPE_t, ndim=1, mode="c"] state,
                        int isherm)
